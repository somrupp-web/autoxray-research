"""
AutoXray Research Monitor
Run:  pip install flask paramiko
Then: python app.py
Open: http://localhost:7860
"""

import json
import re
import time
from flask import Flask, Response, jsonify, render_template_string, request
import paramiko

# Strips ANSI/VT100 escape sequences (colors, cursor moves) from terminal output
_ANSI_RE = re.compile(r'\x1b(?:\[[0-9;]*[A-Za-z]|[()][0-9A-Za-z]|[^[])|'
                      r'\r|\x0f|\x0e')

app = Flask(__name__)

# ── Cluster config ─────────────────────────────────────────────────────────
NODES = [
    {"id": "188",   "ip": "10.137.203.188", "name": "spark-188 (reference)"},
    {"id": "node0", "ip": "10.137.203.228", "name": "node-0 (spark-ccb0)"},
    {"id": "node1", "ip": "10.137.203.184", "name": "node-1 (spark-0c01)"},
    {"id": "node2", "ip": "10.137.203.174", "name": "node-2 (spark-1aa0)"},
    {"id": "node3", "ip": "10.137.203.177", "name": "node-3 (spark-1b93)"},
]
SSH_USER = "nvidia"
SSH_PASS = "nvidia"
REPO     = "/home/nvidia/autoresearch"
_UV_BIN  = "/home/nvidia/.local/bin/uv"


# ── SSH helpers ────────────────────────────────────────────────────────────
def _ssh(ip, timeout=8):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(ip, username=SSH_USER, password=SSH_PASS,
              timeout=timeout, banner_timeout=timeout)
    return c

def ssh_run(ip, cmd, timeout=10):
    try:
        c = _ssh(ip, timeout)
        _, out, _ = c.exec_command(cmd)
        result = out.read().decode(errors="replace")
        c.close()
        return result, None
    except Exception as e:
        return None, str(e)

def _ssh_exec(ip, cmd, timeout=30):
    """Like ssh_run but captures stderr too (useful for debugging Python errors)."""
    try:
        c = _ssh(ip, timeout)
        _, stdout, stderr = c.exec_command(cmd)
        out = stdout.read().decode(errors="replace").strip()
        err = stderr.read().decode(errors="replace").strip()
        c.close()
        return out or None, err or None
    except Exception as e:
        return None, str(e)

def get_node(node_id):
    return next((n for n in NODES if n["id"] == node_id), None)


# ── SFTP file upload ───────────────────────────────────────────────────────
import os as _os

_REPO_LOCAL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

def sftp_put(ip, local_path, remote_path, timeout=20):
    """Upload a single file to the remote machine via SFTP."""
    try:
        c = _ssh(ip, timeout=timeout)
        sftp = c.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()
        c.close()
        return True
    except Exception as e:
        app.logger.warning(f"sftp_put {local_path} -> {remote_path}: {e}")
        return False


def sync_scripts_to_remote(ip):
    """Push loop.sh, test_inference.py and train.py (as baseline) to the remote.
    This bypasses git-remote issues — the remote may point at karpathy/autoresearch,
    not our fork, so git pull would pull the wrong (GPT demo) scripts."""
    results = {}
    for fname, remote_name in [
        ("loop.sh",          "loop.sh"),
        ("test_inference.py","test_inference.py"),
        ("train.py",         "train.py.baseline"),   # stored as baseline, not overwriting live file
    ]:
        local  = _os.path.join(_REPO_LOCAL, fname)
        remote = f"{REPO}/{remote_name}"
        results[fname] = sftp_put(ip, local, remote)
    return results


# ── Status ─────────────────────────────────────────────────────────────────
@app.route("/api/nodes")
def api_nodes():
    return jsonify(NODES)

@app.route("/api/status/<node_id>")
def api_status(node_id):
    node = get_node(node_id)
    if not node:
        return jsonify({"error": "not found"}), 404
    # PID-file approach: written at Start, removed at Stop — survives SSH timeouts
    out, err = ssh_run(node["ip"],
        "curl -sf http://127.0.0.1:8080/v1/models >/dev/null 2>&1 && echo vllm_ready || echo vllm_down;"
        f"PID=$(cat {REPO}/loop.pid 2>/dev/null);"
        f" if [ -n \"$PID\" ] && kill -0 \"$PID\" 2>/dev/null; then echo loop_active; else echo loop_idle; fi;"
        f"cd {REPO} && git log --oneline 2>/dev/null | wc -l;"
        f"grep -c 'keep' {REPO}/results.tsv 2>/dev/null || echo 0",
        timeout=12)
    if err:
        return jsonify({"online": False, "vllm": "offline", "loop": "stopped",
                        "commits": 0, "kept": 0})
    lines = (out or "").splitlines()
    return jsonify({
        "online":  True,
        "vllm":    "ready" if any("vllm_ready" in l for l in lines) else "loading",
        "loop":    "running" if any("loop_active" in l for l in lines) else "stopped",
        "commits": next((l for l in lines if l.strip().isdigit() and int(l) > 0), "0"),
        "kept":    lines[-1] if lines else "0",
    })


# ── Code diff stream (polls train.py every 3s, emits diff on change) ──────
@app.route("/api/code-stream/<node_id>")
def code_stream(node_id):
    node = get_node(node_id)
    if not node:
        return Response("not found", status=404)

    def generate(ip):
        ssh = None
        last_hash = None
        try:
            while True:
                try:
                    if ssh is None or not ssh.get_transport() or \
                            not ssh.get_transport().is_active():
                        if ssh:
                            try: ssh.close()
                            except: pass
                        ssh = _ssh(ip)

                    _, out, _ = ssh.exec_command(
                        f"md5sum {REPO}/train.py 2>/dev/null | cut -d' ' -f1")
                    current_hash = out.read().decode().strip()

                    if current_hash and current_hash != last_hash:
                        # Diff vs train.py.baseline (SFTP-uploaded DenseNet version).
                        # Empty when train.py == baseline; shows OpenCode additions otherwise.
                        # Falls back to git diff if no baseline file exists.
                        _, diff_out, _ = ssh.exec_command(
                            f"if [ -f {REPO}/train.py.baseline ]; then "
                            f"  diff -u {REPO}/train.py.baseline {REPO}/train.py 2>/dev/null || true; "
                            f"else "
                            f"  cd {REPO} && ORIG=$(git log --oneline -- train.py | tail -1 | awk '{{print $1}}') && "
                            f"  git diff $ORIG -- train.py 2>/dev/null; "
                            f"fi")
                        diff = diff_out.read().decode(errors="replace").strip()

                        # Also get the full file
                        _, full_out, _ = ssh.exec_command(
                            f"cat {REPO}/train.py 2>/dev/null")
                        full = full_out.read().decode(errors="replace")

                        payload = {
                            "hash": current_hash,
                            "diff": diff,
                            "full": full,
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                        last_hash = current_hash
                    else:
                        yield ": keepalive\n\n"

                    time.sleep(3)

                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    ssh = None
                    time.sleep(5)

        except GeneratorExit:
            pass
        finally:
            if ssh:
                try: ssh.close()
                except: pass

    return Response(
        generate(node["ip"]),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Raw loop log stream ────────────────────────────────────────────────────
@app.route("/api/log-stream/<node_id>")
def log_stream(node_id):
    node = get_node(node_id)
    if not node:
        return Response("not found", status=404)

    def generate(ip):
        ssh = None
        try:
            ssh = _ssh(ip, timeout=10)
            ch = ssh.get_transport().open_session()
            ch.set_combine_stderr(True)
            cmd = (
                f"LOG={REPO}/loop_node0.log;"
                # wait for file to appear, then follow by name (-F) so
                # deletion+recreation on each Start Training is handled correctly
                f" while ! [ -f \"$LOG\" ]; do sleep 1; done;"
                f" exec tail -F -n 0 \"$LOG\""
            )
            ch.exec_command(cmd)
            buf = b""
            while True:
                if ch.recv_ready():
                    buf += ch.recv(8192)
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        text = _ANSI_RE.sub('', line.decode(errors="replace")).rstrip()
                        if text:
                            yield f"data: {json.dumps(text)}\n\n"
                elif ch.exit_status_ready():
                    break
                else:
                    yield ": keepalive\n\n"
                    time.sleep(0.3)
        except GeneratorExit:
            pass
        except Exception as e:
            yield f"data: {json.dumps(f'[error] {e}')}\n\n"
        finally:
            if ssh:
                try: ssh.close()
                except: pass

    return Response(generate(node["ip"]), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Test inference results ────────────────────────────────────────────────
@app.route("/api/inference/<node_id>")
def api_inference(node_id):
    node = get_node(node_id)
    if not node:
        return jsonify({"error": "not found"}), 404
    out, err = ssh_run(node["ip"], f"cat {REPO}/test_inference_results.json 2>/dev/null")
    if not out or not out.strip():
        return jsonify({"error": "no results yet"})
    try:
        return jsonify(json.loads(out))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/inference-history/<node_id>")
def api_inference_history(node_id):
    node = get_node(node_id)
    if not node:
        return jsonify({"error": "not found"}), 404
    out, err = ssh_run(node["ip"], f"cat {REPO}/test_inference_history.json 2>/dev/null")
    if not out or not out.strip():
        return jsonify([])
    try:
        data = json.loads(out)
        return jsonify(data if isinstance(data, list) else [])
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/xray-image/<node_id>")
def api_xray_image(node_id):
    import base64
    node = get_node(node_id)
    if not node:
        return jsonify({"error": "not found"}), 404
    out, err = ssh_run(node["ip"],
        f"test -f {REPO}/test_xray.png && base64 -w0 {REPO}/test_xray.png || echo ''",
        timeout=15)
    if not out or not out.strip():
        return jsonify({"error": "no image yet"})
    return jsonify({"data": out.strip()})


# ── Current train.py content ──────────────────────────────────────────────
@app.route("/api/trainpy/<node_id>")
def api_trainpy(node_id):
    node = get_node(node_id)
    if not node:
        return jsonify({"error": "not found"}), 404
    out, err = ssh_run(node["ip"], f"cat {REPO}/train.py 2>/dev/null")
    return jsonify({"content": out or ""})


# ── GPU stats ─────────────────────────────────────────────────────────────
@app.route("/api/gpu/<node_id>")
def api_gpu(node_id):
    node = get_node(node_id)
    if not node:
        return jsonify({"error": "not found"}), 404
    out, err = ssh_run(node["ip"],
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu "
        "--format=csv,noheader,nounits 2>/dev/null | head -1",
        timeout=6)
    if not out or not out.strip():
        return jsonify({})
    parts = [p.strip() for p in out.strip().split(',')]
    def safe(x):
        try: return float(x)
        except: return None
    return jsonify({
        "util":      safe(parts[0]) if len(parts) > 0 else None,
        "mem_used":  safe(parts[1]) if len(parts) > 1 else None,
        "mem_total": safe(parts[2]) if len(parts) > 2 else None,
        "temp":      safe(parts[3]) if len(parts) > 3 else None,
    })


# ── Results ────────────────────────────────────────────────────────────────
@app.route("/api/file/<node_id>/results.tsv")
def api_results(node_id):
    node = get_node(node_id)
    if not node:
        return jsonify({"error": "not found"}), 404
    out, err = ssh_run(node["ip"], f"cat {REPO}/results.tsv 2>/dev/null || echo ''")
    return jsonify({"content": out or "", "error": err})


# ── Start / Stop training loop ────────────────────────────────────────────
@app.route("/api/start/<node_id>", methods=["POST"])
def api_start(node_id):
    node = get_node(node_id)
    if not node:
        return jsonify({"error": "not found"}), 404
    data = request.json or {}
    max_iter = max(1, min(100, int(data.get("max_iter", 10))))
    # Step 1: push correct scripts directly via SFTP (bypasses git-remote issues —
    # the remote may point at karpathy/autoresearch, not our fork)
    sync_results = sync_scripts_to_remote(node["ip"])
    app.logger.info(f"SFTP sync to {node['ip']}: {sync_results}")

    # Step 2: kill old session, clear stale files, launch fresh loop
    cmd = (
        f"cd {REPO} && "
        f"chmod +x {REPO}/loop.sh; "
        f"tmux kill-session -t autoresearch 2>/dev/null || true; "
        # Clear stale logs, inference files, and PID; recreate results.tsv with
        # header-only so OpenCode can read it without a "file not found" error
        f"rm -f {REPO}/loop_node*.log {REPO}/loop_run.log "
        f"    {REPO}/test_inference_results.json {REPO}/test_inference_history.json "
        f"    {REPO}/loop.pid; "
        f"printf 'commit\\tval_auc\\tmemory_gb\\tstatus\\tdescription\\n' > {REPO}/results.tsv; "
        # Launch loop
        f"tmux new-session -d -s autoresearch "
        f"'bash {REPO}/loop.sh {max_iter} 2>&1 | tee {REPO}/loop_node0.log' && "
        # Write the tmux pane's shell PID — completely isolated from this SSH session,
        # so it won't die when the SSH connection closes (unlike pgrep-based approach)
        f"sleep 1 && tmux list-panes -t autoresearch -F '#{{pane_pid}}' 2>/dev/null | head -1 > {REPO}/loop.pid && "
        f"echo started"
    )
    out, err = ssh_run(node["ip"], cmd, timeout=25)
    success = bool(out and "started" in out)
    return jsonify({"success": success, "max_iter": max_iter,
                    "synced": sync_results, "error": err})

@app.route("/api/stop/<node_id>", methods=["POST"])
def api_stop(node_id):
    node = get_node(node_id)
    if not node:
        return jsonify({"error": "not found"}), 404
    out, err = ssh_run(node["ip"],
        f"tmux kill-session -t autoresearch 2>/dev/null; "
        f"rm -f {REPO}/loop.pid; "
        f"echo stopped",
        timeout=10)
    return jsonify({"success": bool(out and "stopped" in out), "error": err})


# ── Select new X-ray test sample ─────────────────────────────────────────
@app.route("/api/select-xray/<node_id>", methods=["POST"])
def api_select_xray(node_id):
    node = get_node(node_id)
    if not node:
        return jsonify({"error": "not found"}), 404

    # Find a random image that has 2–8 positive labels so there's a mix of YES/NO
    # Must cd to REPO first so uv run finds the project's pyproject.toml/uv.lock
    find_cmd = (
        f"cd {REPO} && "
        f"{_UV_BIN} run python3 -c \""
        "from medmnist import ChestMNIST; import random; "
        "ds = ChestMNIST(split='test', size=28, download=True); "
        "good = [i for i,(img,lbl) in enumerate(ds) if 2 <= int(lbl.sum()) <= 8]; "
        "print(random.choice(good) if good else random.randint(0, len(ds)-1))"
        "\""
    )
    out, err = _ssh_exec(node["ip"], find_cmd, timeout=60)
    # out may contain uv progress lines before the actual number — grab the last digit line
    digit_line = next((l.strip() for l in reversed((out or "").splitlines()) if l.strip().isdigit()), None)
    if not digit_line:
        return jsonify({"error": f"Could not pick sample: {err or out}"})
    idx = int(digit_line)

    # Save index, clear old history/results, generate the PNG for display
    setup_cmd = (
        f"cd {REPO} && "
        f"echo {idx} > {REPO}/test_xray_idx.txt && "
        f"rm -f {REPO}/test_inference_history.json {REPO}/test_inference_results.json && "
        f"{_UV_BIN} run python3 -c \""
        f"from medmnist import ChestMNIST; "
        f"ds=ChestMNIST(split='test',size=224,download=True); "
        f"img,lbl=ds[{idx}]; img.convert('L').save('{REPO}/test_xray.png'); "
        f"import json; "
        f"print(json.dumps({{\\\"positives\\\": int(lbl.sum()), \\\"total\\\": len(lbl)}}))"
        f"\" && echo ok"
    )
    out2, err2 = _ssh_exec(node["ip"], setup_cmd, timeout=60)
    success = bool(out2 and "ok" in out2)

    # Extract label info from python output (line before "ok")
    label_info = {}
    for line in (out2 or "").splitlines():
        if line.startswith("{"):
            try:
                label_info = json.loads(line)
            except Exception:
                pass

    return jsonify({
        "success": success,
        "idx":      idx,
        "positives": label_info.get("positives", "?"),
        "total":     label_info.get("total", 14),
        "error":     err2 if not success else None,
    })


# ── Reset train.py to original baseline ───────────────────────────────────
@app.route("/api/reset/<node_id>", methods=["POST"])
def api_reset(node_id):
    node = get_node(node_id)
    if not node:
        return jsonify({"error": "not found"}), 404
    # Find the very first commit that introduced train.py and restore it
    out, err = ssh_run(node["ip"],
        f"cd {REPO} && "
        f"ORIG=$(git log --oneline -- train.py | tail -1 | awk '{{print $1}}') && "
        f"git show $ORIG:train.py > train.py && "
        f"echo \"reset_ok:$ORIG\" || echo 'reset_failed'",
        timeout=15)
    success = out and "reset_ok" in out
    commit = out.split("reset_ok:")[-1].strip() if success else ""
    return jsonify({"success": success, "commit": commit, "error": err})


# ── Frontend ───────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AutoXray Research Monitor</title>
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#e6edf3;
  height:100vh;display:flex;flex-direction:column;overflow:hidden}

header{display:flex;align-items:center;padding:0 20px;height:48px;
  background:#161b22;border-bottom:1px solid #30363d;flex-shrink:0;gap:12px}
header h1{font-size:15px;font-weight:600;color:#f0f6fc}
.badge{font-size:11px;padding:2px 8px;border-radius:20px;background:#21262d;color:#8b949e}

.layout{display:flex;flex:1;overflow:hidden}

aside{width:220px;flex-shrink:0;background:#161b22;border-right:1px solid #30363d;
  overflow-y:auto;padding:12px 0}
.sidebar-title{font-size:11px;font-weight:600;color:#8b949e;text-transform:uppercase;
  letter-spacing:.06em;padding:0 16px 8px}
.node-btn{display:flex;align-items:center;width:100%;padding:8px 16px;
  background:transparent;border:none;color:#c9d1d9;cursor:pointer;font-size:13px;
  text-align:left;gap:8px;transition:background .15s}
.node-btn:hover{background:#21262d}
.node-btn.active{background:#1f6feb22;color:#58a6ff}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;background:#484f58}
.dot.online{background:#3fb950}
.dot.loading{background:#d29922;animation:pulse 1.4s infinite}
.dot.offline{background:#f85149}
.node-info{display:flex;flex-direction:column}
.node-name{font-size:12px;font-weight:500}
.node-meta{font-size:10px;color:#8b949e;margin-top:1px}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

main{flex:1;display:flex;flex-direction:column;overflow:hidden}

.tabs{display:flex;border-bottom:1px solid #30363d;background:#161b22;
  padding:0 16px;flex-shrink:0;gap:4px;align-items:center}
.tab{padding:10px 16px;background:transparent;border:none;
  border-bottom:2px solid transparent;color:#8b949e;cursor:pointer;
  font-size:13px;font-weight:500;transition:color .15s}
.tab:hover{color:#c9d1d9}
.tab.active{color:#f0f6fc;border-bottom-color:#1f6feb}
.tabs-spacer{flex:1}
.reset-btn{padding:5px 12px;background:#21262d;border:1px solid #30363d;
  border-radius:6px;color:#8b949e;font-size:12px;cursor:pointer;transition:all .15s}
.reset-btn:hover{background:#30363d}
.start-btn{padding:5px 14px;background:#1a3a1a;border:1px solid #3fb95066;
  border-radius:6px;color:#3fb950;font-size:12px;font-weight:600;cursor:pointer;transition:all .15s}
.start-btn:hover:not(:disabled){background:#3fb95022}
.start-btn:disabled{opacity:.4;cursor:not-allowed}
.stop-btn{padding:5px 12px;background:#3a1a1a;border:1px solid #f8514966;
  border-radius:6px;color:#f85149;font-size:12px;font-weight:600;cursor:pointer;transition:all .15s}
.stop-btn:hover{background:#f8514922}
.iter-input{width:52px;padding:3px 6px;background:#21262d;border:1px solid #30363d;
  border-radius:6px;color:#c9d1d9;font-size:12px;text-align:center}
.iter-input:focus{outline:none;border-color:#58a6ff}

.content{flex:1;overflow:hidden;position:relative}
.panel{display:none;height:100%;overflow:hidden}
.panel.active{display:flex;flex-direction:column}

/* ── Code live view ── */
.code-header{padding:8px 16px;background:#161b22;border-bottom:1px solid #30363d;
  flex-shrink:0;display:flex;align-items:center;gap:10px}
.code-header strong{font-size:12px;color:#c9d1d9}
.code-header span{font-size:11px;color:#8b949e}
.code-scroll{flex:1;overflow-y:auto;background:#0d1117;
  font-family:'Cascadia Code','Fira Code',monospace;font-size:12.5px;line-height:1.7}
.cl{display:flex;padding:0 8px;min-height:22px}
.cl:hover{background:#ffffff06}
.cl-n{width:42px;flex-shrink:0;color:#484f58;user-select:none;
  font-size:11px;text-align:right;padding-right:14px;padding-top:2px}
.cl-c{flex:1;white-space:pre;color:#c9d1d9;overflow:hidden;text-overflow:ellipsis}
@keyframes slide-in{
  from{opacity:0;transform:translateX(-10px);background:#1e4a20}
  to  {opacity:1;transform:translateX(0);     background:#0c2410}
}
@keyframes fade-green{
  0%  {background:#0c2410;border-left-color:#3fb950}
  100%{background:transparent;border-left-color:transparent}
}
.cl.is-new{border-left:3px solid #3fb950;animation:slide-in .2s ease-out,fade-green 5s .2s ease forwards}
.cl.is-new .cl-c{color:#b5f5b8}

/* ── Phase status bar ── */
.phase-bar{display:flex;align-items:center;gap:10px;padding:9px 16px;
  background:#161b22;border-bottom:1px solid #30363d;flex-shrink:0;
  border-left:3px solid #484f58;transition:border-color .4s}
.phase-bar.ph-llm   {border-left-color:#d29922}
.phase-bar.ph-train {border-left-color:#3fb950}
.phase-bar.ph-done  {border-left-color:#58a6ff}
.phase-bar.ph-warn  {border-left-color:#f85149}
.ph-icon{font-size:15px;flex-shrink:0}
.ph-desc{flex:1;font-size:12px;color:#c9d1d9}
.ph-desc strong{font-weight:600}
.ph-gpu{font-size:11px;font-family:'Cascadia Code',monospace;color:#58a6ff;
  background:#1c2433;padding:2px 8px;border-radius:4px;white-space:nowrap}

/* ── Log view ── */
.terminal{flex:1;overflow-y:auto;padding:12px 16px;
  font-family:'Cascadia Code','Fira Code',monospace;font-size:12.5px;
  line-height:1.6;background:#0d1117;white-space:pre-wrap;word-break:break-all}
.log-line{display:block}
.log-line.iter{color:#79c0ff;font-weight:600}
.log-line.best{color:#3fb950;font-weight:600}
.log-line.warn{color:#d29922}
.log-line.err{color:#f85149}
.log-line.muted{color:#484f58}
.log-line.normal{color:#8b949e}

.toolbar{display:flex;align-items:center;gap:8px;padding:6px 16px;
  background:#161b22;border-bottom:1px solid #30363d;flex-shrink:0}
.toolbar span{font-size:12px;color:#8b949e}
.ind{width:8px;height:8px;border-radius:50%;background:#484f58;flex-shrink:0}
.ind.on{background:#3fb950;animation:pulse 2s infinite}

/* ── Results ── */
.results-wrap{flex:1;overflow:auto;padding:16px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;padding:8px 12px;background:#161b22;color:#8b949e;
  font-weight:600;border-bottom:1px solid #30363d;font-size:11px;
  text-transform:uppercase;letter-spacing:.04em}
td{padding:8px 12px;border-bottom:1px solid #21262d;color:#c9d1d9;font-family:monospace}
tr.keep td{background:#1a2b1a}
tr.discard td{color:#484f58}
.auc-best{color:#3fb950;font-weight:600}

/* ── Iteration context ── */
.iter-ctx{background:#161b22;border-top:1px solid #30363d;padding:8px 16px;
  font-family:'Cascadia Code',monospace;font-size:11px;color:#8b949e;
  flex-shrink:0;white-space:pre-wrap;max-height:80px;overflow-y:auto}

.empty{display:flex;align-items:center;justify-content:center;
  height:100%;color:#484f58;font-size:14px}

/* ── Toast ── */
#toast{position:fixed;bottom:20px;right:20px;padding:10px 18px;
  border-radius:8px;font-size:13px;z-index:999;opacity:0;
  transition:opacity .3s;pointer-events:none}
#toast.show{opacity:1}
#toast.ok{background:#1a3a1a;border:1px solid #3fb950;color:#3fb950}
#toast.fail{background:#3a1a1a;border:1px solid #f85149;color:#f85149}
</style>
</head>
<body>

<header>
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
       stroke="#58a6ff" stroke-width="2">
    <path d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0
             0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18"/>
  </svg>
  <h1>AutoXray Research Monitor</h1>
  <span class="badge" id="global-status">loading...</span>
</header>

<div class="layout">
  <aside>
    <div class="sidebar-title">Nodes</div>
    <div id="node-list"></div>
  </aside>

  <main>
    <div class="tabs">
      <button class="tab active" onclick="switchTab('code')">Code Changes</button>
      <button class="tab" onclick="switchTab('log')">Loop Log</button>
      <button class="tab" onclick="switchTab('results')">Results</button>
      <button class="tab" onclick="switchTab('xray')">Test X-Ray</button>
      <div class="tabs-spacer"></div>
      <input class="iter-input" type="number" id="iter-input" value="10" min="1" max="100"
             title="Number of training iterations">
      <button class="start-btn" id="start-btn" onclick="startTraining()">▶ Start Training</button>
      <button class="stop-btn"  id="stop-btn"  onclick="stopTraining()" style="display:none">■ Stop</button>
      <button class="reset-btn" onclick="resetTrainPy()"
              title="Restore train.py to original baseline commit">⟳ Reset</button>
    </div>

    <div class="content">

      <!-- Code Changes -->
      <div class="panel active" id="panel-code">
        <div class="code-header">
          <span class="ind" id="code-ind"></span>
          <strong id="code-label">train.py</strong>
          <span id="code-hash"></span>
        </div>
        <div class="code-scroll" id="code-view">
          <div class="empty">Select a node</div>
        </div>
        <div id="iter-ctx" style="display:none"></div>
      </div>

      <!-- Loop Log -->
      <div class="panel" id="panel-log">
        <div class="toolbar">
          <span class="ind" id="log-ind"></span>
          <span id="log-label">Select a node</span>
        </div>
        <div class="phase-bar" id="phase-bar">
          <span class="ph-icon" id="ph-icon">💤</span>
          <span class="ph-desc" id="ph-desc">Waiting — press <strong>▶ Start Training</strong> to begin</span>
          <span class="ph-gpu" id="ph-gpu" style="display:none"></span>
        </div>
        <div class="terminal" id="terminal"></div>
      </div>

      <!-- Test X-Ray -->
      <div class="panel" id="panel-xray">
        <div class="toolbar">
          <span id="xray-label">Test X-Ray — sample #42</span>
          <div style="margin-left:auto;display:flex;gap:6px">
            <button id="new-sample-btn" onclick="selectNewXraySample()"
              style="padding:3px 12px;background:#1a2b1a;border:1px solid #3fb95066;
                border-radius:6px;color:#3fb950;font-size:12px;cursor:pointer;
                font-weight:600;transition:all .15s"
              title="Pick a new random image that has a mix of YES/NO disease labels">
              ⚄ New Sample
            </button>
            <button onclick="loadXray()"
              style="padding:3px 10px;background:#21262d;border:1px solid #30363d;
                border-radius:6px;color:#c9d1d9;font-size:12px;cursor:pointer">
              ↻ Reload
            </button>
          </div>
        </div>
        <div id="xray-content" style="flex:1;overflow:auto;padding:20px">
          <div class="empty">Select a node and wait for a training run to complete</div>
        </div>
      </div>

      <!-- Results -->
      <div class="panel" id="panel-results">
        <div class="toolbar">
          <span id="results-label">results.tsv</span>
          <button onclick="loadResults()" style="margin-left:auto;padding:3px 10px;
            background:#21262d;border:1px solid #30363d;border-radius:6px;
            color:#c9d1d9;font-size:12px;cursor:pointer">↻ Refresh</button>
        </div>
        <div class="results-wrap" id="results-wrap">
          <div class="empty">Select a node</div>
        </div>
      </div>

    </div>
  </main>
</div>

<div id="toast"></div>

<script>
// ── State ─────────────────────────────────────────────────────────────────
let nodes = [], activeNodeId = null, activeTab = 'code';
let codeEvt = null, logEvt = null;
let changeCount = 0;
let gpuPollTimer = null;

// ── Phase detection ───────────────────────────────────────────────────────
const PHASE_RULES = [
  { re: /Running OpenCode agent/i,
    cls:'ph-llm',   icon:'🤖',
    html:'<strong>LLM Inference</strong> — Qwen3.5-122B is reading train.py &amp; results, planning the next improvement' },
  { re: /Training started|fp16|OneCycleLR/i,
    cls:'ph-train', icon:'🧠',
    html:'<strong>DenseNet-121 Training</strong> — fitting chest X-ray classifier on GPU (10 min budget)' },
  { re: /val_auc:\s*[\d.]+/i,
    cls:'ph-train', icon:'📊',
    html:'<strong>Evaluating</strong> — computing validation AUC on held-out set' },
  { re: /Running test inference/i,
    cls:'ph-llm',   icon:'🔬',
    html:'<strong>Test Inference</strong> — running sample X-ray #42 through trained DenseNet' },
  { re: /NEW BEST/i,
    cls:'ph-done',  icon:'⭐',
    html:'<strong>New best model found!</strong> — keeping this improvement' },
  { re: /No improvement|discard/i,
    cls:'ph-warn',  icon:'↩️',
    html:'<strong>No improvement</strong> — reverting train.py to baseline' },
  { re: /syntax error|SyntaxError/i,
    cls:'ph-warn',  icon:'⚠️',
    html:'<strong>Syntax error</strong> in generated code — skipping this iteration' },
  { re: /══+\s*Iteration\s*(\d+)/i,
    cls:'ph-llm',   icon:'🔄',
    html: (m) => `<strong>Iteration ${m[1]}</strong> — resetting train.py to Karpathy baseline, launching OpenCode` },
  { re: /Loop complete/i,
    cls:'ph-done',  icon:'✅',
    html:'<strong>All iterations complete!</strong> — research loop finished' },
  { re: /Waiting for vLLM/i,
    cls:'',         icon:'⏳',
    html:'<strong>Waiting for vLLM</strong> — checking Qwen3.5-122B server is ready' },
];

function updatePhaseFromLine(line) {
  for (const rule of PHASE_RULES) {
    const m = line.match(rule.re);
    if (m) {
      const html = typeof rule.html === 'function' ? rule.html(m) : rule.html;
      setPhase(rule.cls, rule.icon, html);
      return;
    }
  }
}

function setPhase(cls, icon, html) {
  const bar  = document.getElementById('phase-bar');
  const icEl = document.getElementById('ph-icon');
  const txEl = document.getElementById('ph-desc');
  if (!bar) return;
  bar.className = 'phase-bar ' + cls;
  icEl.textContent = icon;
  txEl.innerHTML   = html;
}

// ── GPU polling ───────────────────────────────────────────────────────────
function startGpuPoll(nodeId) {
  if (gpuPollTimer) clearInterval(gpuPollTimer);
  const el = document.getElementById('ph-gpu');
  const poll = async () => {
    try {
      const d = await (await fetch(`/api/gpu/${nodeId}`)).json();
      if (d.util == null) { el.style.display='none'; return; }
      el.style.display = '';
      const mem = (d.mem_used != null && d.mem_total != null && d.mem_total > 0)
        ? `  VRAM ${(d.mem_used/1024).toFixed(1)}/${(d.mem_total/1024).toFixed(1)} GB`
        : '';
      const tmp = d.temp != null ? `  ${d.temp}°C` : '';
      el.textContent = `GPU ${d.util}%${mem}${tmp}`;
    } catch(e) { el.style.display='none'; }
  };
  poll();
  gpuPollTimer = setInterval(poll, 8000);
}

function stopGpuPoll() {
  if (gpuPollTimer) { clearInterval(gpuPollTimer); gpuPollTimer = null; }
  const el = document.getElementById('ph-gpu');
  if (el) el.style.display = 'none';
}

// ── Init ──────────────────────────────────────────────────────────────────
async function init() {
  const r = await fetch('/api/nodes');
  nodes = await r.json();
  renderSidebar();
  pollStatuses();
  setInterval(pollStatuses, 8000);
  if (nodes.length) selectNode(nodes[0].id);
}

// ── Sidebar ───────────────────────────────────────────────────────────────
function renderSidebar() {
  document.getElementById('node-list').innerHTML = nodes.map(n => `
    <button class="node-btn" id="nb-${n.id}" onclick="selectNode('${n.id}')">
      <span class="dot" id="dot-${n.id}"></span>
      <span class="node-info">
        <span class="node-name">${n.name}</span>
        <span class="node-meta" id="meta-${n.id}">connecting…</span>
      </span>
    </button>`).join('');
}

async function pollStatuses() {
  let online = 0;
  await Promise.all(nodes.map(async n => {
    try {
      const s = await (await fetch(`/api/status/${n.id}`)).json();
      const dot  = document.getElementById(`dot-${n.id}`);
      const meta = document.getElementById(`meta-${n.id}`);
      dot.className = 'dot ' + (s.online
        ? (s.vllm==='ready' ? 'online' : 'loading') : 'offline');
      const loopRunning = s.loop === 'running';
      meta.textContent = s.online
        ? `vLLM:${s.vllm} · ${loopRunning ? '⚡running' : s.kept+' kept'}`
        : 'offline';
      if (s.online) online++;
      // Only update button state when node is reachable — preserve state on SSH timeout
      if (n.id === activeNodeId && s.online) setLoopRunning(loopRunning);
    } catch {}
  }));
  document.getElementById('global-status').textContent =
    `${online}/${nodes.length} nodes online`;
}

function setLoopRunning(running) {
  const startBtn = document.getElementById('start-btn');
  const stopBtn  = document.getElementById('stop-btn');
  startBtn.style.display = running ? 'none' : '';
  startBtn.disabled = false;
  startBtn.textContent = '▶ Start Training';
  stopBtn.style.display  = running ? '' : 'none';
  if (!running) {
    setPhase('', '💤', 'Not running — press <strong>▶ Start Training</strong> to begin');
    stopGpuPoll();
  }
}

// ── Node select ───────────────────────────────────────────────────────────
function selectNode(id) {
  document.querySelectorAll('.node-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`nb-${id}`)?.classList.add('active');
  activeNodeId = id;
  const node = nodes.find(n => n.id === id);
  // Clear stale UI
  document.getElementById('terminal').innerHTML = '';
  if (codeEvt) { codeEvt.close(); codeEvt = null; }
  document.getElementById('code-ind').className = 'ind';
  document.getElementById('code-hash').textContent = '';
  changeCount = 0;

  loadInitialCode(id);
  startLogStream(id);
  startGpuPoll(id);
  setPhase('', '⟳', 'Checking status…');
  if (activeTab === 'results') loadResults();
  if (activeTab === 'xray')    loadXray();
  document.getElementById('code-label').textContent =
    `Watching ${node?.name ?? id} — train.py`;
  document.getElementById('log-label').textContent =
    `Loop log — ${node?.name ?? id}`;
  // Show indeterminate state — pollStatuses() will resolve the real status
  const sb = document.getElementById('start-btn');
  const xb = document.getElementById('stop-btn');
  sb.style.display = ''; sb.disabled = true; sb.textContent = '⟳ checking…';
  xb.style.display = 'none';
  pollStatuses();
}

// ── Tab switch ────────────────────────────────────────────────────────────
const TAB_IDS = ['code','log','results','xray'];
function switchTab(name) {
  activeTab = name;
  document.querySelectorAll('.tab').forEach((t,i) =>
    t.classList.toggle('active', TAB_IDS[i] === name));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById(`panel-${name}`).classList.add('active');
  if (name === 'results') loadResults();
  if (name === 'xray')    loadXray();
}

// ── Code live view helpers ────────────────────────────────────────────────
function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Parse diff → set of NEW line numbers in the resulting file
function getAddedLines(diffText) {
  const added = new Set();
  let lineNo = 0;
  for (const raw of (diffText || '').split('\n')) {
    if (raw.startsWith('@@')) {
      const m = raw.match(/@@ -\d+(?:,\d+)? \+(\d+)/);
      if (m) lineNo = parseInt(m[1]) - 1;
    } else if (raw.startsWith('+') && !raw.startsWith('+++')) {
      lineNo++;
      added.add(lineNo);
    } else if (!raw.startsWith('-')) {
      lineNo++;
    }
  }
  return added;
}

function renderCodeView(content, addedSet, scrollToFirst) {
  const view = document.getElementById('code-view');
  const lines = content.split('\n');
  // Remove trailing empty line artifact
  if (lines.length && lines[lines.length-1] === '') lines.pop();

  let html = '';
  lines.forEach((line, i) => {
    const n = i + 1;
    const cls = addedSet.has(n) ? 'cl is-new' : 'cl';
    html += `<div class="${cls}"><span class="cl-n">${n}</span><span class="cl-c">${esc(line)}</span></div>`;
  });
  view.innerHTML = html;

  if (scrollToFirst && addedSet.size > 0) {
    const firstNew = Math.min(...addedSet);
    const el = view.children[firstNew - 1];
    if (el) el.scrollIntoView({block:'center', behavior:'smooth'});
  }
}

async function loadInitialCode(nodeId) {
  const view = document.getElementById('code-view');
  view.innerHTML = '<div class="empty">Loading train.py…</div>';
  try {
    const d = await (await fetch(`/api/trainpy/${nodeId}`)).json();
    if (d.content) {
      renderCodeView(d.content, new Set(), false);
      const node = nodes.find(n => n.id === nodeId);
      document.getElementById('code-label').textContent =
        `${node?.name ?? nodeId} — train.py (baseline)`;
      document.getElementById('code-hash').textContent = '';
    }
  } catch(e) {
    view.innerHTML = `<div class="empty">Could not load train.py</div>`;
  }
}

// ── Code live stream ──────────────────────────────────────────────────────
function startCodeStream(nodeId) {
  if (codeEvt) { codeEvt.close(); codeEvt = null; }
  changeCount = 0;
  document.getElementById('code-ind').className = 'ind on';

  codeEvt = new EventSource(`/api/code-stream/${nodeId}`);
  codeEvt.onmessage = e => {
    if (!e.data) return;
    const d = JSON.parse(e.data);
    if (d.error) return;

    changeCount++;
    const added = getAddedLines(d.diff);
    renderCodeView(d.full, added, true);

    const node = nodes.find(n => n.id === nodeId);
    const lbl = document.getElementById('code-label');
    if (added.size === 0) {
      lbl.textContent = `${node?.name ?? nodeId} — train.py (baseline · waiting for OpenCode to write…)`;
    } else {
      lbl.textContent = `${node?.name ?? nodeId} — train.py  ·  change #${changeCount}  ·  ${added.size} lines added`;
    }
    document.getElementById('code-hash').textContent = d.hash.slice(0,8);
  };
  codeEvt.onerror = () => {
    document.getElementById('code-ind').className = 'ind';
  };
}

// ── Loop log stream ───────────────────────────────────────────────────────
function startLogStream(nodeId) {
  if (logEvt) { logEvt.close(); logEvt = null; }
  const ind = document.getElementById('log-ind');
  ind.className = 'ind on';

  logEvt = new EventSource(`/api/log-stream/${nodeId}`);
  logEvt.onmessage = e => {
    if (!e.data) return;
    const text = JSON.parse(e.data);
    if (!text) return;
    updatePhaseFromLine(text);
    const term = document.getElementById('terminal');
    const el = document.createElement('span');
    el.className = 'log-line ' + classifyLine(text);
    el.textContent = text + '\n';
    term.appendChild(el);
    while (term.children.length > 3000) term.removeChild(term.firstChild);
    term.scrollTop = term.scrollHeight;
  };
  logEvt.onerror = () => { ind.className = 'ind'; };
}

function classifyLine(l) {
  if (l.includes('══')) return 'iter';
  if (/NEW BEST|keep/i.test(l)) return 'best';
  if (/WARNING|warn/i.test(l)) return 'warn';
  if (/ERROR|error/i.test(l)) return 'err';
  if (l.startsWith('[') && /\d{2}:\d{2}:\d{2}/.test(l)) return 'normal';
  return 'muted';
}

// ── Results ───────────────────────────────────────────────────────────────
async function loadResults() {
  if (!activeNodeId) return;
  const wrap = document.getElementById('results-wrap');
  try {
    const d = await (await fetch(`/api/file/${activeNodeId}/results.tsv`)).json();
    const tsv = (d.content || '').trim();
    if (!tsv) { wrap.innerHTML = '<div class="empty">No results yet</div>'; return; }
    const rows = tsv.split('\n').map(l => l.split('\t'));
    const hdr = rows[0], data = rows.slice(1);
    const maxAuc = Math.max(...data.map(r => parseFloat(r[1])||0));
    wrap.innerHTML = `<table><thead><tr>${hdr.map(h=>`<th>${h}</th>`).join('')}</tr></thead>
      <tbody>${data.map(row => {
        const keep = row[3]==='keep';
        const best = keep && Math.abs(parseFloat(row[1])-maxAuc)<1e-6;
        return `<tr class="${keep?'keep':'discard'}">${
          row.map((c,i)=>`<td class="${i===1&&best?'auc-best':''}">${c}</td>`).join('')}</tr>`;
      }).join('')}</tbody></table>`;
  } catch(e) {
    wrap.innerHTML = `<div class="empty">Error: ${e}</div>`;
  }
}

// ── Reset train.py ────────────────────────────────────────────────────────
async function resetTrainPy() {
  if (!activeNodeId) { toast('Select a node first', false); return; }
  const node = nodes.find(n => n.id === activeNodeId);
  if (!confirm(`Reset train.py on ${node?.name} to original baseline commit?\nThis overwrites the current file.`)) return;
  try {
    const d = await (await fetch(`/api/reset/${activeNodeId}`, {method:'POST'})).json();
    if (d.success) {
      toast(`✓ Restored to commit ${d.commit}`, true);
    } else {
      toast(`✗ Reset failed: ${d.error || 'unknown'}`, false);
    }
  } catch(e) {
    toast(`✗ ${e}`, false);
  }
}

// ── New X-Ray sample selection ────────────────────────────────────────────
async function selectNewXraySample() {
  if (!activeNodeId) { toast('Select a node first', false); return; }
  const btn  = document.getElementById('new-sample-btn');
  const wrap = document.getElementById('xray-content');
  btn.disabled = true;
  btn.textContent = '⟳ picking…';
  wrap.innerHTML = '<div class="empty">Picking a new sample with mixed disease labels…</div>';
  try {
    const d = await (await fetch(`/api/select-xray/${activeNodeId}`, {method:'POST'})).json();
    if (d.success) {
      toast(`✓ Sample #${d.idx} — ${d.positives}/${d.total} positive labels · history cleared`, true);
      document.getElementById('xray-label').textContent =
        `Test X-Ray — sample #${d.idx}  (${d.positives} of ${d.total} diseases present)`;
      await loadXray();
    } else {
      toast(`✗ ${d.error || 'Failed'}`, false);
      wrap.innerHTML = `<div class="empty">Error: ${d.error}</div>`;
    }
  } catch(e) {
    toast(`✗ ${e}`, false);
    wrap.innerHTML = `<div class="empty">Error: ${e}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '⚄ New Sample';
  }
}

// ── Test X-Ray ────────────────────────────────────────────────────────────
async function loadXray() {
  if (!activeNodeId) return;
  const wrap = document.getElementById('xray-content');
  wrap.innerHTML = '<div class="empty">Loading…</div>';
  try {
    const [imgResp, inferResp, histResp] = await Promise.all([
      fetch(`/api/xray-image/${activeNodeId}`),
      fetch(`/api/inference/${activeNodeId}`),
      fetch(`/api/inference-history/${activeNodeId}`)
    ]);
    const imgData    = await imgResp.json();
    const inferData  = await inferResp.json();
    const histData   = await histResp.json();

    if (imgData.error && inferData.error) {
      wrap.innerHTML = `<div class="empty">No inference results yet — run a training iteration first</div>`;
      return;
    }

    // Keep the label in sync with whatever sample is currently active
    if (inferData.test_idx != null) {
      document.getElementById('xray-label').textContent =
        `Test X-Ray — sample #${inferData.test_idx}`;
    }

    const imgSrc = imgData.data
      ? `<img src="data:image/png;base64,${imgData.data}"
              style="width:320px;height:320px;image-rendering:auto;
                     border:2px solid #30363d;border-radius:8px;background:#000">`
      : `<div style="width:320px;height:320px;background:#161b22;border:2px solid #30363d;
                     border-radius:8px;display:flex;align-items:center;justify-content:center;
                     color:#484f58;font-size:12px">No image</div>`;

    let tableRows = '';
    let correctCount = 0, wrongCount = 0;
    if (inferData.predictions) {
      for (const p of inferData.predictions) {
        const isPositive = p.actual === 1;
        const correct    = p.correct;
        if (correct) correctCount++; else wrongCount++;
        const confPct = (p.confidence * 100).toFixed(1);
        const barColor = correct ? '#3fb950' : '#f85149';
        const rowBg    = !correct ? 'background:#2b1a1a' : (isPositive ? 'background:#1a2b1a' : '');
        tableRows += `
          <tr style="${rowBg}">
            <td style="color:#c9d1d9;text-transform:capitalize">${p.disease}</td>
            <td>
              <div style="display:flex;align-items:center;gap:6px">
                <div style="width:80px;height:8px;background:#21262d;border-radius:4px;overflow:hidden">
                  <div style="width:${confPct}%;height:100%;background:${barColor};border-radius:4px"></div>
                </div>
                <span style="color:#8b949e;font-size:11px">${confPct}%</span>
              </div>
            </td>
            <td style="color:${p.predicted?'#79c0ff':'#484f58'}">${p.predicted ? 'YES' : 'no'}</td>
            <td style="color:${isPositive?'#3fb950':'#484f58'}">${isPositive ? 'YES' : 'no'}</td>
            <td style="font-size:16px;text-align:center">${correct ? '✓' : '✗'}</td>
          </tr>`;
      }
    }

    const ts = inferData.timestamp
      ? new Date(inferData.timestamp).toLocaleString() : 'unknown';

    // Build history timeline
    let historySection = '';
    const history = Array.isArray(histData) ? histData : [];
    if (history.length > 0) {
      const maxAcc = Math.max(...history.map(h => h.test_accuracy_pct || 0));
      const histRows = history.map((h, idx) => {
        const isBest = Math.abs((h.test_accuracy_pct || 0) - maxAcc) < 0.05;
        const valAuc = h.val_auc_from_training != null
          ? parseFloat(h.val_auc_from_training).toFixed(4) : '—';
        const accPct = h.test_accuracy_pct != null
          ? h.test_accuracy_pct.toFixed(1) + '%' : '—';
        const correct = h.test_correct != null ? h.test_correct : '—';
        const total   = h.summary?.total ?? 14;
        const ts2     = h.timestamp
          ? new Date(h.timestamp).toLocaleTimeString() : '';
        const rowStyle = isBest ? 'background:#1a2b1a' : (idx % 2 === 1 ? 'background:#161b22' : '');
        const accColor = isBest ? '#3fb950' : '#c9d1d9';
        const barW     = h.test_accuracy_pct != null
          ? Math.round(h.test_accuracy_pct) : 0;
        return `<tr style="${rowStyle}">
          <td style="color:#8b949e;padding:5px 10px;font-size:12px">${h.iteration ?? idx+1}</td>
          <td style="color:#79c0ff;padding:5px 10px;font-family:monospace;font-size:12px">${valAuc}</td>
          <td style="padding:5px 10px">
            <div style="display:flex;align-items:center;gap:6px">
              <div style="width:90px;height:7px;background:#21262d;border-radius:4px;overflow:hidden">
                <div style="width:${barW}%;height:100%;background:${accColor};border-radius:4px;transition:width .4s"></div>
              </div>
              <span style="color:${accColor};font-size:12px;font-weight:${isBest?'700':'400'}">${accPct}</span>
              ${isBest ? '<span style="color:#3fb950;font-size:10px">★ best</span>' : ''}
            </div>
          </td>
          <td style="color:#8b949e;padding:5px 10px;font-size:12px">${correct}/${total}</td>
          <td style="color:#484f58;padding:5px 10px;font-size:11px">${ts2}</td>
        </tr>`;
      }).join('');
      historySection = `
        <div style="margin-top:24px;border-top:1px solid #30363d;padding-top:16px">
          <div style="font-size:12px;font-weight:600;color:#8b949e;text-transform:uppercase;
                      letter-spacing:.06em;margin-bottom:10px">Test Accuracy History</div>
          <table style="width:100%;border-collapse:collapse">
            <thead>
              <tr>
                <th style="text-align:left;padding:5px 10px;color:#8b949e;font-size:11px;
                           text-transform:uppercase;border-bottom:1px solid #30363d">Iter</th>
                <th style="text-align:left;padding:5px 10px;color:#8b949e;font-size:11px;
                           text-transform:uppercase;border-bottom:1px solid #30363d">Val AUC</th>
                <th style="text-align:left;padding:5px 10px;color:#8b949e;font-size:11px;
                           text-transform:uppercase;border-bottom:1px solid #30363d">Test Accuracy</th>
                <th style="text-align:left;padding:5px 10px;color:#8b949e;font-size:11px;
                           text-transform:uppercase;border-bottom:1px solid #30363d">Score</th>
                <th style="text-align:left;padding:5px 10px;color:#8b949e;font-size:11px;
                           text-transform:uppercase;border-bottom:1px solid #30363d">Time</th>
              </tr>
            </thead>
            <tbody>${histRows}</tbody>
          </table>
        </div>`;
    }

    wrap.innerHTML = `
      <div style="display:flex;flex-direction:column;width:100%">
        <div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap">
          <div style="flex-shrink:0">
            ${imgSrc}
            <div style="margin-top:10px;font-size:11px;color:#8b949e;text-align:center">
              ChestMNIST sample #${inferData.test_idx ?? 42}<br>
              ${ts}
            </div>
            <div style="margin-top:12px;display:flex;gap:12px;justify-content:center">
              <div style="text-align:center;padding:8px 16px;background:#1a2b1a;border-radius:8px;border:1px solid #3fb95044">
                <div style="font-size:22px;font-weight:700;color:#3fb950">${correctCount}</div>
                <div style="font-size:11px;color:#8b949e">correct</div>
              </div>
              <div style="text-align:center;padding:8px 16px;background:#2b1a1a;border-radius:8px;border:1px solid #f8514944">
                <div style="font-size:22px;font-weight:700;color:#f85149">${wrongCount}</div>
                <div style="font-size:11px;color:#8b949e">wrong</div>
              </div>
            </div>
          </div>
          <div style="flex:1;overflow:auto;min-width:300px">
            <table style="width:100%;border-collapse:collapse;font-size:12.5px">
              <thead>
                <tr>
                  <th style="text-align:left;padding:6px 10px;color:#8b949e;font-size:11px;
                             text-transform:uppercase;border-bottom:1px solid #30363d">Disease</th>
                  <th style="text-align:left;padding:6px 10px;color:#8b949e;font-size:11px;
                             text-transform:uppercase;border-bottom:1px solid #30363d">Confidence</th>
                  <th style="padding:6px 10px;color:#8b949e;font-size:11px;
                             text-transform:uppercase;border-bottom:1px solid #30363d">Predicted</th>
                  <th style="padding:6px 10px;color:#8b949e;font-size:11px;
                             text-transform:uppercase;border-bottom:1px solid #30363d">Actual</th>
                  <th style="padding:6px 10px;color:#8b949e;font-size:11px;
                             text-transform:uppercase;border-bottom:1px solid #30363d">✓/✗</th>
                </tr>
              </thead>
              <tbody>${tableRows}</tbody>
            </table>
          </div>
        </div>
        ${historySection}
      </div>`;
  } catch(e) {
    wrap.innerHTML = `<div class="empty">Error: ${e}</div>`;
  }
}

// ── Start / Stop training ─────────────────────────────────────────────────
async function startTraining() {
  if (!activeNodeId) { toast('Select a node first', false); return; }
  const maxIter = parseInt(document.getElementById('iter-input').value) || 10;
  const node = nodes.find(n => n.id === activeNodeId);
  if (!confirm(
    `Start ${maxIter}-iteration research loop on ${node?.name}?\n\n` +
    `train.py will be reset to the original Karpathy baseline and\n` +
    `previous inference results will be cleared for a fresh demo.`
  )) return;

  const btn = document.getElementById('start-btn');
  btn.disabled = true;
  btn.textContent = '…starting';

  try {
    const d = await (await fetch(`/api/start/${activeNodeId}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({max_iter: maxIter})
    })).json();
    if (d.success) {
      toast(`✓ Training started — ${maxIter} iterations`, true);
      setLoopRunning(true);
      switchTab('log');
      // Reconnect streams after log file is created by loop.sh
      // Capture nodeId in closure so a node switch can't race this timer
      const nid = activeNodeId;
      setTimeout(() => {
        if (activeNodeId === nid) {
          startLogStream(nid);
          startCodeStream(nid);
        }
      }, 2500);
    } else {
      toast(`✗ Failed to start: ${d.error || 'unknown'}`, false);
      btn.disabled = false;
      btn.textContent = '▶ Start Training';
    }
  } catch(e) {
    toast(`✗ ${e}`, false);
    btn.disabled = false;
    btn.textContent = '▶ Start Training';
  }
}

async function stopTraining() {
  if (!activeNodeId) { toast('Select a node first', false); return; }
  const node = nodes.find(n => n.id === activeNodeId);
  if (!confirm(`Stop training on ${node?.name}?`)) return;
  try {
    const d = await (await fetch(`/api/stop/${activeNodeId}`, {method:'POST'})).json();
    if (d.success) {
      toast('■ Training stopped', true);
      setLoopRunning(false);
    } else {
      toast(`✗ ${d.error || 'not running'}`, false);
    }
  } catch(e) {
    toast(`✗ ${e}`, false);
  }
}

function toast(msg, ok) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'show ' + (ok ? 'ok' : 'fail');
  setTimeout(() => { el.className = ''; }, 3500);
}

init();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

if __name__ == "__main__":
    print("AutoXray Research Monitor -> http://localhost:7860")
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
