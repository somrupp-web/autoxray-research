"""
AutoXray Research Monitor
Run:  pip install flask paramiko
Then: python app.py
Open: http://localhost:7860
"""

import json
import time
from flask import Flask, Response, jsonify, render_template_string, request
import paramiko

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

def get_node(node_id):
    return next((n for n in NODES if n["id"] == node_id), None)


# ── Status ─────────────────────────────────────────────────────────────────
@app.route("/api/nodes")
def api_nodes():
    return jsonify(NODES)

@app.route("/api/status/<node_id>")
def api_status(node_id):
    node = get_node(node_id)
    if not node:
        return jsonify({"error": "not found"}), 404
    out, err = ssh_run(node["ip"],
        "curl -sf http://127.0.0.1:8080/v1/models >/dev/null 2>&1 && echo vllm_ready || echo vllm_down;"
        f"ls {REPO}/loop_node*.log {REPO}/loop_run.log 2>/dev/null | head -1;"
        f"cd {REPO} && git log --oneline 2>/dev/null | wc -l;"
        f"grep -c 'keep' {REPO}/results.tsv 2>/dev/null || echo 0",
        timeout=6)
    if err:
        return jsonify({"online": False, "vllm": "offline", "loop": "stopped",
                        "commits": 0, "kept": 0})
    lines = (out or "").splitlines()
    return jsonify({
        "online":  True,
        "vllm":    "ready" if any("vllm_ready" in l for l in lines) else "loading",
        "loop":    "running" if any(".log" in l for l in lines) else "stopped",
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
                        # Get the diff vs last commit; fall back to full file on first load
                        _, diff_out, _ = ssh.exec_command(
                            f"cd {REPO} && "
                            f"git diff HEAD -- train.py 2>/dev/null")
                        diff = diff_out.read().decode(errors="replace").strip()

                        # Also get the full file
                        _, full_out, _ = ssh.exec_command(
                            f"cat {REPO}/train.py 2>/dev/null")
                        full = full_out.read().decode(errors="replace")

                        # Get current iteration info from the newest loop log
                        _, log_out, _ = ssh.exec_command(
                            f"ls -t {REPO}/loop_node*.log {REPO}/loop_run.log "
                            f"2>/dev/null | head -1 | xargs tail -5 2>/dev/null")
                        log_tail = log_out.read().decode(errors="replace").strip()

                        payload = {
                            "hash":     current_hash,
                            "diff":     diff,
                            "full":     full,
                            "log_tail": log_tail,
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
                f"LOG=$(ls -t {REPO}/loop_node*.log {REPO}/loop_run.log "
                f"2>/dev/null | head -1);"
                f" if [ -n \"$LOG\" ]; then tail -n 200 -f \"$LOG\";"
                f" else tail -n 100 -f /tmp/vllm.log 2>/dev/null; fi"
            )
            ch.exec_command(cmd)
            buf = b""
            while True:
                if ch.recv_ready():
                    buf += ch.recv(8192)
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        text = line.decode(errors="replace").rstrip()
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


# ── Results ────────────────────────────────────────────────────────────────
@app.route("/api/file/<node_id>/results.tsv")
def api_results(node_id):
    node = get_node(node_id)
    if not node:
        return jsonify({"error": "not found"}), 404
    out, err = ssh_run(node["ip"], f"cat {REPO}/results.tsv 2>/dev/null || echo ''")
    return jsonify({"content": out or "", "error": err})


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
.reset-btn{padding:5px 12px;background:#21262d;border:1px solid #f8514966;
  border-radius:6px;color:#f85149;font-size:12px;cursor:pointer;transition:all .15s}
.reset-btn:hover{background:#f8514922}

.content{flex:1;overflow:hidden;position:relative}
.panel{display:none;height:100%;overflow:hidden}
.panel.active{display:flex;flex-direction:column}

/* ── Diff view ── */
.diff-wrap{flex:1;overflow-y:auto;font-family:'Cascadia Code','Fira Code',monospace;
  font-size:12.5px;line-height:1.6;background:#0d1117}
.diff-header{padding:8px 16px;background:#161b22;border-bottom:1px solid #30363d;
  flex-shrink:0;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.diff-header span{font-size:12px;color:#8b949e}
.diff-header strong{font-size:12px;color:#c9d1d9}
.view-toggle{display:flex;gap:0}
.vt-btn{padding:3px 10px;background:#21262d;border:1px solid #30363d;
  color:#8b949e;font-size:11px;cursor:pointer}
.vt-btn:first-child{border-radius:6px 0 0 6px}
.vt-btn:last-child{border-radius:0 6px 6px 0}
.vt-btn.active{background:#1f6feb33;color:#58a6ff;border-color:#1f6feb}

/* Diff lines */
.diff-block{padding:12px 0}
.diff-line{display:flex;padding:0 16px;min-height:20px}
.diff-line:hover{background:#ffffff08}
.dl-num{width:50px;flex-shrink:0;color:#484f58;user-select:none;font-size:11px;padding-top:1px}
.dl-sign{width:16px;flex-shrink:0;font-weight:600}
.dl-code{flex:1;white-space:pre-wrap;word-break:break-all}
.diff-line.add{background:#1a2b1a}.diff-line.add .dl-sign{color:#3fb950}.diff-line.add .dl-code{color:#aff1b0}
.diff-line.del{background:#2b1a1a}.diff-line.del .dl-sign{color:#f85149}.diff-line.del .dl-code{color:#ffa8a8;text-decoration:line-through}
.diff-line.hunk{background:#1c2433}.diff-line.hunk .dl-code{color:#58a6ff;font-size:11px}
.diff-line.ctx .dl-code{color:#8b949e}

/* Full code view */
.full-wrap{flex:1;overflow:auto}
.full-wrap pre{margin:0;border-radius:0;min-height:100%}
.full-wrap code{font-size:12.5px!important}

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
      <button class="reset-btn" onclick="resetTrainPy()"
              title="Restore train.py to original baseline commit">⟳ Reset train.py</button>
    </div>

    <div class="content">

      <!-- Code Changes -->
      <div class="panel active" id="panel-code">
        <div class="diff-header">
          <span class="ind" id="code-ind"></span>
          <strong id="code-label">Select a node to watch code changes</strong>
          <span id="code-hash" style="margin-left:auto;font-family:monospace"></span>
          <div class="view-toggle">
            <button class="vt-btn active" id="btn-diff" onclick="setView('diff')">Diff</button>
            <button class="vt-btn" id="btn-full" onclick="setView('full')">Full file</button>
          </div>
        </div>
        <div class="diff-wrap" id="diff-view">
          <div class="empty">Waiting for train.py to change…</div>
        </div>
        <div class="full-wrap" id="full-view" style="display:none">
          <pre><code class="language-python" id="full-code"></code></pre>
        </div>
        <div class="iter-ctx" id="iter-ctx" style="display:none"></div>
      </div>

      <!-- Loop Log -->
      <div class="panel" id="panel-log">
        <div class="toolbar">
          <span class="ind" id="log-ind"></span>
          <span id="log-label">Select a node</span>
        </div>
        <div class="terminal" id="terminal"></div>
      </div>

      <!-- Test X-Ray -->
      <div class="panel" id="panel-xray">
        <div class="toolbar">
          <span id="xray-label">Test X-Ray — sample #42</span>
          <button onclick="loadXray()" style="margin-left:auto;padding:3px 10px;
            background:#21262d;border:1px solid #30363d;border-radius:6px;
            color:#c9d1d9;font-size:12px;cursor:pointer">↻ Refresh</button>
        </div>
        <div id="xray-content" style="flex:1;overflow:auto;padding:20px;display:flex;gap:24px;align-items:flex-start">
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
let codeView = 'diff';   // 'diff' | 'full'
let changeCount = 0;

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
      const dot = document.getElementById(`dot-${n.id}`);
      const meta = document.getElementById(`meta-${n.id}`);
      dot.className = 'dot ' + (s.online
        ? (s.vllm==='ready' ? 'online' : 'loading') : 'offline');
      meta.textContent = s.online
        ? `vLLM:${s.vllm} · ${s.kept} kept`
        : 'offline';
      if (s.online) online++;
    } catch {}
  }));
  document.getElementById('global-status').textContent =
    `${online}/${nodes.length} nodes online`;
}

// ── Node select ───────────────────────────────────────────────────────────
function selectNode(id) {
  document.querySelectorAll('.node-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`nb-${id}`)?.classList.add('active');
  activeNodeId = id;
  const node = nodes.find(n => n.id === id);
  startCodeStream(id);
  startLogStream(id);
  if (activeTab === 'results') loadResults();
  document.getElementById('code-label').textContent =
    `Watching ${node?.name ?? id} — train.py`;
  document.getElementById('log-label').textContent =
    `Loop log — ${node?.name ?? id}`;
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

// ── View toggle (diff / full) ─────────────────────────────────────────────
function setView(v) {
  codeView = v;
  document.getElementById('btn-diff').classList.toggle('active', v==='diff');
  document.getElementById('btn-full').classList.toggle('active', v==='full');
  document.getElementById('diff-view').style.display = v==='diff' ? '' : 'none';
  document.getElementById('full-view').style.display = v==='full' ? '' : 'none';
}

// ── Code diff stream ──────────────────────────────────────────────────────
function startCodeStream(nodeId) {
  if (codeEvt) { codeEvt.close(); codeEvt = null; }
  changeCount = 0;

  const ind = document.getElementById('code-ind');
  ind.className = 'ind on';

  codeEvt = new EventSource(`/api/code-stream/${nodeId}`);
  codeEvt.onmessage = e => {
    if (!e.data) return;
    const d = JSON.parse(e.data);
    if (d.error) { console.warn('code-stream error:', d.error); return; }

    changeCount++;
    document.getElementById('code-hash').textContent =
      `#${changeCount}  ${d.hash.slice(0,8)}`;

    // Show iteration context
    if (d.log_tail) {
      const ctx = document.getElementById('iter-ctx');
      ctx.style.display = '';
      ctx.textContent = d.log_tail;
    }

    // Diff view
    renderDiff(d.diff, d.full);

    // Full file view
    const fc = document.getElementById('full-code');
    fc.textContent = d.full;
    hljs.highlightElement(fc);
  };
  codeEvt.onerror = () => { ind.className = 'ind'; };
}

function renderDiff(diffText, fullText) {
  const wrap = document.getElementById('diff-view');
  if (!diffText) {
    // No diff means this is the first load (unmodified from HEAD)
    wrap.innerHTML = `<div style="padding:16px;color:#8b949e;font-family:monospace;font-size:12px">
      No uncommitted changes — showing current train.py (${fullText.split('\n').length} lines)</div>`;
    return;
  }

  const lines = diffText.split('\n');
  let html = '<div class="diff-block">';
  let lineNo = 0;

  for (const raw of lines) {
    if (raw.startsWith('@@')) {
      // Parse hunk header to get line number
      const m = raw.match(/@@ -\d+(?:,\d+)? \+(\d+)/);
      if (m) lineNo = parseInt(m[1]) - 1;
      html += `<div class="diff-line hunk">
        <span class="dl-num"></span><span class="dl-sign"></span>
        <span class="dl-code">${esc(raw)}</span></div>`;
    } else if (raw.startsWith('+') && !raw.startsWith('+++')) {
      lineNo++;
      html += `<div class="diff-line add">
        <span class="dl-num">${lineNo}</span>
        <span class="dl-sign">+</span>
        <span class="dl-code">${esc(raw.slice(1))}</span></div>`;
    } else if (raw.startsWith('-') && !raw.startsWith('---')) {
      html += `<div class="diff-line del">
        <span class="dl-num"></span>
        <span class="dl-sign">−</span>
        <span class="dl-code">${esc(raw.slice(1))}</span></div>`;
    } else if (raw.startsWith(' ')) {
      lineNo++;
      html += `<div class="diff-line ctx">
        <span class="dl-num">${lineNo}</span>
        <span class="dl-sign"></span>
        <span class="dl-code">${esc(raw.slice(1))}</span></div>`;
    }
  }
  html += '</div>';
  wrap.innerHTML = html;
  wrap.scrollTop = 0;   // scroll to top to show first change
}

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
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

// ── Test X-Ray ────────────────────────────────────────────────────────────
async function loadXray() {
  if (!activeNodeId) return;
  const wrap = document.getElementById('xray-content');
  wrap.innerHTML = '<div class="empty">Loading…</div>';
  try {
    const [imgResp, inferResp] = await Promise.all([
      fetch(`/api/xray-image/${activeNodeId}`),
      fetch(`/api/inference/${activeNodeId}`)
    ]);
    const imgData   = await imgResp.json();
    const inferData = await inferResp.json();

    if (imgData.error && inferData.error) {
      wrap.innerHTML = `<div class="empty">No inference results yet — run a training iteration first</div>`;
      return;
    }

    const imgSrc = imgData.data
      ? `<img src="data:image/png;base64,${imgData.data}"
              style="width:280px;height:280px;image-rendering:pixelated;
                     border:2px solid #30363d;border-radius:8px;background:#000">`
      : `<div style="width:280px;height:280px;background:#161b22;border:2px solid #30363d;
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

    wrap.innerHTML = `
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
      <div style="flex:1;overflow:auto">
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
      </div>`;
  } catch(e) {
    wrap.innerHTML = `<div class="empty">Error: ${e}</div>`;
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
    print("AutoXray Research Monitor → http://localhost:7860")
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
