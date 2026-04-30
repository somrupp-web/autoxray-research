#!/usr/bin/env python3
"""
setup_cluster_nodes.py — Bootstrap all 4 cluster nodes for autoresearch.

Run from the autoresearch repo root on Windows:
    python setup_cluster_nodes.py

What it does on each node (in parallel):
  1. Detects CUDA version
  2. Installs uv
  3. Creates /home/nvidia/autoresearch/, uploads all project files
  4. Initialises a git repo (required by loop.sh)
  5. Runs `uv sync` to install PyTorch + all deps
  6. Verifies torch, NCCL, and GPU are accessible
"""

import concurrent.futures
import os
import paramiko
import sys

CLUSTER_NODES = [
    ("node-0", "10.137.203.228"),
    ("node-1", "10.137.203.184"),
    ("node-2", "10.137.203.174"),
    ("node-3", "10.137.203.177"),
]

SSH_USER    = "nvidia"
SSH_PASS    = "nvidia"
REPO_REMOTE = "/home/nvidia/autoresearch"
UV          = "/home/nvidia/.local/bin/uv"
LOCAL_REPO  = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FILES = [
    ("pyproject.toml",   "pyproject.toml"),
    ("prepare.py",       "prepare.py"),
    ("train.py",         "train.py"),
    ("train.py",         "train.py.baseline"),
    ("loop.sh",          "loop.sh"),
    ("test_inference.py","test_inference.py"),
    ("program.md",       "program.md"),
    ("plot_results.py",  "plot_results.py"),
]


def _ssh(ip, timeout=15):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(ip, username=SSH_USER, password=SSH_PASS,
              timeout=timeout, banner_timeout=timeout)
    return c


def _run(c, cmd, timeout=300, label=""):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    stdout   = out.read().decode(errors="replace").strip()
    stderr   = err.read().decode(errors="replace").strip()
    exitcode = out.channel.recv_exit_status()
    return stdout, stderr, exitcode


def setup_node(name, ip):
    steps = []

    def log(msg):
        line = f"[{name}] {msg}"
        steps.append(line)
        print(line, flush=True)

    try:
        c = _ssh(ip)
        log("Connected")

        # ── 1. Detect CUDA ────────────────────────────────────────────────
        out, _, _ = _run(c, "nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null | head -1", 15)
        log(f"GPU: {out or 'unknown'}")

        cuda_out, _, _ = _run(c, "nvcc --version 2>/dev/null | grep 'release' | grep -oP 'V[0-9]+' | head -1", 15)
        log(f"CUDA: {cuda_out or 'check manually'}")

        # ── 2. Install uv ─────────────────────────────────────────────────
        out, _, code = _run(c, f"test -f {UV} && echo exists || (curl -LsSf https://astral.sh/uv/install.sh | sh)", 120)
        log(f"uv: {'already installed' if 'exists' in out else ('installed' if code == 0 else f'FAILED: {out[-100:]}')}")

        # ── 3. Create repo dir ────────────────────────────────────────────
        _run(c, f"mkdir -p {REPO_REMOTE}", 10)
        log(f"Directory {REPO_REMOTE} ready")

        # ── 4. Upload project files ───────────────────────────────────────
        sftp = c.open_sftp()
        for local_name, remote_name in UPLOAD_FILES:
            local_path = os.path.join(LOCAL_REPO, local_name)
            if not os.path.exists(local_path):
                log(f"  SKIP {local_name} (not found locally)")
                continue
            sftp.put(local_path, f"{REPO_REMOTE}/{remote_name}")
        sftp.close()
        log(f"Uploaded {len(UPLOAD_FILES)} files")

        # ── 5. Permissions + git init ─────────────────────────────────────
        out, err, code = _run(c, f"""
chmod +x {REPO_REMOTE}/loop.sh
cd {REPO_REMOTE}
git init -q 2>/dev/null || true
git config user.email 'autoresearch@nvidia.com'
git config user.name  'AutoResearch'
git add -A
git commit -q -m 'initial setup' 2>/dev/null || true
echo git_ok
""", 30)
        log(f"Git init: {'ok' if 'git_ok' in out else err[:80]}")

        # ── 6. Create results.tsv header ──────────────────────────────────
        _run(c, f"printf 'commit\\tval_auc\\tmemory_gb\\tstatus\\tdescription\\n' > {REPO_REMOTE}/results.tsv", 10)
        log("results.tsv initialised")

        # ── 7. Install dependencies ───────────────────────────────────────
        log("Running uv sync (installs PyTorch + deps, may take 5-10 min)...")
        out, err, code = _run(c, f"cd {REPO_REMOTE} && {UV} sync 2>&1", 900)
        last = "\n".join((out + err).splitlines()[-5:])
        log(f"uv sync exit={code}:\n  {last.replace(chr(10), chr(10)+'  ')}")
        if code != 0:
            log("WARNING: uv sync failed — check pyproject.toml CUDA index vs installed CUDA")

        # ── 8. Verify torch + NCCL ────────────────────────────────────────
        log("Verifying PyTorch + NCCL...")
        out, err, code = _run(c, f"""cd {REPO_REMOTE} && {UV} run python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('NCCL:', torch.cuda.nccl.version())
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('NCCL available:', torch.distributed.is_nccl_available())
" 2>&1""", 60)
        log(f"Verify:\n  {(out or err).replace(chr(10), chr(10)+'  ')}")

        c.close()
        success = code == 0
        log("Setup COMPLETE" if success else "Setup DONE WITH WARNINGS")
        return name, success

    except Exception as e:
        steps.append(f"[{name}] FATAL: {e}")
        print(f"[{name}] FATAL: {e}", flush=True)
        return name, False


def main():
    print("=" * 60)
    print("AutoResearch Cluster Node Setup")
    print(f"Nodes: {[ip for _, ip in CLUSTER_NODES]}")
    print("=" * 60)
    print()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(setup_node, n, ip): n for n, ip in CLUSTER_NODES}
        results = []
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, ok in sorted(results):
        status = "OK" if ok else "FAILED"
        print(f"  {name:10s}  {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print()
        print("All nodes ready. You can now start training on each node")
        print("via the WebUI, or run loop.sh directly on each node.")
        print()
        print("For DDP training, set on each node:")
        print("  export NCCL_SOCKET_IFNAME=enp1s0f0np0")
    else:
        print()
        print("Some nodes failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
