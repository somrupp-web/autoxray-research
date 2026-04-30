#!/usr/bin/env python3
"""
launch_ddp.py — Launch 4-node DDP training with BiomedCLIP

Usage:  python launch_ddp.py [--nih /path/to/NIH_ChestXray14]

Steps:
  1. Installs open_clip on all nodes if missing
  2. Uploads train_ddp.py and prepare.py to all nodes
  3. Launches torchrun simultaneously on all 4 nodes
  4. Streams live output (rank-0 first, others buffered)
  5. Prints final val_auc summary
"""

import argparse
import concurrent.futures
import os
import paramiko
import sys
import threading
import time

# ── Cluster config ────────────────────────────────────────────────────────────
NODES = [
    # (label,    management IP,    interconnect IP,      rank)
    ("node-0", "10.137.203.228", "100.100.100.10",   0),
    ("node-1", "10.137.203.184", "100.100.100.12",   1),
    ("node-2", "10.137.203.174", "100.100.100.14",   2),
    ("node-3", "10.137.203.177", "100.100.100.16",   3),
]

MASTER_ADDR = "10.137.203.228"   # node-0 management IP (rendezvous/TCPStore)
MASTER_PORT = 29500
# NCCL uses the 200 Gbps interconnect for gradient sync (set via NCCL_SOCKET_IFNAME)
REPO        = "/home/nvidia/autoresearch"
VENV        = f"{REPO}/.venv"
UV_BIN      = "/home/nvidia/.local/bin/uv"
PIP         = f"VIRTUAL_ENV={VENV} {UV_BIN} pip"
TORCHRUN    = f"{VENV}/bin/torchrun"
PYTHON      = f"{VENV}/bin/python3"

LOCAL_REPO  = os.path.dirname(os.path.abspath(__file__))

UPLOAD = [
    "train_ddp.py",
    "prepare.py",
    "program.md",
    "program_ddp.md",
    "loop_ddp.sh",
]


def _ssh(ip, timeout=15):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(ip, username="nvidia", password="nvidia",
              timeout=timeout, banner_timeout=timeout)
    return c


def _run(c, cmd, timeout=120):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    stdout   = out.read().decode(errors="replace").strip()
    stderr   = err.read().decode(errors="replace").strip()
    exitcode = out.channel.recv_exit_status()
    return stdout, stderr, exitcode


def prepare_node(name, mgmt_ip, use_hf=False):
    """Install open_clip + transformers, upload files, pre-download BiomedCLIP."""
    print(f"[{name}] Preparing...", flush=True)
    c = _ssh(mgmt_ip)

    # Check / install open_clip
    out, _, _ = _run(c, f"{PYTHON} -c 'import open_clip; print(open_clip.__version__)' 2>&1", 15)
    if "Error" in out or not out:
        print(f"[{name}] Installing open_clip...", flush=True)
        out, err, code = _run(c, f"{PIP} install open-clip-torch -q 2>&1", 300)
        print(f"[{name}] open_clip install: {'ok' if code == 0 else err[-100:]}", flush=True)
    else:
        print(f"[{name}] open_clip {out} already installed", flush=True)

    # Check / install transformers (needed by BiomedCLIP text encoder)
    out, _, _ = _run(c, f"{PYTHON} -c 'import transformers; print(transformers.__version__)' 2>&1", 15)
    if "Error" in out or not out:
        print(f"[{name}] Installing transformers...", flush=True)
        _, _, code = _run(c, f"{PIP} install transformers -q 2>&1", 300)
        print(f"[{name}] transformers install: {'ok' if code == 0 else 'FAILED'}", flush=True)
    else:
        print(f"[{name}] transformers {out} already installed", flush=True)

    # Upload files
    sftp = c.open_sftp()
    for fname in UPLOAD:
        local = os.path.join(LOCAL_REPO, fname)
        if os.path.exists(local):
            sftp.put(local, f"{REPO}/{fname}")
    sftp.close()
    print(f"[{name}] Files uploaded", flush=True)

    # Pre-download BiomedCLIP weights so training doesn't stall on first run
    print(f"[{name}] Pre-downloading BiomedCLIP (cached after first run)...", flush=True)
    dl_cmd = (
        f"{PYTHON} -c \""
        f"import open_clip; "
        f"open_clip.create_model_and_transforms("
        f"'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'"
        f"); print('BiomedCLIP ready')"
        f"\" 2>&1 | tail -3"
    )
    out, _, code = _run(c, dl_cmd, 900)
    print(f"[{name}] BiomedCLIP: {'ready' if code == 0 else 'download may have failed — ' + out[-80:]}", flush=True)

    if use_hf:
        print(f"[{name}] Pre-downloading NIH ChestX-ray14 from HuggingFace...", flush=True)
        hf_cmd = (
            f"{PYTHON} -c \""
            f"from datasets import load_dataset; "
            f"ds = load_dataset('BahaaEldin0/NIH-Chest-Xray-14'); "
            f"print('HF NIH ready train=' + str(len(ds[\\\"train\\\"])) + ' val=' + str(len(ds[\\\"valid\\\"]))); "
            f"\" 2>&1 | tail -5"
        )
        out, _, code = _run(c, hf_cmd, 3600)
        print(f"[{name}] NIH HF: {'ready — ' + out.strip()[-120:] if code == 0 else 'FAILED — ' + out[-120:]}", flush=True)

    c.close()


def launch_node(name, mgmt_ip, rank, nih_dir="", epochs=0, use_hf=False):
    """Launch torchrun on this node and stream output."""
    c = _ssh(mgmt_ip, timeout=20)

    nih_env    = f"NIH_CHEST_DIR={nih_dir} " if nih_dir else ""
    hf_env     = "USE_HF_NIH=1 " if use_hf else ""
    epochs_env = f"MAX_EPOCHS={epochs} " if epochs > 0 else ""

    # env:// init: set RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT as env vars,
    # run python3 directly — no torchrun rendezvous server needed.
    # NCCL_SOCKET_IFNAME routes gradient sync over the 200 Gbps interconnect.
    cmd = (
        f"cd {REPO} && "
        f"RANK={rank} "
        f"LOCAL_RANK=0 "
        f"WORLD_SIZE={len(NODES)} "
        f"MASTER_ADDR={MASTER_ADDR} "
        f"MASTER_PORT={MASTER_PORT} "
        f"NCCL_SOCKET_IFNAME=enp1s0f0np0 "
        f"NCCL_DEBUG=WARN "
        f"{nih_env}"
        f"{hf_env}"
        f"{epochs_env}"
        f"{PYTHON} train_ddp.py 2>&1"
    )

    # timeout=1800: model download (~5min) + training (10min) + buffer
    # Non-master nodes produce no output; per-read timeout must exceed total runtime.
    _, out, _ = c.exec_command(cmd, timeout=1800)

    raw = out.read()
    text = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
    lines = text.splitlines()

    if rank == 0:
        for line in lines:
            # Strip non-ASCII (tqdm progress bars) so Windows console doesn't crash
            safe = line.encode("ascii", "replace").decode("ascii")
            print(f"[{name}] {safe}", flush=True)

    c.close()
    return rank, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nih", default="",
                        help="Path to NIH ChestX-ray14 on the remote nodes "
                             "(e.g. /home/nvidia/data/NIH_ChestXray14). "
                             "Leave empty to use ChestMNIST.")
    parser.add_argument("--epochs", type=int, default=0,
                        help="Train for this many epochs (0 = use TIME_BUDGET seconds).")
    parser.add_argument("--hf-nih", action="store_true",
                        help="Use HuggingFace NIH ChestX-ray14 (BahaaEldin0/NIH-Chest-Xray-14).")
    args = parser.parse_args()

    nih_dir = args.nih
    if args.hf_nih:
        print("Dataset: NIH ChestX-ray14 via HuggingFace (BahaaEldin0/NIH-Chest-Xray-14)")
    elif nih_dir:
        print(f"Dataset: NIH ChestX-ray14 at {nih_dir}")
    else:
        print("Dataset: ChestMNIST (use --hf-nih for NIH via HuggingFace)")
    if args.epochs:
        print(f"Budget : {args.epochs} epochs")

    # ── Step 1: prepare all nodes in parallel ─────────────────────────────────
    print("\n=== Step 1: Preparing nodes ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(prepare_node, name, mgmt_ip, args.hf_nih)
                for name, mgmt_ip, _, _ in NODES]
        concurrent.futures.wait(futs)
    print("All nodes prepared.\n")

    # ── Step 2: launch DDP on all nodes simultaneously ────────────────────────
    print("=== Step 2: Launching DDP training ===")
    print(f"MASTER: {MASTER_ADDR}:{MASTER_PORT}")
    print(f"Nodes : {len(NODES)} x 1 GPU | effective batch = {32 * len(NODES)}")
    print(f"Budget: 600s\n")

    t_launch = time.time()
    outputs  = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {
            ex.submit(launch_node, name, mgmt_ip, rank, nih_dir, args.epochs, args.hf_nih): (name, rank)
            for name, mgmt_ip, _, rank in NODES
        }
        for f in concurrent.futures.as_completed(futs):
            name_r, rank_r = futs[f]
            try:
                r, text = f.result()
                outputs[r] = text
                if r != 0:
                    # Print non-rank-0 output after completion
                    print(f"\n--- {name_r} output (rank {r}) ---")
                    for line in text.splitlines()[-20:]:   # last 20 lines
                        print(f"  {line}")
            except Exception as e:
                print(f"[{name_r}] LAUNCH ERROR: {e}", flush=True)

    elapsed = time.time() - t_launch

    # ── Step 3: summary ───────────────────────────────────────────────────────
    print("\n=== Results (from rank-0) ===")
    rank0_output = outputs.get(0, "")
    val_auc = None
    for line in rank0_output.splitlines():
        if "val_auc" in line or "training_seconds" in line or \
           "world_size" in line or "global_batch" in line or "peak_vram" in line:
            print(f"  {line.strip()}")
        if line.strip().startswith("val_auc:"):
            try:
                val_auc = float(line.split()[-1])
            except Exception:
                pass

    print(f"\nTotal wall time: {elapsed:.0f}s")
    if val_auc:
        print(f"val_auc: {val_auc:.6f}  (CheXNet target: 0.841)")
    else:
        print("val_auc not found in rank-0 output — check above for errors")


if __name__ == "__main__":
    main()
