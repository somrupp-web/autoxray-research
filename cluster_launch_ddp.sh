#!/bin/bash
# ============================================================
# cluster_launch_ddp.sh — Start the DDP research loop on node-0
#
# Run from any machine (Windows/Mac/Linux) with sshpass.
# node-0 orchestrates all 4 nodes: OpenCode rewrites
# train_ddp.py each iteration, then 4-node DDP training runs.
#
# Usage:
#   ./cluster_launch_ddp.sh [max_iterations] [epochs_per_iter]
#
# Examples:
#   ./cluster_launch_ddp.sh          # 20 iterations, 3 epochs each
#   ./cluster_launch_ddp.sh 30 5     # 30 iterations, 5 epochs each
# ============================================================

NODE0="nvidia@10.137.203.228"
SSH_PASS="nvidia"
REPO_DIR="/home/nvidia/autoresearch"
SESSION="ddp-research"
MAX_ITER="${1:-20}"
EPOCHS="${2:-3}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

remote() {
    sshpass -p "$SSH_PASS" ssh \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        "$NODE0" "$@"
}

# ── check sshpass ─────────────────────────────────────────────
if ! command -v sshpass &>/dev/null; then
    log "ERROR: sshpass not found."
    log "  Ubuntu/Debian : sudo apt install sshpass"
    log "  macOS         : brew install hudochenkov/sshpass/sshpass"
    exit 1
fi

# ── connectivity check ────────────────────────────────────────
log "Connecting to node-0 ($NODE0)..."
remote "echo ok" > /dev/null 2>&1 || { log "ERROR: cannot reach node-0"; exit 1; }
log "node-0 reachable."

# ── upload new files ──────────────────────────────────────────
log "Uploading loop_ddp.sh and program_ddp.md to node-0..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
sshpass -p "$SSH_PASS" scp -o StrictHostKeyChecking=no \
    "$SCRIPT_DIR/loop_ddp.sh" \
    "$SCRIPT_DIR/program_ddp.md" \
    "$SCRIPT_DIR/train_ddp.py" \
    "$SCRIPT_DIR/prepare.py" \
    "nvidia@10.137.203.228:$REPO_DIR/"
remote "chmod +x $REPO_DIR/loop_ddp.sh"
log "Files uploaded."

# ── kill any existing session ─────────────────────────────────
remote "tmux kill-session -t '$SESSION' 2>/dev/null || true"

# ── launch loop_ddp.sh in tmux on node-0 ─────────────────────
log "Launching DDP research loop (tmux session: $SESSION)..."
remote "tmux new-session -d -s '$SESSION' -x 220 -y 50 && \
        tmux send-keys -t '$SESSION' \
            'cd $REPO_DIR && ./loop_ddp.sh $MAX_ITER $EPOCHS 2>&1 | tee loop_ddp.log' Enter"

log ""
log "═══════════════════════════════════════════════════════"
log "DDP research loop started on node-0."
log ""
log "Monitor live:"
log "  sshpass -p nvidia ssh nvidia@10.137.203.228 -t 'tmux attach -t $SESSION'"
log ""
log "Check logs:"
log "  sshpass -p nvidia ssh nvidia@10.137.203.228 'tail -f $REPO_DIR/loop_ddp.log'"
log ""
log "Stop:"
log "  sshpass -p nvidia ssh nvidia@10.137.203.228 'tmux kill-session -t $SESSION'"
log "═══════════════════════════════════════════════════════"
