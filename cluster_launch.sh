#!/bin/bash
# ============================================================
# cluster_launch.sh — start the autoresearch loop on all 4
# DGX Spark nodes from a single command.
#
# Prerequisites on each node:
#   - vLLM installed and model at /home/nvidia/models/Qwen3.5-122B-A10B-AWQ/
#   - OpenCode installed (fnm + node)
#   - uv installed
#   - SSH key auth from this machine to all nodes
#   - tmux installed  (sudo apt install tmux)
#   - git remote 'origin' pointing to GitHub repo
#
# Usage:
#   ./cluster_launch.sh [max_iterations]
# ============================================================

# ── Configure: fill in your node IPs ─────────────────────────
NODES=(
    "nvidia@10.137.203.188"   # node-0
    "nvidia@10.137.203.189"   # node-1  ← update with real IPs
    "nvidia@10.137.203.190"   # node-2
    "nvidia@10.137.203.191"   # node-3
)

REPO_DIR="/home/nvidia/autoresearch"
GITHUB_REPO="https://github.com/somrupp-web/autoxray-research.git"
MAX_ITER="${1:-20}"
VLLM_MODEL="/home/nvidia/models/Qwen3.5-122B-A10B-AWQ"
SESSION="autoresearch"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── helper: run a command on a node ──────────────────────────
remote() {
    local node=$1; shift
    ssh -o ConnectTimeout=10 -o BatchMode=yes "$node" "$@"
}

# ── 1. Verify connectivity ────────────────────────────────────
log "Checking connectivity to all nodes..."
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    if remote "$node" "echo ok" > /dev/null 2>&1; then
        log "  node-$i ($node) ✓"
    else
        log "  node-$i ($node) ✗  — cannot connect. Check SSH keys."
        exit 1
    fi
done

# ── 2. Clone / update repo on each node ──────────────────────
log "Syncing repo on all nodes..."
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    remote "$node" "
        if [ -d '$REPO_DIR/.git' ]; then
            git -C '$REPO_DIR' fetch origin
            git -C '$REPO_DIR' checkout main 2>/dev/null || true
            git -C '$REPO_DIR' pull origin main 2>/dev/null || true
        else
            git clone '$GITHUB_REPO' '$REPO_DIR'
        fi
        # Install Python dependencies
        cd '$REPO_DIR' && /home/nvidia/.local/bin/uv sync
    " && log "  node-$i: repo ready." &
done
wait
log "All repos synced."

# ── 3. Start vLLM on each node (if not already running) ──────
log "Starting vLLM on each node..."
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    remote "$node" "
        if curl -sf http://127.0.0.1:8080/v1/models > /dev/null 2>&1; then
            echo 'vLLM already running on node-$i'
        else
            tmux new-session -d -s vllm-server -x 220 -y 50 2>/dev/null || true
            tmux send-keys -t vllm-server \
                'vllm serve $VLLM_MODEL \
                    --port 8080 \
                    --enable-auto-tool-choice \
                    --tool-call-parser qwen3_xml \
                    --max-model-len 65536 \
                    --gpu-memory-utilization 0.90 \
                    > /tmp/vllm.log 2>&1' Enter
            echo 'vLLM started on node-$i'
        fi
    " &
done
wait

# ── 4. Wait for vLLM to be ready on all nodes ────────────────
log "Waiting for vLLM to be ready on all nodes (up to 5 min)..."
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    log "  Waiting for node-$i..."
    for attempt in $(seq 1 60); do
        if remote "$node" "curl -sf http://127.0.0.1:8080/v1/models" > /dev/null 2>&1; then
            log "  node-$i: vLLM ready ✓"
            break
        fi
        sleep 5
        if [ $attempt -eq 60 ]; then
            log "  node-$i: vLLM not ready after 5 min — check /tmp/vllm.log on that node"
            exit 1
        fi
    done
done

# ── 5. Start loop.sh on each node in a tmux session ──────────
log "Launching research loops..."
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    remote "$node" "
        tmux kill-session -t '$SESSION' 2>/dev/null || true
        tmux new-session -d -s '$SESSION' -x 220 -y 50
        tmux send-keys -t '$SESSION' \
            'cd $REPO_DIR && NODE_ID=$i ./loop.sh $MAX_ITER 2>&1 | tee loop_node${i}.log' Enter
    " && log "  node-$i: loop started (tmux session: $SESSION) ✓"
done

log ""
log "═══════════════════════════════════════════════════════"
log "All 4 research loops are running."
log ""
log "Monitor with:"
log "  ./cluster_status.sh"
log ""
log "Attach to a node's tmux session:"
for i in "${!NODES[@]}"; do
    log "  ssh ${NODES[$i]} -t 'tmux attach -t $SESSION'"
done
log ""
log "Stop all loops:"
log "  ./cluster_stop.sh"
log "═══════════════════════════════════════════════════════"
