#!/bin/bash
# ============================================================
# cluster_launch.sh — start the autoresearch loop on all 4
# DGX Spark nodes from a single command.
#
# Usage:
#   ./cluster_launch.sh [max_iterations]
# ============================================================

# ── Node configuration ────────────────────────────────────────
NODES=(
    "nvidia@10.137.203.228"   # node-0
    "nvidia@10.137.203.184"   # node-1
    "nvidia@10.137.203.174"   # node-2
    "nvidia@10.137.203.177"   # node-3
)
SSH_PASS="nvidia"

REPO_DIR="/home/nvidia/autoresearch"
GITHUB_REPO="https://github.com/somrupp-web/autoxray-research.git"
MAX_ITER="${1:-20}"
VLLM_MODEL="/home/nvidia/models/Qwen3.5-122B-A10B-AWQ"
SESSION="autoresearch"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── helper: run a command on a remote node ────────────────────
remote() {
    local node=$1; shift
    sshpass -p "$SSH_PASS" ssh \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        "$node" "$@"
}

# ── check sshpass is installed ────────────────────────────────
if ! command -v sshpass &>/dev/null; then
    log "ERROR: sshpass not found. Install it first:"
    log "  Ubuntu/Debian : sudo apt install sshpass"
    log "  macOS         : brew install hudochenkov/sshpass/sshpass"
    exit 1
fi

# ── 1. Verify connectivity ────────────────────────────────────
log "Checking connectivity to all nodes..."
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    if remote "$node" "echo ok" > /dev/null 2>&1; then
        log "  node-$i  $node  ✓"
    else
        log "  node-$i  $node  ✗  — cannot connect. Check IP/credentials."
        exit 1
    fi
done
log "All nodes reachable."

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
                '/home/nvidia/.local/bin/vllm serve $VLLM_MODEL \
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
    log "  Waiting for node-$i ($node)..."
    for attempt in $(seq 1 60); do
        if remote "$node" "curl -sf http://127.0.0.1:8080/v1/models > /dev/null 2>&1"; then
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

# ── 5. Launch loop.sh on each node in tmux ───────────────────
log "Launching research loops..."
CLUSTER_SIZE=${#NODES[@]}
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    remote "$node" "
        tmux kill-session -t '$SESSION' 2>/dev/null || true
        tmux new-session -d -s '$SESSION' -x 220 -y 50
        tmux send-keys -t '$SESSION' \
            'cd $REPO_DIR && CLUSTER_SIZE=$CLUSTER_SIZE NODE_ID=$i ./loop.sh $MAX_ITER 2>&1 | tee loop_node${i}.log' Enter
    " && log "  node-$i: loop started (tmux session: $SESSION) ✓"
done

log ""
log "═══════════════════════════════════════════════════════"
log "All 4 research loops are running."
log ""
log "Monitor  :  ./cluster_status.sh"
log "Stop all :  ./cluster_stop.sh"
log ""
log "Attach to a node's tmux session:"
for i in "${!NODES[@]}"; do
    log "  sshpass -p nvidia ssh nvidia@${NODES[$i]#nvidia@} -t 'tmux attach -t $SESSION'"
done
log "═══════════════════════════════════════════════════════"
