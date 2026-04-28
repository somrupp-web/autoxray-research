#!/bin/bash
# cluster_status.sh — show live status of all 4 research loops

NODES=(
    "nvidia@10.137.203.228"   # node-0
    "nvidia@10.137.203.184"   # node-1
    "nvidia@10.137.203.174"   # node-2
    "nvidia@10.137.203.177"   # node-3
)
SSH_PASS="nvidia"
REPO_DIR="/home/nvidia/autoresearch"

remote() {
    sshpass -p "$SSH_PASS" ssh \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=5 \
        "$1" "${@:2}" 2>/dev/null
}

echo "═══════════════════════════════════════════════════════════════"
echo "  Autoresearch Cluster Status  —  $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════"

for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    echo ""
    echo "── node-$i  ($node) ────────────────────────────────────────"

    LASTLOG=$(remote "$node" "tail -5 $REPO_DIR/loop_node${i}.log 2>/dev/null")
    if [ -n "$LASTLOG" ]; then
        echo "$LASTLOG"
    else
        echo "  (no log yet)"
    fi

    BEST=$(remote "$node" "grep 'keep' $REPO_DIR/results.tsv 2>/dev/null \
           | awk '{print \$2, \$5}' | sort -n | tail -1")
    [ -n "$BEST" ] && echo "  Local best: $BEST"

    VLLM=$(remote "$node" "curl -sf http://127.0.0.1:8080/v1/models > /dev/null 2>&1 \
           && echo 'running' || echo 'DOWN'")
    echo "  vLLM: ${VLLM:-unknown}"
done

echo ""
echo "── Global best (all branches) ─────────────────────────────────"
git fetch origin 2>/dev/null || true
GLOBAL_BEST=$(
    for n in 0 1 2 3; do
        git show "origin/node-${n}:results.tsv" 2>/dev/null | grep 'keep' | awk '{print $2}'
    done | sort -n | tail -1
)
echo "  Best val_auc across cluster: ${GLOBAL_BEST:-(not yet available)}"
echo "  CheXNet target: 0.841"
echo "═══════════════════════════════════════════════════════════════"
