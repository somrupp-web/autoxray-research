#!/bin/bash
# ============================================================
# Autonomous X-ray Research Loop
#
# Single-node usage  (default):
#   ./loop.sh [max_iterations]
#
# Cluster usage  (set by cluster_launch.sh automatically):
#   CLUSTER_SIZE=4 NODE_ID=0 ./loop.sh [max_iterations]
# ============================================================

REPO_DIR="/home/nvidia/autoresearch"
UV="/home/nvidia/.local/bin/uv"
OPENCODE=$(find /home/nvidia/.local/bin /home/nvidia/.local/share/fnm -name opencode -type f 2>/dev/null | head -1)
MODEL="vllm//home/nvidia/models/Qwen3.5-122B-A10B-AWQ"
MAX_ITER="${1:-20}"
NODE_ID="${NODE_ID:-0}"
CLUSTER_SIZE="${CLUSTER_SIZE:-1}"          # 1 = single node, 4 = cluster
BRANCH="node-${NODE_ID}"
NODES=(0 1 2 3)

cd "$REPO_DIR"

log() { echo "[$(date '+%H:%M:%S')] [node-${NODE_ID}] $*"; }

# ── Mode banner ───────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
if [ "$CLUSTER_SIZE" -gt 1 ]; then
    echo "║  MODE : CLUSTER  (node ${NODE_ID} of $((CLUSTER_SIZE-1)), ${CLUSTER_SIZE} nodes total)         ║"
    echo "║  SYNC : git branch ${BRANCH}  →  GitHub             ║"
else
    echo "║  MODE : SINGLE NODE                                  ║"
    echo "║  SYNC : local only  (no git push/fetch)              ║"
fi
echo "║  ITER : ${MAX_ITER} iterations  ·  10-min training budget    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── prerequisites ─────────────────────────────────────────────
[ -z "$OPENCODE" ] && { log "ERROR: opencode binary not found"; exit 1; }
log "OpenCode : $OPENCODE"
log "Model    : $MODEL"
log "Branch   : $BRANCH"
log "Max iter : $MAX_ITER"

# ── checkout node branch (cluster only) ──────────────────────
if [ "$CLUSTER_SIZE" -gt 1 ]; then
    git fetch origin 2>/dev/null || true
    if git show-ref --quiet "refs/remotes/origin/$BRANCH"; then
        git checkout -B "$BRANCH" "origin/$BRANCH" 2>/dev/null || git checkout "$BRANCH"
    else
        git checkout -B "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
    fi
fi

# ── wait for vLLM on this node ────────────────────────────────
log "Waiting for vLLM on :8080..."
for i in $(seq 1 60); do
    curl -sf http://127.0.0.1:8080/v1/models > /dev/null 2>&1 && { log "vLLM ready."; break; }
    sleep 5
done
curl -sf http://127.0.0.1:8080/v1/models > /dev/null 2>&1 || { log "ERROR: vLLM not ready. Abort."; exit 1; }

# ── sync: pull results from all node branches into results.tsv ──
# In cluster mode: merges all node branches from GitHub.
# In single-node mode: no-op (results.tsv is already local).
sync_global_results() {
    if [ "$CLUSTER_SIZE" -le 1 ]; then
        log "Single-node mode — skipping cluster sync."
        return
    fi
    git fetch origin 2>/dev/null || true
    > /tmp/global_results.tsv
    for n in "${NODES[@]}"; do
        git show "origin/node-${n}:results.tsv" 2>/dev/null \
            | grep -v '^commit' \
            >> /tmp/global_results.tsv || true
    done
    grep -v '^commit' results.tsv 2>/dev/null >> /tmp/global_results.tsv || true
    sort -u -k1,1 /tmp/global_results.tsv \
        | { printf 'commit\tval_auc\tmemory_gb\tstatus\tdescription\n'; cat; } \
        > results.tsv
    log "Synced: $(grep -c 'keep\|discard' results.tsv 2>/dev/null || echo 0) total experiments across cluster."
}

# ── push local results to origin (cluster only) ──────────────
push_results() {
    if [ "$CLUSTER_SIZE" -le 1 ]; then
        return   # single-node: nothing to push
    fi
    git add results.tsv results_chart.png 2>/dev/null || true
    git diff --cached --quiet || \
        git commit -m "node-${NODE_ID}: results sync after iter-${iter}"
    git push origin "$BRANCH" 2>/dev/null || {
        git pull --rebase origin "$BRANCH" 2>/dev/null \
            && git push origin "$BRANCH" 2>/dev/null || true
    }
}

# ── original baseline commit for train.py ─────────────────────
ORIG_COMMIT=$(git log --oneline -- train.py | tail -1 | awk '{print $1}')
log "Baseline train.py commit: $ORIG_COMMIT"

# ── clear stale inference results so WebUI starts clean ────────
rm -f "$REPO_DIR/test_inference_results.json" "$REPO_DIR/test_inference_history.json"
log "Cleared previous inference results — fresh session."

# ── main loop ─────────────────────────────────────────────────
for iter in $(seq 1 "$MAX_ITER"); do
    log "══════ Iteration $iter / $MAX_ITER ══════"

    # Always start each iteration from the original unmodified train.py
    git show "$ORIG_COMMIT":train.py > train.py
    log "Reset train.py to original baseline."

    sync_global_results

    BEST_AUC=$(grep "keep" results.tsv 2>/dev/null | awk '{print $2}' | sort -n | tail -1)
    BEST_AUC="${BEST_AUC:-0.7255}"
    TRIED=$(grep -c 'keep\|discard' results.tsv 2>/dev/null || echo 0)
    log "Global best val_auc=$BEST_AUC  |  Total experiments=$TRIED"

    if [ "$CLUSTER_SIZE" -gt 1 ]; then
        CONTEXT="You are AI research agent node-${NODE_ID} running on a ${CLUSTER_SIZE}-node NVIDIA DGX Spark cluster.
NOTE: $((CLUSTER_SIZE-1)) other agents are running simultaneously. results.tsv contains ALL nodes'
experiments — check it carefully and choose a technique not yet tried by any agent."
    else
        CONTEXT="You are an AI research agent running on a single NVIDIA DGX Spark."
    fi

    PROMPT="${CONTEXT}

Your goal: improve chest X-ray disease classification. Current best val_auc=${BEST_AUC}.

Steps you MUST follow:
1. Read /home/nvidia/autoresearch/train.py
2. Read /home/nvidia/autoresearch/results.tsv
3. Read /home/nvidia/autoresearch/program.md
4. Choose ONE specific improvement not yet tried that could beat val_auc=${BEST_AUC}
5. Write the complete new train.py using the bash tool:
   cat > /home/nvidia/autoresearch/train.py << 'PYEOF'
   [complete new python file]
   PYEOF
6. Stop. Do not run training yourself.

IMPORTANT: Use bash to write the file (step 5). Do NOT use the edit tool."

    log "Running OpenCode agent..."
    "$OPENCODE" run --model "$MODEL" "$PROMPT"
    OPENCODE_EXIT=$?

    if [ $OPENCODE_EXIT -ne 0 ]; then
        log "WARNING: OpenCode exited $OPENCODE_EXIT — skipping."
        git checkout train.py 2>/dev/null || true
        continue
    fi

    if ! python3 -m py_compile train.py 2>/tmp/syntax_err.txt; then
        log "WARNING: syntax error in train.py — skipping."
        log "$(cat /tmp/syntax_err.txt)"
        git checkout train.py 2>/dev/null || true
        continue
    fi

    if git diff --quiet train.py; then
        log "WARNING: train.py unchanged — skipping."
        continue
    fi

    HEADLINE=$(head -5 train.py | grep -i '#\|"""\|experiment' | head -1 \
               | sed 's/[#"]*//g' | xargs)
    git add train.py
    git commit -m "node-${NODE_ID} iter-${iter}: ${HEADLINE:-LLM improvement}" \
        || { log "Commit failed — skipping."; continue; }
    COMMIT=$(git rev-parse --short HEAD)
    log "Committed $COMMIT"

    log "Training (10-min budget)..."
    $UV run train.py 2>&1 | tee run.log
    log "Training complete."

    # Flexible extraction: handles any format OpenCode might use for output
    VAL_AUC=$(grep -Eo 'val_auc[=: ]+[0-9]+\.[0-9]+' run.log | grep -Eo '[0-9]+\.[0-9]+' | head -1)
    PEAK_MB=$(grep -Eo 'peak_vram_mb[=: ]+[0-9]+' run.log | grep -Eo '[0-9]+$' | head -1)
    VRAM_GB=$(python3 -c \
        "print(f'{float(\"${PEAK_MB:-0}\")/1024:.1f}')" 2>/dev/null || echo "?")

    log "val_auc=$VAL_AUC  global_best=$BEST_AUC"

    BETTER=$(python3 -c "
try:    print('yes' if float('${VAL_AUC:-0}') > float('$BEST_AUC') else 'no')
except: print('no')
")

    if [ "$BETTER" = "yes" ]; then
        STATUS="keep"
        log "NEW BEST — keeping."
    else
        STATUS="discard"
        log "No improvement — reverting."
        git reset HEAD~1
        git checkout train.py
    fi

    DESC="node-${NODE_ID} iter-${iter}: ${HEADLINE:-improvement}"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$COMMIT" "${VAL_AUC:-0.000000}" "$VRAM_GB" "$STATUS" "$DESC" \
        >> results.tsv

    $UV run "$REPO_DIR/plot_results.py" results.tsv results_chart.png 2>/dev/null \
        && log "Chart updated."

    log "Running test inference on sample X-ray..."
    $UV run "$REPO_DIR/test_inference.py" --val-auc "${VAL_AUC:-0}" --iter "$iter" > /tmp/infer_out.txt 2>&1 \
        && log "Test inference complete." \
        || log "WARNING: test inference failed — check /tmp/infer_out.txt"

    push_results
    log "Pushed to origin/$BRANCH"
    echo ""
done

log "Loop complete — $MAX_ITER iterations done on node-${NODE_ID}."
