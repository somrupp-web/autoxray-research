#!/bin/bash
# ============================================================
# Autonomous X-ray Research Loop
# Agent: OpenCode + Qwen3.5-122B-AWQ (vLLM on DGX Spark)
# OpenCode reads files and writes train.py — not the wrapper
# ============================================================

REPO_DIR="/home/nvidia/autoresearch"
UV="/home/nvidia/.local/bin/uv"
OPENCODE=$(find /home/nvidia/.local/share/fnm -name opencode -type f 2>/dev/null | head -1)
MODEL="vllm//home/nvidia/models/Qwen3.5-122B-A10B-AWQ"
MAX_ITER="${1:-20}"

cd "$REPO_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── verify prerequisites ──────────────────────────────────────
[ -z "$OPENCODE" ] && { log "ERROR: opencode binary not found"; exit 1; }
log "OpenCode: $OPENCODE"
log "Model:    $MODEL"

# ── wait for vLLM ─────────────────────────────────────────────
log "Waiting for vLLM on port 8080..."
for i in $(seq 1 60); do
    curl -sf http://127.0.0.1:8080/v1/models > /dev/null 2>&1 && { log "vLLM ready."; break; }
    sleep 5
done
curl -sf http://127.0.0.1:8080/v1/models > /dev/null 2>&1 || { log "ERROR: vLLM not ready. Abort."; exit 1; }

# ── main loop ────────────────────────────────────────────────
for iter in $(seq 1 "$MAX_ITER"); do
    log "═══ Iteration $iter / $MAX_ITER ═══"

    BEST_AUC=$(grep "keep" results.tsv | awk '{print $2}' | sort -n | tail -1)
    log "Current best val_auc = $BEST_AUC"

    # OpenCode is the agent: reads files, proposes change, writes train.py via bash
    PROMPT="You are an AI research agent on an NVIDIA DGX Spark.

Your goal: improve chest X-ray disease classification. Current best val_auc=${BEST_AUC}.

Steps you MUST follow:
1. Read /home/nvidia/autoresearch/train.py
2. Read /home/nvidia/autoresearch/results.tsv
3. Read /home/nvidia/autoresearch/program.md
4. Choose ONE specific improvement likely to increase val_auc above ${BEST_AUC}
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
        log "WARNING: OpenCode exited with code $OPENCODE_EXIT. Skipping iteration."
        git checkout train.py 2>/dev/null || true
        continue
    fi

    # Syntax-check train.py before committing
    if ! python3 -m py_compile train.py 2>/tmp/syntax_err.txt; then
        log "WARNING: train.py has syntax errors — skipping iteration."
        log "$(cat /tmp/syntax_err.txt)"
        git checkout train.py 2>/dev/null || true
        continue
    fi

    # Check train.py was actually modified
    if git diff --quiet train.py; then
        log "WARNING: train.py unchanged after OpenCode run. Skipping."
        continue
    fi

    # Commit
    HEADLINE=$(head -5 train.py | grep -i '#\|"""\|experiment' | head -1 | sed 's/[#"]*//g' | xargs)
    git add train.py
    git commit -m "iter-${iter}: ${HEADLINE:-LLM improvement}" || { log "Commit failed. Skipping."; continue; }
    COMMIT=$(git rev-parse --short HEAD)
    log "Committed $COMMIT"

    # Unload Ollama if running to free VRAM for training
    ollama stop 2>/dev/null || true

    # Train for 5 minutes
    log "Training (5-minute budget)..."
    $UV run train.py > run.log 2>&1
    log "Training complete."

    # Parse metrics
    VAL_AUC=$(grep "^val_auc:" run.log | awk '{print $2}')
    PEAK_MB=$(grep "^peak_vram_mb:" run.log | awk '{print $2}')
    VRAM_GB=$(python3 -c "print(f'{float(\"${PEAK_MB:-0}\")/1024:.1f}')")

    log "val_auc=$VAL_AUC  best_so_far=$BEST_AUC"

    # Keep or discard
    BETTER=$(python3 -c "print('yes' if float('${VAL_AUC:-0}') > float('$BEST_AUC') else 'no')")
    if [ "$BETTER" = "yes" ]; then
        STATUS="keep"
        log "✓ NEW BEST — keeping."
    else
        STATUS="discard"
        log "✗ No improvement — reverting."
        git reset HEAD~1
        git checkout train.py
    fi

    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$COMMIT" "$VAL_AUC" "$VRAM_GB" "$STATUS" "iter-${iter}" >> results.tsv

    echo ""
    cat results.tsv
    $UV run /home/nvidia/autoresearch/plot_results.py results.tsv results_chart.png 2>/dev/null && log "Chart updated: results_chart.png"
    echo ""
done

log "Loop complete — $MAX_ITER iterations done."
