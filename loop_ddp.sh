#!/bin/bash
# ============================================================
# loop_ddp.sh — Autonomous 4-node DDP research loop
#
# Runs on node-0 only. OpenCode (Qwen3.5-122B) rewrites
# train_ddp.py each iteration, then 4-node DDP training
# runs across all nodes via NCCL AllReduce.
#
# Usage (on node-0):
#   ./loop_ddp.sh [max_iterations] [epochs_per_iter]
#
# Example:
#   ./loop_ddp.sh 20 3    # 20 iterations, 3 epochs each
# ============================================================

REPO_DIR="/home/nvidia/autoresearch"
UV="/home/nvidia/.local/bin/uv"
PYTHON="$REPO_DIR/.venv/bin/python3"
OPENCODE=$(find /home/nvidia/.bun/install/global/node_modules/opencode-linux-arm64/bin \
                 /home/nvidia/.local/bin \
                 /home/nvidia/.local/share/fnm \
                 -name opencode -type f 2>/dev/null | head -1)
MODEL="vllm//home/nvidia/models/Qwen3.5-122B-A10B-AWQ"
MAX_ITER="${1:-20}"
EPOCHS_PER_ITER="${2:-3}"

# ── Cluster config ────────────────────────────────────────────
MASTER_ADDR="10.137.203.228"   # node-0 management IP (TCPStore rendezvous)
MASTER_PORT=29500
WORLD_SIZE=4

declare -A NODE_MGT_IPS=(
    [1]="10.137.203.184"
    [2]="10.137.203.174"
    [3]="10.137.203.177"
)
SSH_KEY="/home/nvidia/.ssh/id_ed25519_shared"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10"

cd "$REPO_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── sync train_ddp.py to all remote nodes ──────────────────
sync_to_nodes() {
    local file="${1:-train_ddp.py}"
    for rank in 1 2 3; do
        scp $SSH_OPTS \
            "$REPO_DIR/$file" \
            "nvidia@${NODE_MGT_IPS[$rank]}:$REPO_DIR/$file" &
    done
    wait
    log "$file synced to all nodes."
}

# ── kill stale training processes on all nodes ──────────────
cleanup_nodes() {
    log "Cleaning up stale processes on all nodes..."
    pkill -f train_ddp.py 2>/dev/null; fuser -k ${MASTER_PORT}/tcp 2>/dev/null; true
    for rank in 1 2 3; do
        ssh $SSH_OPTS "nvidia@${NODE_MGT_IPS[$rank]}" \
            "pkill -f train_ddp.py 2>/dev/null; fuser -k ${MASTER_PORT}/tcp 2>/dev/null; true" &
    done
    wait
    sleep 3
    log "Cleanup done."
}

# ── launch 4-node DDP: remote ranks in background, rank-0 locally ──
run_ddp() {
    local epochs="$1"
    local port="${MASTER_PORT}"

    # Remote nodes (ranks 1-3) in background
    for rank in 1 2 3; do
        ssh $SSH_OPTS "nvidia@${NODE_MGT_IPS[$rank]}" \
            "cd $REPO_DIR && \
             RANK=$rank LOCAL_RANK=0 WORLD_SIZE=$WORLD_SIZE \
             MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$port \
             NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_DEBUG=WARN \
             USE_HF_NIH=1 MAX_EPOCHS=$epochs \
             $PYTHON train_ddp.py > /tmp/ddp_rank${rank}.log 2>&1" &
    done

    # Rank-0 runs locally — output captured to run.log
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=$WORLD_SIZE \
        MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$port \
        NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_DEBUG=WARN \
        USE_HF_NIH=1 MAX_EPOCHS=$epochs \
        $PYTHON train_ddp.py 2>&1 | tee run.log
    local exit_code=${PIPESTATUS[0]}

    wait   # wait for all remote nodes to finish
    return $exit_code
}

# ── banner ───────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  MODE : 4-NODE DDP RESEARCH LOOP (node-0 master)    ║"
echo "║  LLM  : Qwen3.5-122B  via  OpenCode                 ║"
echo "║  DATA : NIH ChestX-ray14 (HuggingFace)              ║"
echo "║  ITER : ${MAX_ITER} iterations · ${EPOCHS_PER_ITER} epochs/iter              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── prerequisites ─────────────────────────────────────────────
[ -z "$OPENCODE" ] && { log "ERROR: opencode binary not found"; exit 1; }
[ -f "$SSH_KEY" ] || { log "ERROR: SSH key not found: $SSH_KEY"; exit 1; }
log "OpenCode : $OPENCODE"
log "Model    : $MODEL"
log "Max iter : $MAX_ITER  |  Epochs/iter: $EPOCHS_PER_ITER"

# ── ensure vLLM is running — restart if crashed ───────────────
VLLM_START_SCRIPT="/tmp/start_qwen_vllm.sh"
VLLM_LOG="/tmp/vllm_qwen.log"

if curl -sf http://127.0.0.1:8080/v1/models > /dev/null 2>&1; then
    log "vLLM already ready on :8080."
else
    log "vLLM not responding on :8080 — checking process..."
    if pgrep -f "vllm serve" > /dev/null 2>&1; then
        log "vLLM process exists but not ready yet — will wait."
    else
        log "vLLM process not found — restarting..."
        pkill -9 -f "vllm" 2>/dev/null; sleep 2
        if [ -f "$VLLM_START_SCRIPT" ]; then
            log "Starting vLLM via $VLLM_START_SCRIPT ..."
            bash "$VLLM_START_SCRIPT" &
        else
            log "Start script not found — launching vLLM with defaults..."
            export MAX_JOBS=4
            /home/nvidia/vllm-env/bin/vllm serve /home/nvidia/models/Qwen3.5-122B-A10B-AWQ \
                --port 8080 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
                --max-model-len 65536 --gpu-memory-utilization 0.85 \
                --compilation-config '{"mode": 0}' >> "$VLLM_LOG" 2>&1 &
        fi
        log "vLLM restart initiated (PID $!) — waiting for it to become ready..."
    fi
fi

# ── wait for vLLM on node-0 ───────────────────────────────────
log "Waiting for vLLM on :8080 (up to 35 min for large model load)..."
for i in $(seq 1 420); do
    curl -sf http://127.0.0.1:8080/v1/models > /dev/null 2>&1 && { log "vLLM ready."; break; }
    [ $((i % 12)) -eq 0 ] && log "  still waiting... ${i}/420 attempts ($(( i*5/60 ))m elapsed)"
    sleep 5
done
curl -sf http://127.0.0.1:8080/v1/models > /dev/null 2>&1 || { log "ERROR: vLLM not ready after 35 min. Abort."; exit 1; }

# ── save baseline (once) ──────────────────────────────────────
BASELINE="$REPO_DIR/train_ddp.py.baseline"
if [ ! -f "$BASELINE" ]; then
    cp "$REPO_DIR/train_ddp.py" "$BASELINE"
    log "Baseline saved to train_ddp.py.baseline"
else
    log "Baseline exists: train_ddp.py.baseline"
fi

# ── best code — always reset to baseline at loop start ────────
# Within this run, BEST_CODE accumulates improvements iteration by iteration.
# A new loop invocation always starts fresh from baseline + BiomedCLIP pretrained.
BEST_CODE="$REPO_DIR/train_ddp.py.best"
cp "$BASELINE" "$BEST_CODE"
log "Best code reset to baseline for this run."

# Clear best_model.pth so train_ddp.py starts from BiomedCLIP pretrained weights
rm -f "$REPO_DIR/best_model.pth"
for rank in 1 2 3; do
    ssh $SSH_OPTS "nvidia@${NODE_MGT_IPS[$rank]}" "rm -f $REPO_DIR/best_model.pth" \
        || log "WARNING: could not clear best_model.pth on node-${rank} — may use stale weights on first iter" &
done
wait
log "best_model.pth cleared on all nodes — starting from BiomedCLIP pretrained weights."

# ── main loop ─────────────────────────────────────────────────
for iter in $(seq 1 "$MAX_ITER"); do
    log "══════ Iteration $iter / $MAX_ITER ══════"

    # Reset to best code so OpenCode builds on the best result so far
    cp "$BEST_CODE" "$REPO_DIR/train_ddp.py"
    log "Reset train_ddp.py to best code."

    # Global best from results.tsv
    BEST_AUC=$(awk -F'\t' '$4 == "keep" {print $2}' results.tsv 2>/dev/null | sort -n | tail -1)
    BEST_AUC="${BEST_AUC:-0.0}"
    TRIED=$(awk -F'\t' '$4 == "keep" || $4 == "discard"' results.tsv 2>/dev/null | wc -l || echo 0)
    log "Best val_auc=$BEST_AUC  |  Total experiments=$TRIED"

    # ── OpenCode prompt ───────────────────────────────────────
    PROMPT="You are an AI research agent running on a 4-node NVIDIA DGX Spark cluster.

Your goal: improve chest X-ray disease classification on NIH ChestX-ray14 (HuggingFace).
Metric: val_auc (mean AUC-ROC across 14 diseases, higher is better).
Current best val_auc=${BEST_AUC}. Target: beat CheXNet benchmark of 0.841.

IMPORTANT CONTEXT:
- train_ddp.py already contains the BEST code from all previous iterations (not the original baseline).
- best_model.pth contains the BEST weights from all previous training runs and will be loaded
  automatically at the start of training via strict=False. You do NOT need to load it yourself.
- Each iteration builds on the best result so far — the model keeps improving.
- Choose an improvement that builds on what is already working, not a completely different approach.

Steps you MUST follow exactly:
1. Read /home/nvidia/autoresearch/train_ddp.py
2. Read /home/nvidia/autoresearch/results.tsv
3. Read /home/nvidia/autoresearch/program_ddp.md
4. Choose ONE specific improvement not yet tried that could beat val_auc=${BEST_AUC}
5. Use the Edit tool to modify ONLY the code inside the EXPERIMENT section markers in
   /home/nvidia/autoresearch/train_ddp.py. The four editable sections are:
     CLASSIFIER_HEAD  — model architecture / head layers
     TRANSFORMS       — data augmentation (PIL.Image input, must stay tensor-compatible)
     OPTIMIZER        — optimizer, scheduler, criterion, lr, weight_decay
     TRAIN_STEP       — per-batch forward / backward / grad-clip / optimizer step
   Edit one or more sections. Leave all other code untouched.
6. Stop. Do not run training yourself.

IMPORTANT: Use the Edit tool only (step 5).
DO NOT rewrite the entire file. DO NOT use bash cat > to overwrite train_ddp.py."

    log "Running OpenCode agent (iter $iter)..."
    "$OPENCODE" run --model "$MODEL" "$PROMPT"
    OPENCODE_EXIT=$?

    if [ $OPENCODE_EXIT -ne 0 ]; then
        log "WARNING: OpenCode exited $OPENCODE_EXIT — skipping."
        cp "$BEST_CODE" "$REPO_DIR/train_ddp.py"
        continue
    fi

    # ── syntax check ─────────────────────────────────────────
    if ! python3 -m py_compile train_ddp.py 2>/tmp/syntax_err.txt; then
        log "WARNING: syntax error in train_ddp.py — skipping."
        log "$(cat /tmp/syntax_err.txt)"
        cp "$BEST_CODE" "$REPO_DIR/train_ddp.py"
        continue
    fi

    # ── unchanged check ───────────────────────────────────────
    if diff -q "$BEST_CODE" train_ddp.py > /dev/null 2>&1; then
        log "WARNING: train_ddp.py unchanged from best code — skipping."
        continue
    fi

    # ── save experiment code (train_ddp.py is never committed to git) ────────
    HEADLINE=$(head -5 train_ddp.py | grep -i '#\|"""\|experiment' | head -1 \
               | sed 's/[#"]*//g' | xargs)
    mkdir -p experiments
    EXP_FILE="experiments/iter-${iter}.py"
    cp train_ddp.py "$EXP_FILE"
    git add "$EXP_FILE" results.tsv 2>/dev/null || true
    git commit -m "iter-${iter}: ${HEADLINE:-LLM improvement}" \
        || { log "Commit failed — skipping."; continue; }
    COMMIT=$(git rev-parse --short HEAD)
    log "Saved experiment to $EXP_FILE ($COMMIT)"

    # ── sync to all nodes ─────────────────────────────────────
    sync_to_nodes "train_ddp.py"

    # ── clean up stale processes before DDP ──────────────────
    cleanup_nodes

    # ── launch 4-node DDP training ────────────────────────────
    log "Launching 4-node DDP training (${EPOCHS_PER_ITER} epochs)..."
    run_ddp "$EPOCHS_PER_ITER"
    TRAIN_EXIT=$?
    log "Training complete (exit=$TRAIN_EXIT)."

    # ── crash detection ───────────────────────────────────────
    if [ "$TRAIN_EXIT" -ne 0 ]; then
        CRASH_MSG=$(grep -m1 "Error:\|Traceback\|RuntimeError" run.log 2>/dev/null || echo "unknown")
        log "WARNING: Training CRASHED — reverting. Reason: $CRASH_MSG"
        cp "$BEST_CODE" "$REPO_DIR/train_ddp.py"
        sync_to_nodes "train_ddp.py"
        continue
    fi

    # ── parse metrics ─────────────────────────────────────────
    VAL_AUC=$(grep -Eo 'val_auc[^0-9]+[0-9]+\.[0-9]+' run.log | grep -Eo '[0-9]+\.[0-9]+' | head -1)
    PEAK_MB=$(grep -Eo 'peak_vram_mb[=: ]+[0-9]+' run.log | grep -Eo '[0-9]+$' | head -1)
    VRAM_GB=$(python3 -c "print(f'{float(\"${PEAK_MB:-0}\")/1024:.1f}')" 2>/dev/null || echo "?")

    if [ -z "$VAL_AUC" ]; then
        log "WARNING: val_auc not found in run.log — skipping."
        log "Last 5 lines: $(tail -5 run.log | tr '\n' '|')"
        cp "$BEST_CODE" "$REPO_DIR/train_ddp.py"
        sync_to_nodes "train_ddp.py"
        continue
    fi

    log "val_auc=$VAL_AUC  |  best=$BEST_AUC"

    BETTER=$(python3 -c "
try:    print('yes' if float('${VAL_AUC:-0}') > float('$BEST_AUC') else 'no')
except: print('no')
")

    if [ "$BETTER" = "yes" ]; then
        STATUS="keep"
        log "NEW BEST — keeping $COMMIT ($EXP_FILE)."
        # Update best code and best model weights for next iteration
        cp train_ddp.py "$BEST_CODE"
        cp "$REPO_DIR/trained_model.pth" "$REPO_DIR/best_model.pth" 2>/dev/null && {
            sync_to_nodes "best_model.pth"
            log "best_model.pth updated and synced."
        } || log "WARNING: trained_model.pth not found — best_model.pth not updated."
    else
        STATUS="discard"
        log "No improvement — best code restored."
        cp "$BEST_CODE" "$REPO_DIR/train_ddp.py"
        sync_to_nodes "train_ddp.py"
    fi

    DESC="iter-${iter}: ${HEADLINE:-improvement}"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$COMMIT" "${VAL_AUC:-0.000000}" "$VRAM_GB" "$STATUS" "$DESC" \
        >> results.tsv

    $UV run "$REPO_DIR/plot_results.py" results.tsv results_chart.png 2>/dev/null \
        && log "Chart updated."

    echo ""
done

log "Loop complete — $MAX_ITER iterations done."
