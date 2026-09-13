#!/bin/bash
# Auto-resume watchdog — runs on every codespace container start
# (devcontainer.json postStartCommand). If the PTB GRU training was
# interrupted (idle-timeout suspend), relaunch it detached; it resumes
# from nn_gru_last.pt (epoch + optimizer + scheduler) automatically.
#
# Rules:
#   - a running nn_expert.py  -> do nothing
#   - log says ALL EPOCHS DONE -> training finished, do nothing
#   - no PTB log at all        -> training never started here, do nothing
#     (only ever RESUME, never cold-start on unrelated restarts)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/benchmarks/results/ptb/nn_train.log"

pgrep -f "nn_expert.py" > /dev/null && exit 0
[ -f "$LOG" ] || exit 0
grep -q "ALL EPOCHS DONE" "$LOG" && exit 0

cd "$ROOT" || exit 1
export HEDGE_RESULTS="$ROOT/benchmarks/results/ptb"
export HEDGE_WARMUP=442
setsid nohup python -u nn_expert.py >> "$LOG" 2>&1 < /dev/null &
echo "[auto_resume] $(date '+%F %T') relaunched nn_expert.py (resume from last checkpoint)" >> "$LOG"
