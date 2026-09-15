#!/usr/bin/env bash
# Regenerates every router-dependent number in the paper for ONE corpus.
#
#   bash scripts/run_paper_pipeline.sh wt2      # WikiText-2
#   bash scripts/run_paper_pipeline.sh ptb      # Penn Treebank
#
# Stages (each prints a table the paper quotes verbatim):
#   collect  causal, candidate-independent gate features -> gate_{split}.npz
#   money    the money table + the gate-size sweep (h selected on VALID only)
#   seeds    3-seed mean/sd + ensembles for the headline rows
#   select   the selective-GRU cost/quality frontier
#   energy   ms/token + joule proxy, interleaved rounds
#
# The GRU expert and its aligned scores (nn_logp_*.npy) are inputs, not
# outputs: retrain them with nn_expert.py first if they are missing.
# KenLM / 5-gram baselines are independent of the router: kenlm_baseline.py,
# fix_5gram.py.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

case "${1:-}" in
  wt2) RES="benchmarks/results";     LABEL="WT-2" ;;
  ptb) RES="benchmarks/results/ptb"; LABEL="PTB"  ;;
  *)   echo "usage: bash scripts/run_paper_pipeline.sh {wt2|ptb}"; exit 2 ;;
esac
export HEDGE_RESULTS="$RES"
stage() { echo; echo "=== [$LABEL] $* ==="; date -u "+    (%Y-%m-%dT%H:%M:%SZ)"; }

if [ "${SKIP_COLLECT:-0}" = "1" ] && [ -f "$RES/gate_train.npz" ]; then
  stage "collect SKIPPED (SKIP_COLLECT=1, npz present)"
else
  stage "collect causal gate features"
  python gate_lm.py --collect || exit 1
fi
python -m unittest tests.test_gate -v || exit 1

stage "money table + gate-size sweep"
python gate_hybrid.py | tee "$RES/money_table.log" || exit 1

stage "seed audit (3 seeds, mean/sd + ensembles)"
python seed_audit.py | tee "$RES/seed_audit.log" || exit 1

stage "selective GRU activation"
python benchmarks/selective_gru.py || exit 1

stage "router diagnostic (why routing does not transfer)"
python benchmarks/router_diagnostic.py || exit 1

stage "energy / latency harness"
python benchmarks/energy_harness.py | tee "$RES/energy.log" || exit 1

stage "normalization audit (current causal router, must be Z=1)"
python benchmarks/normalization_audit.py --positions 150 || exit 1

stage "normalization audit (invalidated v1 leaky router)"
python benchmarks/normalization_audit.py --legacy --positions 100 || exit 1

stage "footprint metadata"
python benchmarks/footprint_meta.py > "$RES/footprint.log" 2>&1 || exit 1
cat "$RES/nn_meta.json"

echo
echo "=== [$LABEL] pipeline complete ==="
