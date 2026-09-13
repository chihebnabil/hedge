"""
Fix the 5-gram baseline anomaly (Phase 6).

The original `--stage five` tuned the 5-gram with discount_scale frozen at
1.0 (3 combos) while the trigram got a 30-combo grid and chose scale=1.15.
Since the repo's discounts scale linearly with `discount_scale` (see
PAPER_NOTES §9 knob warning), the 302.5 test PP row was a tuning artifact,
not a fair comparison. This script re-runs the 5-gram with the SAME grid and
the SAME 25k valid subsample the trigram got, appends a corrected row to
results.json, and leaves the original row in place for transparency.

    python benchmarks/fix_5gram.py
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_benchmark as rb  # noqa: E402
from ZipfNextWordPredictor import ZipfNextWordPredictor  # noqa: E402


def main():
    g = rb.load_gold()
    train_lines = [" ".join(s) for s in g["train"]]

    if os.path.exists(rb.BEAST5_PKL):
        m = ZipfNextWordPredictor.load(rb.BEAST5_PKL)
        train_s = 0.0
    else:
        t0 = time.perf_counter()
        m = ZipfNextWordPredictor(corpus_texts=train_lines, n_gram_size=5,
                                  smoothing="beast")
        train_s = time.perf_counter() - t0
        m.save(rb.BEAST5_PKL)
        print(f"  trained 5-gram in {train_s:.1f}s")

    m.personalization = 0.0
    # fair fight: the trigram's exact tune protocol
    rep = m.tune(rb.subsample_text_25k(g["valid"]),
                 discount_scales=[0.85, 1.0, 1.15],
                 zipf_prior_weights=[0.0, 0.05, 0.1, 0.2, 0.3],
                 adaptive_options=[True, False])
    b = rep["best"]
    print(f"  tune: {rep['trials_run']} combos -> scale={b['discount_scale']} "
          f"z={b['zipf_prior_weight']} adaptive={b['adaptive_prior']}")

    metrics = rb.eval_predictor(m, g["test"], g["valid"])
    metrics["suggestion_ms"] = rb.suggestion_latency(
        m, rb.latency_contexts(g["test"]))
    rb.record("BEAST 5-gram (fair tune)", "beast(n=5)", train_s, metrics,
              model_path=rb.BEAST5_PKL,
              note=(f"same grid+subsample as trigram: scale={b['discount_scale']}, "
                    f"z={b['zipf_prior_weight']}, adaptive={b['adaptive_prior']}"))
    print(f"  corrected 5-gram: test PP {metrics['perplexity']:.2f} "
          f"(was 302.52 frozen-scale)")
    os.remove(rb.BEAST5_PKL)


if __name__ == "__main__":
    main()
