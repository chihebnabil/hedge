"""
seed_audit.py — 3-seed ±sd for the Phase-4 money-table rows (reviewer
requirement: headline rows need error bars).

Rows audited (EXACT configs from gate_hybrid.py's table):
  GATE(6), train-trained, h=128, 12 ep, wd=1e-4   (headline row, 90.28 @seed0)
  GATE(6), valid-trained, h=64, 15 ep             (fair-data control, 77.49 @seed0)
  static-alpha (6 experts)                        (cheap — full error bars)

fixed-λ needs no audit (deterministic grid on valid).

Usage:
  python seed_audit.py              # 3 seeds
  python seed_audit.py --seeds 5    # more seeds
"""

import os
import sys
import numpy as np

from gate_hybrid import load_split, train_gate, train_static_alpha, nll_pp


def audit(name, fit_fn, seeds):
    pps = []
    As = []
    for s in range(seeds):
        A = fit_fn(s)
        As.append(A)
        pp = nll_pp(A, Lte)
        pps.append(pp)
        print(f"  [{name}] seed {s}: test PP {pp:.2f}", flush=True)
    m = float(np.mean(pps))
    sd = float(np.std(pps, ddof=1)) if seeds > 1 else 0.0
    print(f"  => {name}: {m:.2f} ± {sd:.2f} ({seeds} seeds)", flush=True)
    if seeds > 1:
        ens = nll_pp(np.mean(As, axis=0), Lte)
        print(f"  => {name}: {seeds}-gate ENSEMBLE PP {ens:.2f}", flush=True)
    return pps


if __name__ == "__main__":
    seeds = 3
    if "--seeds" in sys.argv:
        seeds = int(sys.argv[sys.argv.index("--seeds") + 1])

    data = {k: load_split(k) for k in ("train", "valid", "test")}
    (Ytr, Ltr), (Yva, Lva), (Yte, Lte) = (data["train"], data["valid"],
                                          data["test"])
    nexp = Ltr.shape[1]

    print(f"== seed audit: GATE(6) train-trained "
          f"h={os.environ.get('HEDGE_GATE_H', '128')}, 12 ep ==", flush=True)
    audit("gate6-train", lambda s: train_gate(
        Ytr, Ltr, nexp,
        hidden=int(os.environ.get("HEDGE_GATE_H", "128")),
        epochs=12, seed=s)[0](Yte), seeds)

    print("== seed audit: GATE(6) valid-trained h=64, 15 ep ==", flush=True)
    audit("gate6-valid", lambda s: train_gate(
        Yva, Lva, nexp, hidden=64, epochs=15, seed=s)[0](Yte), seeds)

    print("== seed audit: static-alpha (6 experts) ==", flush=True)
    audit("static-alpha", lambda s: np.repeat(
        train_static_alpha(Ltr, nexp, seed=s), len(Yte), 0), seeds)

    if os.environ.get("HEDGE_RESULTS", "").rstrip("/").endswith("ptb"):
        print("== anchors (seed 0, ptb money_table.log): gate6-train 78.81 | "
              "gate6-valid 63.93 | static-alpha 101.74 | fixed-lambda 92.65 ==",
              flush=True)
    else:
        print("== anchors (seed 0, from gate_hybrid table): gate6-train 90.28 | "
              "gate6-valid 77.49 | static-alpha 170.97 ==", flush=True)
