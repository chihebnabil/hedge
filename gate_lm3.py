"""
gate_lm3.py — Stage 2b: cache-age distribution shift + train-vs-valid control.

History: train-stream features were once collected with an always-full
512-token cache while valid/test start empty, so the raw cache features
(window size, raw count) encoded "how old is the cache" and shifted between
splits. The fix was the age-invariant density c/(t+1).

gate_lm.FeatMaker now emits those densities directly (features 16/17), so
`age_invariant` below is a validated pass-through kept for cached pipelines.
"""

import math
import os

import numpy as np

from gate_lm2 import report, train_gate

CACHE_DIR = os.environ.get(
    "HEDGE_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "benchmarks", "results"))
NEXP = 5


def age_invariant(X):
    """Pass-through: the collector already emits age-invariant cache
    densities (columns 16/17). Kept so cached npz pipelines stay compatible."""
    return X


def main():
    d = {}
    for k in ("train", "valid", "test"):
        z = np.load(os.path.join(CACHE_DIR, f"gate_{k}.npz"))
        d[k] = (z["X"], z["L"])
    Xtr, Ltr = d["train"]
    Xva, Lva = d["valid"]
    Xte, Lte = d["test"]

    Ytr, Yva, Yte = map(age_invariant, (Xtr, Xva, Xte))

    print("== age-invariant features: does train-trained catch up? ==")
    for name, h, ep in (("h=64, 12ep", 64, 12), ("h=128, 12ep", 128, 12),
                        ("h=256, 12ep", 256, 12)):
        pred = train_gate(Ytr, Ltr, hidden=h, epochs=ep, wd=1e-4)
        report(f"gate[AI] train-trained {name}", pred(Yte), Lte)
    for name, h, ep in (("h=64, 15ep", 64, 15), ("h=256, 25ep", 256, 25)):
        pred = train_gate(Yva, Lva, hidden=h, epochs=ep)
        report(f"gate[AI] valid-trained {name}", pred(Yte), Lte)

    print("== small ensembles on age-invariant features ==")
    preds = [train_gate(Ytr, Ltr, hidden=128, epochs=12, wd=1e-4, seed=s)
             for s in range(5)]
    report("gate[AI] train-trained h=128, 5-seed",
           np.mean([p(Yte) for p in preds], axis=0), Lte)
    preds = [train_gate(Yva, Lva, hidden=64, epochs=15, seed=s)
             for s in range(5)]
    report("gate[AI] valid-trained h=64, 5-seed",
           np.mean([p(Yte) for p in preds], axis=0), Lte)

    print("== votes + minimal evidence ==")
    cols = list(range(10)) + [16, 17, 18, 19, 20, 21]  # votes, densities, bucket
    pred = train_gate(Ytr[:, cols], Ltr, hidden=64, epochs=12, wd=1e-4)
    report("gate[votes+density+bucket] train-trained h=64",
           pred(Yte[:, cols]), Lte)

    print("== anchors: see PAPER_NOTES.md §0 (v1 anchors invalidated, §12) ==")


if __name__ == "__main__":
    main()
