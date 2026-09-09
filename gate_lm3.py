"""
gate_lm3.py — Stage 2b: fix the cache-age distribution shift.
Train-stream features were collected with an always-full 512-token cache;
valid/test start empty. Features 11 (log window size) and 12 (log count)
leak "how old is the cache". Replace with the age-invariant density
c/(t+1) and drop t. Then re-run the train-vs-valid training comparison.
"""

import math
import os

import numpy as np

from gate_lm2 import report, train_gate

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "benchmarks", "results")
NEXP = 5


def age_invariant(X):
    X2 = X.copy()
    t = np.expm1(X[:, 11].astype(np.float64))
    c = np.expm1(X[:, 12].astype(np.float64))
    X2[:, 11] = 0.0                       # drop window-age feature
    X2[:, 12] = np.log1p(c / (t + 1.0))   # cache density, age-free
    return X2


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

    print("== votes + minimal evidence on age-invariant features ==")
    cols = [0, 1, 2, 3, 4, 12, 13, 14, 15, 16]   # votes, cache density, bucket
    pred = train_gate(Ytr[:, cols], Ltr, hidden=64, epochs=12, wd=1e-4)
    report("gate[AI, votes+density+bucket] train-trained h=64",
           pred(Yte[:, cols]), Lte)

    print("== anchors: online bucket mixer 199.50 | rung-1 104.81 | "
          "rung-2 best 95.26 | oracle 89.11 ==")


if __name__ == "__main__":
    main()
