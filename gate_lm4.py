"""
gate_lm4.py — confirm the headline result (valid-trained h=64 gate on
age-invariant features ≈ oracle) across seeds, plus one train-trained row.
Keeps runtime to a few minutes with per-epoch progress.
"""

import math
import os

import numpy as np

from gate_lm2 import train_gate
from gate_lm3 import age_invariant

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "benchmarks", "results")


def nll_pp(A, L):
    P = np.exp(L.astype(np.float64))
    return math.exp(float(np.mean(-np.log(np.maximum(np.sum(A * P, axis=1),
                                                     1e-300)))))


def main():
    d = {}
    for k in ("valid", "test"):
        z = np.load(os.path.join(CACHE_DIR, f"gate_{k}.npz"))
        d[k] = (age_invariant(z["X"]), z["L"])
    Yva, Lva = d["valid"]
    Yte, Lte = d["test"]
    oracle = math.exp(float(np.mean(-Lte.max(axis=1))))
    print(f"oracle PP={oracle:.2f}", flush=True)

    print("== valid-trained h=64 (the headline config), 3 seeds ==", flush=True)
    pps = []
    for s in range(3):
        pred = train_gate(Yva, Lva, hidden=64, epochs=15, seed=s)
        pp = nll_pp(pred(Yte), Lte)
        pps.append(pp)
        print(f"  seed {s}: test PP={pp:.2f}", flush=True)
    m = float(np.mean(pps))
    sd = float(np.std(pps))
    print(f"  => {m:.2f} ± {sd:.2f}  (oracle {oracle:.2f})", flush=True)

    print("== train-trained h=64, 8 epochs (for the table) ==", flush=True)
    z = np.load(os.path.join(CACHE_DIR, "gate_train.npz"))
    Ytr, Ltr = age_invariant(z["X"]), z["L"]
    pred = train_gate(Ytr, Ltr, hidden=64, epochs=8, wd=1e-4, seed=0)
    print(f"  test PP={nll_pp(pred(Yte), Lte):.2f}", flush=True)
    print("== anchors: see PAPER_NOTES.md §0 (v1 anchors invalidated, "
          "§12) ==", flush=True)


if __name__ == "__main__":
    main()
