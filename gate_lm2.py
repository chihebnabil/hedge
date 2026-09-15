"""
gate_lm2.py — Stage 2: ablations + fair-data control + stronger training.
Runs on the cached per-position datasets from gate_lm.py (no recollection).

E1 ablations:   bucket-only | evidence-only | votes-only | all features
                (all feature blocks are causal/candidate-independent)
E2 fair data:   gate trained on VALID only (193k) — apples-to-apples
                against the online mixer that was warmed on valid
E3 stronger:    hidden sweep, more epochs, weight decay, 5-seed ensemble
"""

import math
import os

import numpy as np

from gate_lm import EVID_COLS, NF, VOTE_COLS

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("HEDGE_RESULTS",
                           os.path.join(HERE, "benchmarks", "results"))
NEXP = 5


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll_of(A, L):
    P = np.exp(L.astype(np.float64))
    return float(np.mean(-np.log(np.maximum(np.sum(A * P, axis=1), 1e-300))))


def train_gate(X, L, hidden=64, epochs=8, bs=4096, lr=3e-3, wd=1e-4,
               seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    N, F = X.shape
    W1 = rng.normal(0, (2 / F) ** 0.5, (F, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, (2 / hidden) ** 0.5, (hidden, NEXP))
    b2 = np.zeros(NEXP)
    params = [W1, b1, W2, b2]
    mts = [np.zeros_like(p) for p in params]
    vts = [np.zeros_like(p) for p in params]
    P = np.exp(L.astype(np.float64))
    step = 0
    for ep in range(epochs):
        idx = rng.permutation(N)
        tot = 0.0
        for s in range(0, N, bs):
            j = idx[s:s + bs]
            xb, Pb = X[j].astype(np.float64), P[j]
            h = np.maximum(xb @ W1 + b1, 0.0)
            a = softmax(h @ W2 + b2)
            pm = np.maximum(np.sum(a * Pb, axis=1, keepdims=True), 1e-300)
            tot += float(np.mean(-np.log(pm))) * len(j)
            g = a * (1.0 - Pb / pm)
            gW2 = h.T @ g / len(j) + wd * W2
            gb2 = g.mean(axis=0)
            gh = (g @ W2.T) * (h > 0)
            gW1 = xb.T @ gh / len(j) + wd * W1
            gb1 = gh.mean(axis=0)
            step += 1
            for p, gdp, mt, vt in zip(params, (gW1, gb1, gW2, gb2), mts, vts):
                mt *= 0.9
                mt += 0.1 * gdp
                vt *= 0.999
                vt += 0.001 * gdp * gdp
                p -= lr * (mt / (1 - 0.9 ** step)) / \
                     (np.sqrt(vt / (1 - 0.999 ** step)) + 1e-8)
        if verbose:
            print(f"    ep{ep+1} train NLL {tot/N:.4f}", flush=True)

    def predict(Xb):
        h = np.maximum(Xb @ W1 + b1, 0.0)
        return softmax(h @ W2 + b2)

    return predict


def report(name, A, L):
    pp = math.exp(nll_of(A, L))
    print(f"  {name:<44} PP={pp:8.2f}")
    return pp


def main():
    d = {}
    for k in ("train", "valid", "test"):
        z = np.load(os.path.join(CACHE_DIR, f"gate_{k}.npz"))
        d[k] = (z["X"], z["L"])
    Xtr, Ltr = d["train"]
    Xva, Lva = d["valid"]
    Xte, Lte = d["test"]

    FEATS = {
        f"all ({NF})": list(range(NF)),
        f"votes only ({VOTE_COLS[0]}-{VOTE_COLS[-1]})": VOTE_COLS,
        f"evidence only ({EVID_COLS[0]}-{EVID_COLS[-1]})": EVID_COLS,
        "bucket only (18-21)": [18, 19, 20, 21],
    }
    print("== E1: which features carry the routing signal? "
          "(train=full train, h=64, 8 epochs) ==")
    for name, cols in FEATS.items():
        pred = train_gate(Xtr[:, cols], Ltr, hidden=64, epochs=8)
        report(f"gate[{name}]", pred(Xte[:, cols]), Lte)

    print("== E2: fair-data control (train on VALID 193k, like the "
          "online mixer did) ==")
    pred = train_gate(Xva, Lva, hidden=64, epochs=15)
    report("gate[all], valid-trained, h=64", pred(Xte), Lte)
    pred = train_gate(Xva, Lva, hidden=256, epochs=25)
    report("gate[all], valid-trained, h=256", pred(Xte), Lte)

    print("== E3: stronger recipe on full train (wd, ensemble) ==")
    pred = train_gate(Xtr, Ltr, hidden=64, epochs=12, wd=1e-4, seed=1,
                      verbose=True)
    report("gate[all], h=64, 12ep, wd=1e-4", pred(Xte), Lte)
    preds = [train_gate(Xtr, Ltr, hidden=64, epochs=12, wd=1e-4, seed=s)
             for s in range(5)]
    A = np.mean([p(Xte) for p in preds], axis=0)
    report("gate[all], 5-seed ensemble", A, Lte)
    preds_va = [train_gate(Xva, Lva, hidden=256, epochs=25, seed=s)
                for s in range(5)]
    A = np.mean([p(Xte) for p in preds_va], axis=0)
    report("gate[all], valid-trained h=256, 5-seed ens", A, Lte)

    print("== anchors: online bucket mixer 199.50 | MLP rung-1 104.81 | "
          "oracle 89.11 ==")


if __name__ == "__main__":
    main()
