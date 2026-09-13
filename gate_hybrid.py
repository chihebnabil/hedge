"""
gate_hybrid.py — Phase 4: the money table.

Adds the trained GRU as expert #6 (the 6th column; "expert #7" in README
numbering) to the 5 frozen statistical experts and compares, on the SAME
217,004-position test stream and aligned gate npz rows:

  pure GRU                  standalone neural expert (aligned rows)
  gate[5 experts]           sanity row: re-trained without the GRU (~99.7)
  fixed-lambda KN3+GRU      classic interpolation, lambda* picked on VALID
                            (Sak 2013 / Levit 2023 lineage) — load-bearing
                            baseline: the gate must beat this or the claim
                            shrinks honestly (PROGRESS.md risk note)
  static-alpha (6 experts)  best single global mixture vector, train-fit
  GATE hybrid (6 experts)   the system: learned evidence-aware routing
  oracle (6 experts)        hindsight bound

Staleness guard: refuses to run against stale GRU scores — nn_logp_*.npy
must be at least as new as nn_gru_best.pt (regenerate with
`python nn_expert.py --score-only`). --mock exercises the full plumbing
with a synthetic GRU column (numbers are meaningless, shapes are real).

Usage:
  python gate_hybrid.py            # the money table (after GRU training)
  python gate_hybrid.py --mock     # plumbing dry-run (no torch, fast)
"""

import math
import os
import sys

import numpy as np

from gate_lm3 import age_invariant

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HEDGE_RESULTS",
                     os.path.join(HERE, "benchmarks", "results"))
NF = 20
NAMES = ["kn3", "lag2", "cache", "lag3", "uni", "gru"]
BEST = os.path.join(OUT, "nn_gru_best.pt")
COUNTS_PKL = os.path.join(HERE, "benchmarks", "results", "beast_trigram.pkl")
GRU_MB_FP32 = 32.9          # 8.2M params, fp32; fp16 = 16.4


# -- data --------------------------------------------------------------------- #

def load_split(split, mock=False):
    z = np.load(os.path.join(OUT, f"gate_{split}.npz"))
    X, L = z["X"], z["L"]
    n = len(X)
    if mock:
        rng = np.random.default_rng(0)
        gru = rng.normal(-8.0, 2.0, size=n).astype(np.float32)
    else:
        npy = os.path.join(OUT, f"nn_logp_{split}.npy")
        best_mt = os.path.getmtime(BEST)
        if not os.path.exists(npy) or os.path.getmtime(npy) < best_mt:
            sys.exit(f"nn_logp_{split}.npy stale or missing vs nn_gru_best.pt "
                     f"— run: python nn_expert.py --score-only")
        gru = np.load(npy)
        assert len(gru) == n, f"{split}: GRU rows {len(gru):,} != gate rows {n:,}"
    return age_invariant(X), np.concatenate([L, gru[:, None]], axis=1)


# -- models / training --------------------------------------------------------- #

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll_pp(A, L):
    P = np.exp(L.astype(np.float64))
    return math.exp(float(np.mean(-np.log(np.maximum(
        np.sum(A * P, axis=1), 1e-300)))))


def train_gate(X, L, nexp, hidden=64, epochs=12, bs=4096, lr=3e-3, wd=1e-4,
               seed=0, return_weights=False):
    """MLP gate over nexp expert votes; identical math to gate_lm2.train_gate."""
    rng = np.random.default_rng(seed)
    N, F = X.shape
    W1 = rng.normal(0, (2 / F) ** 0.5, (F, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, (2 / hidden) ** 0.5, (hidden, nexp))
    b2 = np.zeros(nexp)
    params = [W1, b1, W2, b2]
    mts = [np.zeros_like(p) for p in params]
    vts = [np.zeros_like(p) for p in params]
    P = np.exp(L.astype(np.float64))
    step = 0
    for _ in range(epochs):
        idx = rng.permutation(N)
        for s in range(0, N, bs):
            j = idx[s:s + bs]
            xb, Pb = X[j].astype(np.float64), P[j]
            h = np.maximum(xb @ W1 + b1, 0.0)
            a = softmax(h @ W2 + b2)
            pm = np.maximum(np.sum(a * Pb, axis=1, keepdims=True), 1e-300)
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

    def predict(Xb):
        h = np.maximum(Xb @ W1 + b1, 0.0)
        return softmax(h @ W2 + b2)

    kb = sum(p.size for p in params) * 4 / 1024
    if return_weights:
        return predict, kb, (W1, b1, W2, b2)
    return predict, kb


def train_static_alpha(L, nexp, epochs=8, lr=5e-2, seed=0):
    """Single global mixture vector (bias-only softmax), fit by NLL."""
    rng = np.random.default_rng(seed)
    b = np.zeros(nexp)
    mt = np.zeros(nexp)
    vt = np.zeros(nexp)
    P = np.exp(L.astype(np.float64))
    N = len(L)
    step = 0
    for _ in range(epochs):
        idx = rng.permutation(N)
        for s in range(0, N, 8192):
            j = idx[s:s + 8192]
            a = softmax(np.tile(b, (len(j), 1)))
            pm = np.maximum(np.sum(a * P[j], axis=1, keepdims=True), 1e-300)
            g = (a * (1.0 - P[j] / pm)).mean(axis=0)
            step += 1
            mt *= 0.9
            mt += 0.1 * g
            vt *= 0.999
            vt += 0.001 * g * g
            b -= lr * (mt / (1 - 0.9 ** step)) / \
                 (np.sqrt(vt / (1 - 0.999 ** step)) + 1e-8)
    return softmax(b[None, :])


def fixed_lambda(Lva, Lte, i=0, j=5, grid=201):
    """p = lam*p_i + (1-lam)*p_j ; lambda* on valid, honest test report."""
    lams = np.linspace(0.0, 1.0, grid)
    Pva_i = np.exp(Lva[:, i].astype(np.float64))
    Pva_j = np.exp(Lva[:, j].astype(np.float64))
    nlls = [float(np.mean(-np.log(np.maximum(l * Pva_i + (1 - l) * Pva_j,
                                             1e-300)))) for l in lams]
    lam = float(lams[int(np.argmin(nlls))])
    Pte_i = np.exp(Lte[:, i].astype(np.float64))
    Pte_j = np.exp(Lte[:, j].astype(np.float64))
    pp = math.exp(float(np.mean(-np.log(np.maximum(
        lam * Pte_i + (1 - lam) * Pte_j, 1e-300)))))
    return lam, pp


# -- table --------------------------------------------------------------------- #

def row(name, pp, extra=""):
    print(f"  {name:<34} PP={pp:9.2f}  {extra}")


def main(mock=False):
    print("loading splits (mock=..." if mock else "loading splits ...")
    Y, L = {}, {}
    for k in ("train", "valid", "test"):
        Y[k], L[k] = load_split(k, mock)
    nexp = L["train"].shape[1]
    print(f"  experts per position: {nexp}  "
          f"({', '.join(NAMES[:nexp])})")

    corpus = "PTB" if OUT.rstrip("/").endswith("ptb") else "WT-2"
    n_test = {"PTB": "75,623", "WT-2": "217,004"}[corpus]
    print(f"== the money table ({corpus} test, {n_test} aligned positions) ==")
    e5 = np.zeros((1, nexp)); e5[0, 5] = 1.0
    row("pure GRU", nll_pp(np.repeat(e5, len(L["test"]), 0), L["test"]),
        f"[{GRU_MB_FP32:.1f} MB fp32 / 16.4 fp16]")
    e0 = np.zeros((1, nexp)); e0[0, 0] = 1.0
    row("pure KN3 (anchor)", nll_pp(np.repeat(e0, len(L["test"]), 0),
                                    L["test"]), "[21.4 MB counts]")

    pred5, kb5 = train_gate(Y["train"], L["train"][:, :5], nexp=5)
    row("gate[5 experts]", nll_pp(pred5(Y["test"]), L["test"][:, :5]),
        f"[21.4 MB + gate {kb5:.1f} KB]")

    lam, pp = fixed_lambda(L["valid"], L["test"])
    row("fixed-lambda KN3+GRU", pp, f"[lambda*={lam:.2f} on valid, "
        f"21.4+{GRU_MB_FP32:.1f} MB]")

    a = train_static_alpha(L["train"], nexp)
    A = np.repeat(a, len(L["test"]), 0)
    row("static-alpha (6 experts)", nll_pp(A, L["test"]),
        "[alpha " + " ".join(f"{n} {x:.2f}" for n, x in zip(NAMES, a.ravel())) + "]")

    best = (None, 1e9, 0)
    for h in (24, 64, 128, 256):
        pred, kb = train_gate(Y["train"], L["train"], nexp, hidden=h)
        va = nll_pp(pred(Y["valid"]), L["valid"])
        print(f"    [gate h={h}: valid PP {va:.2f}]")
        if va < best[1]:
            best = (pred, va, kb)
    pp6 = nll_pp(best[0](Y["test"]), L["test"])
    row("GATE hybrid (6 experts)", pp6,
        f"[21.4 MB + gate {best[2]:.1f} KB + {GRU_MB_FP32:.1f} MB]")

    predv, kbv = train_gate(Y["valid"], L["valid"], nexp, hidden=64, epochs=15)
    row("GATE hybrid, valid-trained", nll_pp(predv(Y["test"]), L["test"]),
        "[fair-data control]")

    oracle = math.exp(float(np.mean(-L["test"].max(axis=1))))
    row("oracle (6 experts)", oracle, "[bound]")

    print("== verdict ==")
    gap = pp6 - oracle
    print(f"  gate vs oracle gap: {gap:+.2f} PP ({gap / oracle:.1%})")
    if pp6 < pp:
        print(f"  GATE BEATS fixed-lambda ({pp6:.2f} < {pp:.2f}) "
              f"— load-bearing claim survives on {corpus}")
    else:
        print(f"  GATE LOSES to fixed-lambda ({pp6:.2f} > {pp:.2f}) — "
              f"reposition as routing analysis (PROGRESS.md watchlist)")


if __name__ == "__main__":
    main(mock="--mock" in sys.argv)
