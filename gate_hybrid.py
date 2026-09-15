"""
gate_hybrid.py — Phase 4: the money table.

Adds the trained GRU as expert #6 (the 6th column; "expert #7" in README
numbering) to the 5 frozen statistical experts and compares, on the SAME
217,004-position test stream and aligned gate npz rows:

  pure GRU                  standalone neural expert (aligned rows)
  gate[5 experts]           sanity row: re-trained without the GRU
  fixed-lambda KN3+GRU      classic interpolation (Mikolov et al. 2011 /
                            Sundermeyer et al. 2012 lineage): p = lam*p_kn3
                            + (1-lam)*p_gru, lambda* picked on VALID —
                            load-bearing baseline: the gate must beat this or
                            the claim shrinks honestly (PROGRESS.md risk note)
  static-alpha (6 experts)  best single global mixture vector, fit on TRAIN
                            and (fair) on VALID — same selection budget as
                            lambda* and the gate's hidden size
  GATE hybrid (6 experts)   the system: learned evidence-aware routing
  best single expert        normalized hindsight bound over expert SELECTION
  per-position best expert  UNNORMALIZED diagnostic (mass > 1 over the vocab);
                            no normalized mixture can or should match it

Every row except the last is a normalized distribution over the vocabulary,
so the perplexities are mutually comparable. load_split() asserts that no
feature column reproduces a label column (the bug that invalidated the first
version of this table — see gate_lm.py's docstring and tests/test_gate.py).

Staleness guard: refuses to run against stale GRU scores — nn_logp_*.npy
must be at least as new as nn_gru_best.pt (regenerate with
`python nn_expert.py --score-only`). --mock exercises the full plumbing
with a synthetic GRU column (numbers are meaningless, shapes are real).

Usage:
  python gate_hybrid.py            # the money table (after GRU training)
  python gate_hybrid.py --mock     # plumbing dry-run (no torch, fast)
"""

import json
import math
import os
import sys

import numpy as np

from gate_lm import NF
from gate_lm3 import age_invariant

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HEDGE_RESULTS",
                     os.path.join(HERE, "benchmarks", "results"))
NAMES = ["kn3", "lag2", "cache", "lag3", "uni", "gru"]
BEST = os.path.join(OUT, "nn_gru_best.pt")
COUNTS_PKL = os.path.join(OUT, "beast_trigram.pkl")
GRU_MB_FP32 = 32.9          # WT-2 fallback: 8.2M unique params, fp32, tied


def selected_h(default=128):
    """Hidden size the money table selected on validation (HEDGE_GATE_H wins)."""
    env = os.environ.get("HEDGE_GATE_H")
    if env:
        return int(env)
    path = os.path.join(OUT, "gate_selected.json")
    if os.path.exists(path):
        with open(path) as f:
            return int(json.load(f)["h"])
    return default


def tri_mb():
    """Trigram counts pickle for the active corpus (MB)."""
    try:
        return os.path.getsize(COUNTS_PKL) / 1e6
    except OSError:
        return 0.0


def gru_mb():
    """GRU footprint (MB, fp32) for the active corpus: unique parameters,
    embeddings tied (nn_gru_best.pt clones the tied decoder, so the file is
    ~2x the in-memory footprint). Written by benchmarks/footprint_meta.py."""
    meta = os.path.join(OUT, "nn_meta.json")
    if not os.path.exists(BEST):
        return 0.0
    if os.path.exists(meta):
        with open(meta) as f:
            return json.load(f)["params"] * 4 / 1e6
    return GRU_MB_FP32


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
    assert_no_label_leak(X, L, split)
    return age_invariant(X), np.concatenate([L, gru[:, None]], axis=1)


def assert_no_label_leak(X, L, split):
    """The router must never see the scored word's own probabilities.

    Regression guard for the bug that invalidated the first money table:
    features [0:5] were log p_i(w_gold), i.e. bit-identical to the labels.
    The gate could then read the answer, its "mixture" summed to ~2.7 over
    the vocabulary, and every gate perplexity was inflated by that factor."""
    for j in range(X.shape[1]):
        for k in range(L.shape[1]):
            if np.array_equal(X[:, j], L[:, k]):
                sys.exit(f"{split}: feature column {j} IS label column {k} — "
                         f"the router can read the gold token. Features must "
                         f"be candidate-independent (see gate_lm.FeatMaker).")


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


def fair_router(Yva, Lva, nexp, hs=(24, 64, 128, 256), epochs=15, seed=0,
                folds=2, verbose=True):
    """The router tuned on VALIDATION ONLY: the same data budget the
    fixed-lambda weight and the static mixture get.

    Hidden size is picked by K-fold cross-validation INSIDE valid (fit on the
    other folds, score the held-out fold), then the router is refit on all of
    valid at that size. Training on train instead is a trap: the statistical
    experts are far stronger on the stream they were built from than on
    held-out text (KN3 34.7 -> 165.1 PP from train to test on PTB), so a
    train-fit router learns trust that does not transfer.

    Returns (predict, kb, h, cv_pp, curve, weights)."""
    n = len(Yva)
    fold_of = np.arange(n) % folds
    curve = {}
    for h in hs:
        tot = 0.0
        for f in range(folds):
            tr = fold_of != f
            te = ~tr
            pred, _ = train_gate(Yva[tr], Lva[tr], nexp, hidden=h,
                                 epochs=epochs, seed=seed)
            A = pred(Yva[te])
            P = np.exp(Lva[te].astype(np.float64))
            tot += float(np.sum(-np.log(np.maximum(np.sum(A * P, axis=1),
                                                   1e-300))))
        curve[h] = math.exp(tot / n)
        if verbose:
            print(f"    [valid-tuned router h={h:3d}: CV valid PP "
                  f"{curve[h]:.2f}]", flush=True)
    h_best = min(curve, key=lambda k: curve[k])
    pred, kb, wts = train_gate(Yva, Lva, nexp, hidden=h_best, epochs=epochs,
                               seed=seed, return_weights=True)
    return pred, kb, h_best, curve[h_best], curve, wts


def fixed_lambda(Lva, Lte, i=0, j=5, grid=201):
    """p = lam*p_i + (1-lam)*p_j ; lambda* on valid, honest test report.
    Default i=0 (kn3), j=5 (gru): lam is the weight on the N-GRAM expert,
    (1-lam) the weight on the GRU. Returns (lam, test_pp)."""
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
        f"[{gru_mb():.1f} MB fp32]")
    e0 = np.zeros((1, nexp)); e0[0, 0] = 1.0
    row("pure KN3 (anchor)", nll_pp(np.repeat(e0, len(L["test"]), 0),
                                    L["test"]), f"[{tri_mb():.1f} MB counts]")

    pred5, kb5 = train_gate(Y["train"], L["train"][:, :5], nexp=5)
    row("gate[5 experts]", nll_pp(pred5(Y["test"]), L["test"][:, :5]),
        f"[statistical experts + gate {kb5:.1f} KB]")

    lam, pp = fixed_lambda(L["valid"], L["test"])
    row("fixed-lambda KN3+GRU", pp,
        f"[lambda*={lam:.2f} on KN3, GRU gets {1-lam:.2f}; tuned on valid; "
        f"{tri_mb() + gru_mb():.1f} MB]")

    # The static mixture gets the SAME selection budget as every other tuned
    # row: lambda* and the gate's hidden size are picked on VALID, so the
    # valid-fit alpha is the fair baseline and the train-fit one is reported
    # only to show what fitting it on train does (it lands far from its own
    # best component because the GRU overfits train).
    pp_static_valid = None
    for tag, fit in (("train-fit", L["train"]), ("VALID-fit", L["valid"])):
        a = train_static_alpha(fit, nexp)
        pp_static = nll_pp(np.repeat(a, len(L["test"]), 0), L["test"])
        if tag == "VALID-fit":
            pp_static_valid = pp_static
        row(f"static-alpha (6 experts, {tag})", pp_static,
            "[alpha " + " ".join(f"{n} {x:.2f}"
                                 for n, x in zip(NAMES, a.ravel())) + "]")

    best = (None, 1e9, 0, 0, None)
    print("    [gate-size sweep: hidden size selected on VALID only]",
          flush=True)
    for h in (24, 64, 128, 256):
        pred, kb, wts = train_gate(Y["train"], L["train"], nexp, hidden=h,
                                   return_weights=True)
        va = nll_pp(pred(Y["valid"]), L["valid"])
        te = nll_pp(pred(Y["test"]), L["test"])
        print(f"    [gate h={h:3d}  {kb:5.1f} KB  valid {va:7.2f}  "
              f"test {te:7.2f}]", flush=True)
        if va < best[1]:
            best = (pred, va, kb, h, wts)
    pp6 = nll_pp(best[0](Y["test"]), L["test"])
    row("GATE hybrid (6 experts)", pp6,
        f"[h={best[3]}, gate {best[2]:.1f} KB, valid {best[1]:.2f}]")
    np.savez(os.path.join(OUT, f"gate_mlp_h{best[3]}.npz"),
             W1=best[4][0], b1=best[4][1], W2=best[4][2], b2=best[4][3])
    with open(os.path.join(OUT, "gate_selected.json"), "w") as f:
        json.dump({"corpus": corpus, "h": int(best[3]),
                   "kb": round(float(best[2]), 1),
                   "valid_pp": round(float(best[1]), 2),
                   "test_pp": round(float(pp6), 2),
                   "n_features": int(Y["train"].shape[1]),
                   "selected_on": "valid"}, f, indent=2)

    print("    [router tuned on VALID only — same budget as lambda* and "
          "alpha; h by 2-fold CV inside valid]", flush=True)
    predv, kbv, hv, cvv, curve_v, wts_v = fair_router(Y["valid"], L["valid"],
                                                      nexp)
    ppv = nll_pp(predv(Y["test"]), L["test"])
    row("ROUTER, valid-tuned (h by CV)", ppv,
        f"[h={hv}, gate {kbv:.1f} KB, CV valid {cvv:.2f}]")
    np.savez(os.path.join(OUT, f"gate_fair_h{hv}.npz"),
             W1=wts_v[0], b1=wts_v[1], W2=wts_v[2], b2=wts_v[3])
    with open(os.path.join(OUT, "gate_fair.json"), "w") as f:
        json.dump({"corpus": corpus, "h": int(hv), "kb": round(float(kbv), 1),
                   "cv_valid_pp": round(float(cvv), 2),
                   "test_pp": round(float(ppv), 2),
                   "cv_curve": {str(k): round(v, 2) for k, v in curve_v.items()},
                   "trained_on": "valid", "h_selected_by": "2-fold CV on valid",
                   "n_features": int(Y["valid"].shape[1])}, f, indent=2)

    pred5v, kb5v, h5v, cv5v, _c5, _w5 = fair_router(
        Y["valid"], L["valid"][:, :5], 5, verbose=False)
    row("router over 5 stat. experts, valid-tuned",
        nll_pp(pred5v(Y["test"]), L["test"][:, :5]),
        f"[h={h5v}, gate {kb5v:.1f} KB, no GRU]")

    onehot = np.eye(nexp)
    best_single = min((nll_pp(np.repeat(onehot[k:k + 1], len(L["test"]), 0),
                              L["test"]), NAMES[k]) for k in range(nexp))
    row("best single expert (hindsight)", best_single[0],
        f"[normalized bound: always pick {best_single[1]}]")
    oracle = math.exp(float(np.mean(-L["test"].max(axis=1))))
    row("per-position best expert", oracle,
        "[UNNORMALIZED diagnostic: mass > 1, not achievable by a mixture]")

    print("== verdict ==")
    print(f"  per-position best expert (UNNORMALIZED diagnostic, mass > 1): "
          f"{oracle:.2f} — no normalized mixture can reach this")
    strongest = min(pp, pp_static_valid)
    for name, val in (("train-trained router", pp6),
                      ("valid-tuned router", ppv)):
        verdict = "BEATS" if val < strongest else "LOSES to"
        print(f"  {name} {val:.2f} {verdict} the strongest tuned baseline "
              f"(fixed-lambda {pp:.2f} / static-valid {pp_static_valid:.2f}) "
              f"on {corpus}", flush=True)


if __name__ == "__main__":
    main(mock="--mock" in sys.argv)
