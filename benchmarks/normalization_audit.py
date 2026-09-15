#!/usr/bin/env python3
"""Normalization audit: is the router's mixture actually a distribution?

Any mixture LM whose weights depend on the word being scored,

    P(w|ctx) = sum_i a_i(x_{ctx,w}) * p_i(w|ctx),

is NOT a probability distribution: summing it over the vocabulary gives
Z(ctx) > 1, and the "perplexity" computed from it is deflated by roughly that
factor. Every perplexity in the paper must come from a model with Z = 1, which
is what makes the rows comparable to each other.

This script measures Z(ctx) over the full vocabulary at a random sample of
aligned test positions, for

  * the CURRENT causal router (gate_lm.FeatMaker: context-only features), and
  * the INVALIDATED v1 router (features [0:5] were log p_i(w_gold) and [12]
    was the gold word's cache count), reconstructed here so the artifact stays
    reproducible from `benchmarks/results/legacy_leaky_v1/`.

It reports the deflation factor exp(mean log Z), the sample perplexity as
reported, and the same perplexity after per-context normalization.

    python benchmarks/normalization_audit.py                 # WT-2, causal gate
    python benchmarks/normalization_audit.py --legacy        # WT-2, v1 leaky gate
    HEDGE_RESULTS=benchmarks/results/ptb python benchmarks/normalization_audit.py
    python benchmarks/normalization_audit.py --positions 400 --legacy
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import mixer_lm                                       # noqa: E402
import nn_expert                                      # noqa: E402
from gate_lm import LP_MIN, FeatMaker                 # noqa: E402
from gate_hybrid import OUT, gru_mb, selected_h, train_gate, load_split  # noqa: E402

NAMES = ["kn3", "lag2", "cache", "lag3", "uni", "gru"]


# --------------------------------------------------------------------------- #
# v1 (invalidated) feature construction, kept for the reproduction record.
# Mirrors gate_lm.collect() at commit 9d2eac4: the "votes" were the scored
# word's own log-probabilities, and feature 12 was its cache count.
# --------------------------------------------------------------------------- #

def legacy_features(mixer, ctx, i, m, cand, logp_row=None):
    """logp_row: precomputed [log p_i(cand)] for the 5 statistical experts."""
    tri = mixer.tri_model
    e1, e2, e3, e4, e5 = mixer.experts[:5]
    payload = tri._counts[2].get(tuple(ctx[-2:])) if len(ctx) == 2 else None
    if payload is None:
        mass, n3p, bkt = 0.0, 0.0, 0
    else:
        mass, n3p = payload[1], payload[3]
        bkt = 1 if mass == 1 else (2 if mass <= 5 else 3)
    c2 = ctx[-2] if len(ctx) >= 2 else None
    c3 = ctx[-3] if len(ctx) >= 3 else None
    t2 = e2.totals.get(c2, 0) if c2 is not None else 0
    t3 = e4.totals.get(c3, 0) if c3 is not None else 0
    x = np.zeros(20, dtype=np.float64)
    if logp_row is not None:
        x[:5] = logp_row
    else:
        for j, e in enumerate((e1, e2, e3, e4, e5)):
            x[j] = max(math.log(e.prob(cand, ctx)), LP_MIN)
    x[5] = math.log1p(mass)
    x[6] = math.log1p(n3p)
    x[7] = math.log1p(t2)
    x[8] = 1.0 if t2 > 0 else 0.0
    x[9] = math.log1p(t3)
    x[10] = 1.0 if t3 > 0 else 0.0
    x[11] = math.log1p(len(e3.win))
    x[12] = math.log1p(e3.counts.get(cand, 0))
    x[13 + bkt] = 1.0
    x[17] = i / max(m - 1, 1)
    x[18] = math.log1p(m)
    x[19] = 1.0 if i == 1 else 0.0
    # the age-invariant transform of gate_lm3 (v1)
    t = math.expm1(x[11])
    c = math.expm1(x[12])
    x[11] = 0.0
    x[12] = math.log1p(c / (t + 1.0))
    return x


def legacy_age_invariant(X):
    """The v1 age-invariant transform (gate_lm3.age_invariant at 9d2eac4).
    Without it the legacy router is fed features it was never trained on."""
    X2 = X.astype(np.float64).copy()
    t = np.expm1(X2[:, 11])
    c = np.expm1(X2[:, 12])
    X2[:, 11] = 0.0
    X2[:, 12] = np.log1p(c / (t + 1.0))
    return X2


def gate_weights(rows, W1, b1, W2, b2):
    h = np.maximum(rows @ W1 + b1, 0.0)
    z = h @ W2 + b2
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--legacy", action="store_true",
                    help="audit the invalidated v1 gate (leaky features)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    corpus = "PTB" if OUT.rstrip("/").endswith("ptb") else "WT-2"
    res = OUT
    if args.legacy:
        res = os.path.join(OUT, "legacy_leaky_v1")
    print(f"== normalization audit: {corpus}, "
          f"{'v1 LEAKY' if args.legacy else 'causal'} router, "
          f"{args.positions} sampled positions ==", flush=True)

    def npz(split):
        z = np.load(os.path.join(res, f"gate_{split}.npz"))
        g = np.load(os.path.join(OUT, f"nn_logp_{split}.npy"))
        X = z["X"]
        if args.legacy:
            X = legacy_age_invariant(X)
        return X, np.concatenate([z["L"], g[:, None]], axis=1)

    Xte, Lte = npz("test")
    n_test = len(Xte)
    print(f"  artifacts: {res}  X{Xte.shape} L{Lte.shape}", flush=True)

    # ---- the leak detector ------------------------------------------------ #
    Xtr, Ltr = npz("train")
    leaky = [(j, k) for j in range(Xtr.shape[1]) for k in range(Ltr.shape[1])
             if np.array_equal(Xtr[:, j], Ltr[:, k])]
    print(f"  feature columns identical to a label column: "
          f"{leaky if leaky else 'none'}", flush=True)
    del Xtr, Ltr

    # ---- router weights --------------------------------------------------- #
    if args.legacy:
        # v1 headline sizes: h=128 on WT-2, h=24 on PTB (its 3-point valid grid)
        h_sel = 24 if OUT.rstrip("/").endswith("ptb") else 128
        cached = os.path.join(res, f"gate_mlp_h{h_sel}.npz")
        if os.path.exists(cached):
            z = np.load(cached)
            W1, b1, W2, b2 = z["W1"], z["b1"], z["W2"], z["b2"]
            print(f"  v1 router: cached weights h={h_sel}", flush=True)
        else:
            # retrain the v1 router from the v1 (leaky) artifacts. Note that
            # gate_hybrid.load_split() would REFUSE to load them: its guard
            # fires on exactly this dataset. That is the point.
            from gate_hybrid import train_gate as _tg
            Xtr, Ltr = npz("train")
            _, _kb, (W1, b1, W2, b2) = _tg(Xtr, Ltr, nexp=Ltr.shape[1],
                                           hidden=h_sel, epochs=12, seed=0,
                                           return_weights=True)
            del Xtr, Ltr
            print(f"  v1 router: retrained from the legacy artifacts "
                  f"(h={h_sel}, 12 ep, seed 0)", flush=True)
    else:
        fair = os.path.join(OUT, "gate_fair.json")
        if os.path.exists(fair):
            import json as _json
            with open(fair) as f:
                h_sel = int(_json.load(f)["h"])
            cache = os.path.join(OUT, f"gate_fair_h{h_sel}.npz")
        else:
            h_sel = selected_h(128)
            cache = os.path.join(OUT, f"gate_mlp_h{h_sel}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            W1, b1, W2, b2 = z["W1"], z["b1"], z["W2"], z["b2"]
        else:
            (Ytr, Ltr2) = load_split("train")
            _, kb, (W1, b1, W2, b2) = train_gate(
                Ytr, Ltr2, nexp=Ltr2.shape[1], hidden=h_sel, epochs=12,
                seed=0, return_weights=True)
            del Ytr, Ltr2
    W1, b1, W2, b2 = (np.atleast_2d(W1), np.ravel(b1),
                      np.atleast_2d(W2), np.ravel(b2))
    print(f"  router: h={h_sel}, {W1.shape[0]} input features", flush=True)

    A = gate_weights(Xte.astype(np.float64), W1, b1, W2, b2)
    P = np.exp(Lte.astype(np.float64))
    p_gold = np.maximum(np.sum(A * P, axis=1), 1e-300)
    pp_reported = math.exp(float(np.mean(-np.log(p_gold))))
    print(f"  reported test PP over all {n_test:,} positions: "
          f"{pp_reported:.2f}", flush=True)

    # ---- vocabulary mass at sampled positions ----------------------------- #
    streams, mixer = mixer_lm.build(verbose=False)
    e1, e2, e3, e4, e5 = mixer.experts[:5]
    w2i, unk, i2w = mixer.w2i, mixer.unk, mixer.i2w
    V = len(i2w)
    allw = np.arange(V)
    fm = FeatMaker(mixer)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    nn_expert.EOS = V
    gru = nn_expert.GRULM(V + 1)
    gru.load_state_dict(torch.load(os.path.join(OUT, "nn_gru_best.pt")))
    gru.eval()

    base_uni = np.array([e5.p.get(int(k), 0.0) for k in allw])
    rng = np.random.default_rng(args.seed)
    sample = set(rng.choice(n_test, size=min(args.positions, n_test),
                            replace=False).tolist())

    def stat_probs(ctx):
        """(V,5) matrix of every expert's probability of every vocabulary word."""
        M = np.empty((V, 5))
        M[:, 0] = [e1.prob(int(k), ctx) for k in allw]
        for col, e, lag in ((1, e2, 2), (3, e4, 3)):
            c = ctx[-lag] if len(ctx) >= lag else None
            t = e.totals.get(c, 0) if c is not None else 0
            cnt = np.array([e.counts.get((c, int(k)), 0)
                            if c is not None else 0 for k in allw])
            M[:, col] = np.maximum((cnt + mixer_lm.BIGRAM_MU * base_uni)
                                   / (t + mixer_lm.BIGRAM_MU),
                                   mixer_lm.EXPERT_FLOOR)
        t = len(e3.win)
        cnt = np.array([e3.counts.get(int(k), 0) for k in allw])
        M[:, 2] = np.maximum((cnt + mixer_lm.CACHE_MU * base_uni)
                             / (t + mixer_lm.CACHE_MU), mixer_lm.EXPERT_FLOOR)
        M[:, 4] = np.maximum(base_uni, mixer_lm.EXPERT_FLOOR)
        return M

    rows, t0 = [], time.time()
    r = 0
    for sent in streams["test"]:
        ids = [w2i.get(w, unk) for w in sent]
        m = len(ids)
        for i in range(1, m):
            if r in sample:
                ctx = tuple(ids[max(0, i - 3):i])
                M = stat_probs(ctx)
                logp = np.maximum(np.log(M), LP_MIN)
                if args.legacy:
                    cc = np.array([e3.counts.get(int(k), 0) for k in allw])
                    rows_in = np.stack([
                        legacy_features(mixer, ctx, i, m, int(k), logp[k])
                        for k in allw])
                    rows_in[:, 12] = np.log1p(cc)   # v1 feature 12: gold/cand count
                else:
                    x = fm.features(list(ctx), i)
                    rows_in = np.tile(x.astype(np.float64), (V, 1))
                Aw = gate_weights(rows_in, W1, b1, W2, b2)
                with torch.no_grad():
                    lg, _ = gru(torch.tensor(ids[:i],
                                             dtype=torch.int64).view(1, -1), None)
                # renormalize over the V scored words: the GRU's softmax spans
                # V+1 (it includes eos), and eos is not part of the vocabulary
                # we enumerate here. Without this the control rows would read
                # Z = 0.96 instead of 1.000 and the audit would be measuring
                # its own bookkeeping.
                lsm = torch.log_softmax(lg[0, -1], -1).numpy()[:V]
                glp = lsm - np.log(np.exp(lsm).sum())
                Pv = np.exp(np.concatenate([logp, glp[:, None]], axis=1))
                Z = float(np.sum(Aw * Pv))
                rows.append((r, -math.log(float(p_gold[r])), math.log(max(Z, 1e-300))))
                if len(rows) % 25 == 0:
                    print(f"    {len(rows)}/{len(sample)} positions "
                          f"({time.time()-t0:.0f}s)", flush=True)
            e3.update(ids[i])       # same causal advance as the collector
            r += 1

    nll = np.array([x[1] for x in rows])
    logZ = np.array([x[2] for x in rows])
    pp_sample = math.exp(float(nll.mean()))
    # normalized: P_norm(w) = P_reported(w) / Z  ->  -log P_norm = nll + log Z
    pp_norm = math.exp(float((nll + logZ).mean()))
    lines = [
        f"normalization audit — {corpus}, "
        f"{'v1 LEAKY (candidate-conditioned)' if args.legacy else 'causal (context-only)'} router",
        f"sampled positions                : {len(rows)}",
        f"reported test PP (all positions) : {pp_reported:.2f}",
        f"reported PP on the sample        : {pp_sample:.2f}",
        f"mean log Z(ctx) over vocabulary  : {logZ.mean():+.4f}"
        f"   (mass x{math.exp(logZ.mean()):.3f})",
        f"NORMALIZED PP on the sample      : {pp_norm:.2f}",
        f"implied deflation of the reported number: "
        f"x{math.exp(logZ.mean()):.2f} (normalized = reported x Z)",
        "a mixture LM must have Z = 1 at every context; Z > 1 means the",
        "router's weights depend on the word being scored and the perplexity",
        "is not comparable to a normalized baseline.",
    ]
    print()
    for ln in lines:
        print("  " + ln)
    out = args.out or os.path.join(
        res, "normalization_audit_legacy.log" if args.legacy
        else "normalization_audit.log")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
