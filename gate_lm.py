"""
gate_lm.py — Stage 1 of the neural ladder: replace the mixer's 4-bucket
softmax with a tiny neural gate (MLP), trained offline on per-position data.

Setup (all experts FROZEN, only the gate learns):
  features x (24 dims, ALL CAUSAL AND CANDIDATE-INDEPENDENT — they are a
  function of the context alone, never of the word being scored):
    [0:5]   log p_i(top1_i(ctx))     each expert's confidence in its OWN
                                     favourite word (the "votes")
    [5:10]  1.0 iff top1_i(ctx) == the plurality favourite of the five
    [10]    log1p(mass of bigram context in KN3)      (context evidence)
    [11]    log1p(distinct KN3 continuations of ctx)
    [12,13] log1p(lag2 ctx total), hit flag
    [14,15] log1p(lag3 ctx total), hit flag
    [16]    log1p(distinct cache types / (window+1))   (age-invariant)
    [17]    log1p(top cache count / (window+1))        (age-invariant)
    [18:22] KN-mass bucket one-hot (the old model's whole input)
    [22]    log1p(tokens seen so far in this sentence)
    [23]    sentence-start flag
  Dropped as non-causal: the sentence-length and position-fraction features
  of the first version (both need the sentence END, which is future
  information at scoring time).
  labels L (5 dims): log p_i(w_gold) — the target-side data the mixture is
  scored with. Labels appear ONLY in L, never in x.
  gate: 24 -> H -> 5 logits -> softmax alpha ; mixture P = sum_i a_i p_i
  loss = -log P   (same objective as the PP benchmark)

  INVARIANT (tests/test_gate.py): alpha is a function of the context alone,
  so P(w|ctx) is a normalized distribution over the vocabulary and the
  reported perplexity is comparable to the baselines'. An earlier version of
  this file fed log p_i(w_gold) in as features [0:5] — the gate could then
  read the answer, the "mixture" summed to ~2.7 over the vocabulary, and
  every gate perplexity was inflated by that factor. Do not reintroduce any
  feature that depends on the scored word.

Baselines reported on the SAME 217,004-position test stream:
  global static alpha | bucket softmax (static, learned on train)
  | ONLINE bucket mixer (= the 199.5 protocol) | MLP gate | oracle.

Usage:  python gate_lm.py            (caches datasets as .npz for re-runs)
        python gate_lm.py --collect  (rebuild the .npz caches and exit)
"""

import math
import os
import sys
import time

import numpy as np

from mixer_lm import FLOOR, build

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("HEDGE_RESULTS",
                           os.path.join(HERE, "benchmarks", "results"))
NF = 24
NEXP = 5
LP_MIN = math.log(FLOOR)
NAMES = ["kn3", "lag2", "cache", "lag3", "uni"]
VOTE_COLS = list(range(0, 10))
EVID_COLS = list(range(10, NF))


# -- causal feature extraction ------------------------------------------------ #

class FeatMaker:
    """Gate features + expert labels for one position.

    `features()` is the ONLY input the router ever sees; it must not depend on
    the word being scored. `labels()` holds the target-side log-probs used to
    score the mixture. Keeping the two apart is what makes the perplexity a
    perplexity (see the module docstring and tests/test_gate.py)."""

    def __init__(self, mixer):
        self.mixer = mixer
        self.tri = mixer.tri_model
        self.e1, self.e2, self.e3, self.e4, self.e5 = mixer.experts[:5]
        self.w2i, self.i2w, self.unk = mixer.w2i, mixer.i2w, mixer.unk
        self.big2_t = self.e2.totals
        self.big3_t = self.e4.totals

    def top1s(self, ctx_ids):
        """[(wid, p)] per expert — each expert's own favourite, context-only."""
        return (self.e1.top1(ctx_ids), self.e2.top1(ctx_ids),
                self.e3.top1(ctx_ids), self.e4.top1(ctx_ids),
                self.e5.top1(ctx_ids))

    def features(self, ctx_ids, i):
        """24-dim causal feature vector for position i (0-based count of
        tokens already seen in the sentence)."""
        tops = self.top1s(ctx_ids)
        fav = [w for w, _p in tops]
        plur = max(set(fav), key=fav.count)

        payload = (self.tri._counts[2].get(tuple(ctx_ids[-2:]))
                   if len(ctx_ids) >= 2 else None)
        if payload is None:
            mass, n3p, bkt = 0.0, 0.0, 0
        else:
            mass, n3p = payload[1], payload[3]
            bkt = 1 if mass == 1 else (2 if mass <= 5 else 3)
        c2 = ctx_ids[-2] if len(ctx_ids) >= 2 else None
        c3 = ctx_ids[-3] if len(ctx_ids) >= 3 else None
        t2 = self.big2_t.get(c2, 0) if c2 is not None else 0
        t3 = self.big3_t.get(c3, 0) if c3 is not None else 0
        win = len(self.e3.win) + 1.0

        x = np.zeros(NF, dtype=np.float32)
        for j, (_w, p) in enumerate(tops):
            x[j] = max(math.log(p), LP_MIN)
            x[NEXP + j] = 1.0 if fav[j] == plur else 0.0
        x[10] = math.log1p(mass)
        x[11] = math.log1p(n3p)
        x[12] = math.log1p(t2)
        x[13] = 1.0 if t2 > 0 else 0.0
        x[14] = math.log1p(t3)
        x[15] = 1.0 if t3 > 0 else 0.0
        x[16] = math.log1p(len(self.e3.counts) / win)
        x[17] = math.log1p((max(self.e3.counts.values()) if self.e3.counts
                            else 0) / win)
        x[18:22] = 0.0
        x[18 + bkt] = 1.0
        x[22] = math.log1p(i)
        x[23] = 1.0 if i == 1 else 0.0
        return x

    def labels(self, ctx_ids, wid):
        """5-dim log p_i(w_gold): the target-side scores, NEVER a feature."""
        ctx = tuple(ctx_ids)
        return np.array([
            max(math.log(self.e1.prob(wid, ctx)), LP_MIN),
            max(math.log(self.e2.prob(wid, ctx)), LP_MIN),
            max(math.log(self.e3.prob(wid, ctx)), LP_MIN),
            max(math.log(self.e4.prob(wid, ctx)), LP_MIN),
            max(math.log(self.e5.prob(wid, ctx)), LP_MIN),
        ], dtype=np.float32)


# -- data collection --------------------------------------------------------- #

def collect(mixer, sents, desc):
    """One causal stream over sents; returns X (N,24) f32, L (N,5) f32.
    Cache state evolves during the pass (deployment-realistic)."""
    fm = FeatMaker(mixer)
    w2i, unk = fm.w2i, fm.unk
    X = np.empty((256_000, NF), dtype=np.float32)
    L = np.empty((256_000, NEXP), dtype=np.float32)
    n = 0
    t0 = time.time()
    for si, sent in enumerate(sents):
        ids = [w2i.get(w, unk) for w in sent]
        m = len(ids)
        for i in range(1, m):
            wid = ids[i]
            ctx = ids[max(0, i - 3):i]
            X[n] = fm.features(ctx, i)
            L[n] = fm.labels(ctx, wid)
            n += 1
            if n == len(X):
                X = np.resize(X, (len(X) * 2, NF))
                L = np.resize(L, (len(L) * 2, NEXP))
            mixer.experts[2].update(wid)     # causal state advance
        if (si + 1) % 2000 == 0:
            print(f"  [{desc}] sent {si+1}/{len(sents)}  pos {n:,}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"  [{desc}] done: {n:,} positions in {time.time()-t0:.0f}s",
          flush=True)
    return X[:n].copy(), L[:n].copy()


def get_data(mixer, streams, force=False):
    paths = {k: os.path.join(CACHE_DIR, f"gate_{k}.npz")
             for k in ("train", "valid", "test")}
    if not force and all(os.path.exists(p) for p in paths.values()):
        out = {}
        for k, p in paths.items():
            d = np.load(p)
            out[k] = (d["X"], d["L"])
        return out
    out = {}
    out["train"] = collect(mixer, streams["train"], "train")
    mixer.experts[2] = mixer.experts[2].__class__(mixer.experts[4])  # reset
    out["valid"] = collect(mixer, streams["valid"], "valid")
    mixer.experts[2] = mixer.experts[2].__class__(mixer.experts[4])  # reset
    out["test"] = collect(mixer, streams["test"], "test")
    for k, p in paths.items():
        np.savez_compressed(p, X=out[k][0], L=out[k][1])
    return out


# -- models ------------------------------------------------------------------ #

def nll_of(A, L):
    """-log sum_i A_i exp(L_i), averaged. A (N,5), L (N,5)."""
    P = np.exp(L.astype(np.float64))
    return float(np.mean(-np.log(np.maximum(np.sum(A * P, axis=1), 1e-300))))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_gate(X, L, hidden=64, epochs=4, bs=4096, lr=3e-3, seed=0):
    """MLP gate trained to minimize mixture NLL. Returns predict fn + info."""
    rng = np.random.default_rng(seed)
    N, F = X.shape
    W1 = rng.normal(0, (2 / F) ** 0.5, (F, hidden)).astype(np.float64)
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, (2 / hidden) ** 0.5, (hidden, NEXP)).astype(np.float64)
    b2 = np.zeros(NEXP)
    params = [W1, b1, W2, b2]
    mts = [np.zeros_like(p) for p in params]
    vts = [np.zeros_like(p) for p in params]
    P = np.exp(L.astype(np.float64))                  # (N,5) expert probs
    nll_hist = []
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
            loss = float(np.mean(-np.log(pm)))
            tot += loss * len(j)
            g = a * (1.0 - Pb / pm)                   # dNLL/dz
            gW2 = h.T @ g / len(j)
            gb2 = g.mean(axis=0)
            gh = (g @ W2.T) * (h > 0)
            gW1 = xb.T @ gh / len(j)
            gb1 = gh.mean(axis=0)
            step += 1
            for p, gdp, mt, vt in zip(params, (gW1, gb1, gW2, gb2), mts, vts):
                mt *= 0.9
                mt += 0.1 * gdp
                vt *= 0.999
                vt += 0.001 * gdp * gdp
                p -= lr * (mt / (1 - 0.9 ** step)) / \
                     (np.sqrt(vt / (1 - 0.999 ** step)) + 1e-8)
        nll_hist.append(tot / N)
        print(f"  epoch {ep+1}: train NLL {nll_hist[-1]:.4f}")

    def predict(Xb):
        h = np.maximum(Xb @ W1 + b1, 0.0)
        return softmax(h @ W2 + b2)

    kbytes = (W1.size + b1.size + W2.size + b2.size) * 4 / 1024
    return predict, {"nll": nll_hist, "params_kb": kbytes}


def train_linear(X, L, cols, epochs=6, bs=8192, lr=3e-2, seed=0):
    """Static softmax gate on selected feature cols (global or bucket)."""
    rng = np.random.default_rng(seed)
    F = len(cols)
    W = np.zeros((F, NEXP))
    b = np.zeros(NEXP)
    params = [W, b]
    mts = [np.zeros_like(p) for p in params]
    vts = [np.zeros_like(p) for p in params]
    P = np.exp(L.astype(np.float64))
    N = len(X)
    step = 0
    for _ in range(epochs):
        idx = rng.permutation(N)
        for s in range(0, N, bs):
            j = idx[s:s + bs]
            xb = X[np.ix_(j, cols)].astype(np.float64)
            Pb = P[j]
            a = softmax(xb @ W + b)
            pm = np.maximum(np.sum(a * Pb, axis=1, keepdims=True), 1e-300)
            g = a * (1.0 - Pb / pm)
            step += 1
            grads = [xb.T @ g / len(j), g.mean(axis=0)]
            for p, gdp, mt, vt in zip(params, grads, mts, vts):
                mt *= 0.9
                mt += 0.1 * gdp
                vt *= 0.999
                vt += 0.001 * gdp * gdp
                p -= lr * (mt / (1 - 0.9 ** step)) / \
                     (np.sqrt(vt / (1 - 0.999 ** step)) + 1e-8)

    def predict(Xb):
        return softmax(Xb[:, cols].astype(np.float64) @ W + b)

    return predict


# -- evaluation -------------------------------------------------------------- #

def report(name, A, L, n=1):
    pp = math.exp(nll_of(A, L))
    print(f"  {name:<38} PP={pp:8.2f}")
    return pp


def main():
    streams, mixer = build()
    if "--collect" in sys.argv:
        get_data(mixer, streams, force=True)
        for k in ("train", "valid", "test"):
            d = np.load(os.path.join(CACHE_DIR, f"gate_{k}.npz"))
            print(f"  gate_{k}.npz: X{d['X'].shape} L{d['L'].shape}")
        return
    data = get_data(mixer, streams)
    for split in ("train", "valid", "test"):
        X, L = data[split]
        oracle = math.exp(float(np.mean(-L.max(axis=1))))
        print(f"[{split}] {len(X):,} positions | oracle PP={oracle:.2f}")

    Xtr, Ltr = data["train"]
    Xva, Lva = data["valid"]
    Xte, Lte = data["test"]

    print("== frozen-expert baselines (same test stream) ==")
    uni = np.full((len(Xte), NEXP), 1.0 / NEXP)
    report("equal mix (static)", uni, Lte)
    A = np.tile(train_linear(Xtr, Ltr, cols=[18])(Xtr[:1]).ravel(), (len(Xte), 1))
    report("global static alpha (learned)", A, Lte)
    Abkt = train_linear(Xtr, Ltr, cols=[18, 19, 20, 21])(Xte)
    report("bucket softmax, static (learned)", Abkt, Lte)

    print("== neural gate ==")
    best = (None, 1e9)
    for hidden in (24, 64, 128):
        print(f" training MLP gate (hidden={hidden}) on "
              f"{len(Xtr):,} train positions ...")
        pred, info = train_gate(Xtr, Ltr, hidden=hidden)
        va = nll_of(pred(Xva), Lva)
        print(f"  hidden={hidden}: valid NLL={va:.4f} "
              f"(PP={math.exp(va):.2f})  gate={info['params_kb']:.1f} KB")
        if va < best[1]:
            best = (pred, va)
    Amlp = best[0](Xte)
    pp_mlp = report("MLP gate (frozen, test)", Amlp, Lte)

    # diagnostics: where the gate wins (diff<0 => gate assigns more mass)
    A_b = Abkt
    pm_m = np.sum(Amlp * np.exp(Lte.astype(np.float64)), axis=1)
    pm_b = np.sum(A_b * np.exp(Lte.astype(np.float64)), axis=1)
    diff = np.log(np.maximum(pm_m, 1e-300)) - np.log(np.maximum(pm_b, 1e-300))
    print(f"  vs static bucket: gate wins on {float((diff>0).mean()):.1%} of "
          f"positions, mean log-gain {float(diff.mean()):+.4f}")
    for bkt in range(4):
        m = Xte[:, 18 + bkt] > 0.5
        if m.sum():
            d = diff[m]
            print(f"    bucket {bkt}: n={int(m.sum()):>7,}  "
                  f"mean log-gain {float(d.mean()):+.4f}")
    pick = Amlp.argmax(axis=1)
    print("  gate expert picks: " +
          ", ".join(f"{NAMES[i]} {float((pick==i).mean()):.0%}"
                    for i in range(NEXP)))
    print(f"== MLP gate (5 experts, train-trained): {pp_mlp:.2f} | anchors: "
          f"see PAPER_NOTES.md §0 (the v1 anchor numbers that used to be "
          f"printed here are invalidated — §12) ==")


if __name__ == "__main__":
    main()
