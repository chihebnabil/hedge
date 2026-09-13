"""
Energy/latency harness (Phase 6): per-token cost of each system, stated method.

Method:
- Workload: first 200 test sentences, scored token-by-token in a causal
  stream (deployment-realistic: cache updates + GRU state carried across
  sentences within each repeat; state reset between repeats).
- Timing: wall (perf_counter) and CPU (process_time); 3 repeats, BEST
  repeat reported (standard practice for latency).
- Energy: ESTIMATE ONLY — CPU seconds/token x 15 W (nominal mobile-class
  TDP). Stated as a proxy, not measured power.
- Models (all read the same token stream):
    kn3        exact MKN trigram prob per token (_p_kn via expert call)
    mixer5     full 5-expert statistical stack, mixed (mixer.run)
    gru        2x256 GRU, token-by-token batch-1 (interactive worst case)
    fixed-lam  KN3+GRU fixed interpolation, lam* from the money table
    gate6      full hybrid: 5 expert votes + evidence features + age-
               invariant transform + gate MLP + GRU vote + mixture
- Sizes: tri pickle file, lag/cache/unigram expert pickles, gate KB, GRU fp32.

    python benchmarks/energy_harness.py            # WT-2
    HEDGE_RESULTS=benchmarks/results/ptb python benchmarks/energy_harness.py
"""

import math
import os
import pickle
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from gate_lm import LP_MIN, NF                      # noqa: E402
from gate_lm3 import age_invariant                  # noqa: E402
from gate_hybrid import load_split, train_gate, OUT  # noqa: E402
import mixer_lm                                     # noqa: E402
import nn_expert                                    # noqa: E402

TDP_W = 15.0          # nominal mobile-class TDP for the joule proxy
N_SENTS = 200
REPEATS = 3
LAM = {"WT-2": 0.21, "PTB": 0.28}


def timed(fn, repeats=REPEATS):
    best = None
    for _ in range(repeats):
        t0w, t0c = time.perf_counter(), time.process_time()
        n = fn()
        w, c = time.perf_counter() - t0w, time.process_time() - t0c
        if best is None or w < best[0]:
            best = (w, c, n)
    return best


class GruScorer:
    def __init__(self, V):
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        torch.manual_seed(42)
        nn_expert.EOS = V - 1
        self.V = V
        self.m = nn_expert.GRULM(V)
        self.m.load_state_dict(torch.load(os.path.join(OUT, "nn_gru_best.pt")))
        self.m.eval()
        self.h = None

    def reset(self):
        self.h = None

    def step(self, wid):
        with torch.no_grad():
            logits, self.h = self.m(
                torch.tensor([wid], dtype=torch.int64).view(1, 1), self.h)
            return torch.log_softmax(logits[0, -1], -1)


def main():
    corpus = "PTB" if OUT.rstrip("/").endswith("ptb") else "WT-2"
    streams, mixer = mixer_lm.build()
    e1, e2, e3, e4, e5 = mixer.experts
    tri = mixer.tri_model
    w2i, unk = mixer.w2i, mixer.unk
    sents = streams["test"][:N_SENTS]
    lam = LAM[corpus]

    # gate weights: WT-2 uses the cached headline gate; PTB trains h=24 fresh
    if corpus == "WT-2":
        z = np.load(os.path.join(OUT, "gate_mlp_h128.npz"))
        W1, b1, W2, b2 = z["W1"], z["b1"], z["W2"], z["b2"]
        gate_kb = W1.size * 4 / 1024 + W2.size * 4 / 1024
    else:
        (Ytr, Ltr) = load_split("train")
        _, gate_kb, (W1, b1, W2, b2) = train_gate(
            Ytr, Ltr, nexp=Ltr.shape[1], hidden=24, epochs=12, seed=0,
            return_weights=True)
        del Ytr, Ltr
    W1 = np.atleast_2d(W1)
    b1, W2, b2 = np.ravel(b1), np.atleast_2d(W2), np.ravel(b2)

    gs = GruScorer(len(mixer.i2w) + 1)
    gru_mb = sum(p.numel() for p in gs.m.parameters()) * 4 / 1e6
    pcont = e5.p

    def kn3():
        n = 0
        gs.reset()
        for sent in sents:
            ids = [w2i.get(w, unk) for w in sent]
            for i in range(1, len(ids)):
                e1.prob(ids[i], tuple(ids[max(0, i - 2):i]))
                n += 1
        return n

    def mixer5():
        mixer.experts[2] = e3.__class__(e5)
        r = mixer.run(sents)
        return r.get("n", r.get("tokens", 0)) or 1

    def gru():
        n = 0
        gs.reset()
        for sent in sents:
            ids = [w2i.get(w, unk) for w in sent]
            for i in range(1, len(ids)):
                gs.step(ids[i - 1])
                n += 1
        return n

    def fixed_lam():
        n = 0
        gs.reset()
        for sent in sents:
            ids = [w2i.get(w, unk) for w in sent]
            for i in range(1, len(ids)):
                p = e1.prob(ids[i], tuple(ids[max(0, i - 2):i]))
                lp = gs.step(ids[i - 1])
                math.log(lam * p + (1 - lam) * math.exp(float(lp[ids[i]]))
                         + 1e-300)
                n += 1
        return n

    def gate6():
        n = 0
        gs.reset()
        e3c = e3.__class__(e5)
        big2_t, big2_c = e2.totals, e2.counts
        big3_t, big3_c = e4.totals, e4.counts
        for sent in sents:
            ids = [w2i.get(w, unk) for w in sent]
            m = len(ids)
            for i in range(1, m):
                wid = ids[i]
                ctx = tuple(ids[max(0, i - 3):i])
                p_kn = e1.prob(wid, ctx)
                p_l2 = e2.prob(wid, ctx)
                p_ca = e3c.prob(wid, ctx)
                p_l3 = e4.prob(wid, ctx)
                p_un = e5.prob(wid, ctx)
                payload = tri._counts[2].get(ctx[-2:]) if len(ctx) == 2 else None
                if payload is None:
                    mass, n3p, bkt = 0.0, 0.0, 0
                else:
                    mass, n3p = payload[1], payload[3]
                    bkt = 1 if mass == 1 else (2 if mass <= 5 else 3)
                c2 = ctx[-2] if len(ctx) >= 2 else None
                c3 = ctx[-3] if len(ctx) >= 3 else None
                t2 = big2_t.get(c2, 0) if c2 is not None else 0
                t3 = big3_t.get(c3, 0) if c3 is not None else 0
                x = np.empty(NF, dtype=np.float32)
                x[0] = max(math.log(p_kn), LP_MIN)
                x[1] = max(math.log(p_l2), LP_MIN)
                x[2] = max(math.log(p_ca), LP_MIN)
                x[3] = max(math.log(p_l3), LP_MIN)
                x[4] = max(math.log(p_un), LP_MIN)
                x[5] = math.log1p(mass)
                x[6] = math.log1p(n3p)
                x[7] = math.log1p(t2)
                x[8] = 1.0 if t2 > 0 else 0.0
                x[9] = math.log1p(t3)
                x[10] = 1.0 if t3 > 0 else 0.0
                x[11] = math.log1p(len(e3c.win))
                x[12] = math.log1p(e3c.counts.get(wid, 0))
                x[13:17] = 0.0
                x[13 + bkt] = 1.0
                x[17] = i / max(m - 1, 1)
                x[18] = math.log1p(m)
                x[19] = 1.0 if i == 1 else 0.0
                xa = age_invariant(x.reshape(1, -1))
                hid = np.maximum(xa @ W1 + b1, 0.0)
                zz = hid @ W2 + b2
                a = np.exp(zz - zz.max())
                a /= a.sum()
                lp = gs.step(ids[i - 1])
                votes = np.array([x[0], x[1], x[2], x[3], x[4],
                                  float(lp[wid])], dtype=np.float64)
                math.log(float(a.ravel() @ np.exp(votes)) + 1e-300)
                e3c.update(wid)
                n += 1
        return n

    print(f"== energy harness: {corpus}, {len(sents)} test sentences, "
          f"lam*={lam}, TDP proxy {TDP_W:.0f} W ==", flush=True)
    rows = []
    for name, fn in (("kn3", kn3), ("mixer5", mixer5), ("gru", gru),
                     ("fixed-lam", fixed_lam), ("gate6 (full)", gate6)):
        w, c, n = timed(fn)
        msw = w / n * 1e3
        j = w / n * TDP_W * 1e3   # wall x TDP proxy, mJ/token
        rows.append((name, msw, c / n * 1e3, j, n))
        print(f"  {name:<12} wall {msw:7.3f} ms/tok   cpu {c/n*1e3:7.3f}"
              f" ms/tok   ~{j:6.3f} mJ/tok   (n={n:,})", flush=True)

    def pkl_mb(obj):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            pickle.dump(obj, f)
            s = os.path.getsize(f.name)
        os.unlink(f.name)
        return s / 1e6

    print(f"\n  sizes: tri pickle {os.path.getsize(os.path.join(OUT, 'beast_trigram.pkl'))/1e6:.1f} MB"
          f" | lag2 {pkl_mb({'t': e2.totals, 'c': e2.counts}):.1f} MB"
          f" | lag3 {pkl_mb({'t': e4.totals, 'c': e4.counts}):.1f} MB"
          f" | cache+uni {pkl_mb({'c': dict(e3.counts), 'p': pcont}):.1f} MB"
          f" | gate {gate_kb:.1f} KB | GRU {gru_mb:.1f} MB fp32", flush=True)


if __name__ == "__main__":
    main()
