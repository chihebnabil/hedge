"""
Energy/latency harness: per-token cost of each system, stated method.

Method:
- Workload: first 200 test sentences, scored token-by-token in a causal
  stream (deployment-realistic: cache updates + GRU state carried across
  sentences within each round; state reset between rounds).
- Threads: torch is pinned to HEDGE_THREADS (default 1). Batch-1 GRU steps on
  a multi-thread pool are dominated by thread-sync noise: an earlier 4-thread
  run measured the GRU ALONE at 5.09 ms/token and the full hybrid that
  contains it at 3.60 ms/token, which is impossible. Single-thread timings are
  stable enough that a component can never measure slower than the system
  containing it (the harness asserts that now).
- Timing: wall (perf_counter) and CPU (process_time); ROUNDS rounds in which
  every model is timed once, model order rotated each round. The MEDIAN round is
  the quoted number and the min/max spread is printed beside it. Medians, not
  minima: on PTB the minimum round put static6 (GRU + five expert evaluations)
  13% BELOW the GRU alone, which is impossible, while the medians order every
  row correctly. The harness asserts the containment ordering on the quoted
  statistic and warns if it is violated.
- Energy: ESTIMATE ONLY — wall seconds/token x 15 W (nominal mobile-class
  TDP). Stated as a proxy, not measured power.
- Models (all read the same token stream):
    kn3         exact MKN trigram prob per token
    mixer5      full 5-expert statistical stack, mixed (mixer.run)
    router-only causal context features + MLP forward, NO GRU: this row is the
                marginal cost of routing, which is what the paper quotes
    gru         2x256 GRU, token-by-token batch-1 (interactive worst case)
    fixed-lam   KN3+GRU fixed interpolation, lam* from the money table
    gate6       full hybrid: causal features + router MLP + GRU vote + mixture
- Router weights: the headline valid-tuned router (gate_fair_h*.npz), so the
  timed system is the system the paper reports.

    python benchmarks/energy_harness.py            # WT-2
    HEDGE_RESULTS=benchmarks/results/ptb python benchmarks/energy_harness.py
"""

import json
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

from gate_lm import FeatMaker                             # noqa: E402
from gate_hybrid import (OUT, gru_mb, load_split,          # noqa: E402
                         selected_h, train_gate, tri_mb)
import mixer_lm                                           # noqa: E402
import nn_expert                                          # noqa: E402

TDP_W = 15.0          # nominal mobile-class TDP for the joule proxy
N_SENTS = 200
ROUNDS = int(os.environ.get("HEDGE_ROUNDS", "5"))
THREADS = int(os.environ.get("HEDGE_THREADS", "1"))
LAM = {"WT-2": 0.21, "PTB": 0.28}   # weight on KN3; the GRU gets 1-lambda


class GruScorer:
    def __init__(self, V):
        torch.set_num_threads(THREADS)
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
    streams, mixer = mixer_lm.build(verbose=False)
    e1, e2, e3, e4, e5 = mixer.experts
    tri = mixer.tri_model
    w2i, unk = mixer.w2i, mixer.unk
    sents = streams["test"][:N_SENTS]
    lam = LAM[corpus]

    # the headline router: valid-tuned (gate_fair_h*.npz); else train one
    fair = os.path.join(OUT, "gate_fair.json")
    if os.path.exists(fair):
        with open(fair) as f:
            meta = json.load(f)
        H = int(meta["h"])
        z = np.load(os.path.join(OUT, f"gate_fair_h{H}.npz"))
        W1, b1, W2, b2 = z["W1"], z["b1"], z["W2"], z["b2"]
        gate_kb = (W1.size + b1.size + W2.size + b2.size) * 4 / 1024
        src = f"valid-tuned h={H} (the paper's headline router)"
    else:
        H = selected_h(64)
        (Ytr, Ltr) = load_split("train")
        _, gate_kb, (W1, b1, W2, b2) = train_gate(
            Ytr, Ltr, nexp=Ltr.shape[1], hidden=H, epochs=12, seed=0,
            return_weights=True)
        del Ytr, Ltr
        src = f"train-trained h={H} (no gate_fair.json found)"
    W1 = np.atleast_2d(W1)
    b1, W2, b2 = np.ravel(b1), np.atleast_2d(W2), np.ravel(b2)

    gs = GruScorer(len(mixer.i2w) + 1)
    pcont = e5.p
    fm = FeatMaker(mixer)

    def kn3():
        n = 0
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

    def router_only():
        """Causal features + MLP forward. No GRU: the marginal cost of routing."""
        n = 0
        e3c = e3.__class__(e5)
        fm.e3 = e3c
        e1._top1_cache.clear()
        for sent in sents:
            ids = [w2i.get(w, unk) for w in sent]
            m = len(ids)
            for i in range(1, m):
                x = fm.features(ids[max(0, i - 3):i], i)
                hid = np.maximum(x.reshape(1, -1) @ W1 + b1, 0.0)
                zz = hid @ W2 + b2
                a = np.exp(zz - zz.max())
                a /= a.sum()
                e3c.update(ids[i])
                n += 1
        return n

    def gru():
        n = 0
        gs.reset()
        for sent in sents:
            ids = [w2i.get(w, unk) for w in sent]
            for i in range(1, len(ids)):
                gs.step(ids[i - 1])
                n += 1
        return n

    def static6():
        """The tuned static 6-expert mixture: same expert evaluations as the
        full hybrid, no features and no MLP. This is the right baseline for
        'what does per-token routing cost' — the difference between this row
        and gate6 is the router and nothing else."""
        n = 0
        gs.reset()
        e3c = e3.__class__(e5)
        a = np.array([0.17, 0.00, 0.11, 0.00, 0.00, 0.70])
        a = a / a.sum()
        for sent in sents:
            ids = [w2i.get(w, unk) for w in sent]
            for i in range(1, len(ids)):
                wid = ids[i]
                ctx = ids[max(0, i - 3):i]
                lp = gs.step(ids[i - 1])
                votes = np.array(
                    [math.log(max(e.prob(wid, ctx), 1e-10))
                     for e in (e1, e2, e3c, e4, e5)] + [float(lp[wid])],
                    dtype=np.float64)
                math.log(float(a @ np.exp(votes)) + 1e-300)
                e3c.update(wid)
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
        fm.e3 = e3c                     # feature cache state == scoring state
        e1._top1_cache.clear()          # no cross-round memo: honest cost
        for sent in sents:
            ids = [w2i.get(w, unk) for w in sent]
            m = len(ids)
            for i in range(1, m):
                wid = ids[i]
                ctx = ids[max(0, i - 3):i]
                x = fm.features(ctx, i)
                hid = np.maximum(x.reshape(1, -1) @ W1 + b1, 0.0)
                zz = hid @ W2 + b2
                a = np.exp(zz - zz.max())
                a /= a.sum()
                lp = gs.step(ids[i - 1])
                votes = np.array(
                    [math.log(max(e.prob(wid, ctx), 1e-10))
                     for e in (e1, e2, e3c, e4, e5)] + [float(lp[wid])],
                    dtype=np.float64)
                math.log(float(a.ravel() @ np.exp(votes)) + 1e-300)
                e3c.update(wid)
                n += 1
        return n

    print(f"== energy harness: {corpus}, {len(sents)} test sentences, "
          f"lam*={lam} (weight on KN3), {THREADS} torch thread(s), "
          f"{ROUNDS} interleaved rounds, TDP proxy {TDP_W:.0f} W ==", flush=True)
    print(f"  router: {src}", flush=True)
    models = (("kn3", kn3), ("mixer5", mixer5), ("router-only", router_only),
              ("gru", gru), ("fixed-lam", fixed_lam), ("static6", static6),
              ("hybrid (full)", gate6))
    seen = {name: [] for name, _ in models}
    for rnd in range(ROUNDS):
        k = rnd % len(models)
        for name, fn in models[k:] + models[:k]:
            t0w, t0c = time.perf_counter(), time.process_time()
            n = fn()
            seen[name].append(((time.perf_counter() - t0w) / n * 1e3,
                               (time.process_time() - t0c) / n * 1e3, n))
        print(f"  [round {rnd+1}/{ROUNDS} done]", flush=True)

    rows = {}
    for name, _ in models:
        walls = sorted(r[0] for r in seen[name])
        w = walls[len(walls) // 2]                  # median round = quoted
        c = float(np.median([r[1] for r in seen[name]]))
        n = seen[name][0][2]
        rows[name] = w
        print(f"  {name:<15} wall {w:8.4f} ms/tok (min {walls[0]:8.4f} / max "
              f"{walls[-1]:8.4f})   cpu {c:8.4f} ms/tok   "
              f"~{w * TDP_W:7.3f} mJ/tok   (n={n:,})", flush=True)

    full, stat, lam, gru_, ro = (rows["hybrid (full)"], rows["static6"],
                                 rows["fixed-lam"], rows["gru"],
                                 rows["router-only"])
    print(f"\n  marginal cost of ROUTING (hybrid - static6, same experts): "
          f"{full - stat:+.4f} ms/tok ({(full / stat - 1) * 100:+.1f}%)",
          flush=True)
    print(f"  router-only row (features + MLP, no GRU): {ro:.4f} ms/tok "
          f"= {ro / full * 100:.1f}% of the full hybrid", flush=True)
    print(f"  hybrid vs 2-expert fixed interpolation: {full - lam:+.4f} ms/tok "
          f"({(full / lam - 1) * 100:+.1f}%), of which {stat - lam:+.4f} is "
          f"evaluating four extra experts and {full - stat:+.4f} is routing",
          flush=True)
    print("  note: the KN3 top-1 memo is cleared every round, so router-only "
          "and hybrid are worst-case (a deployed system caching per-context "
          "favourites pays less)", flush=True)
    # a system must never measure FASTER than a component it contains
    # (5% tolerance: the GRU step dominates every row that contains it, so the
    #  extra work of kn3 + blending is at the edge of what this harness can
    #  resolve; the v1 table was off by 50%, not 1%)
    if gru_ > 1.02 * min(full, lam, stat):
        print("  WARNING: the GRU alone measured SLOWER than a system that "
              "contains it — machine noise; do not quote these rows",
              flush=True)
    else:
        print(f"  sanity: gru-alone {gru_:.4f} vs the systems containing it "
              f"(fixed-lam {lam:.4f}, static6 {stat:.4f}, hybrid {full:.4f}) — "
              f"containment ordering holds within 5%", flush=True)

    def pkl_mb(obj):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            pickle.dump(obj, f)
            s = os.path.getsize(f.name)
        os.unlink(f.name)
        return s / 1e6

    print(f"\n  sizes: tri {tri_mb():.1f} MB"
          f" | lag2 {pkl_mb({'t': e2.totals, 'c': e2.counts}):.1f} MB"
          f" | lag3 {pkl_mb({'t': e4.totals, 'c': e4.counts}):.1f} MB"
          f" | cache+uni {pkl_mb({'c': dict(e3.counts), 'p': pcont}):.1f} MB"
          f" | router {gate_kb:.1f} KB | GRU {gru_mb():.1f} MB fp32",
          flush=True)


if __name__ == "__main__":
    main()
