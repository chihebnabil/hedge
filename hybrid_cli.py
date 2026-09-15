"""
hybrid_cli.py — interactive tester for the trained hybrid stack
(5 statistical experts + GRU + neural gate, the Phase-4 system).

First run trains + caches the headline router (valid-tuned, hidden size by
2-fold CV on validation — the audited config; saved to gate_fair_h<H>.npz).
Then:

  python hybrid_cli.py suggest "the united states of"   # one-shot top-8
  python hybrid_cli.py complete "once upon a"           # greedy continuation
  python hybrid_cli.py                                  # interactive REPL:
    suggest <text>    top next words + gate mixture weights
    complete <text>   greedy continuation (10 words)
    observe <text>    feed text (cache + GRU context warm-up)
    reset             clear cache + GRU state
    quit

Honest demo note: the GRU reads only your prompt (cold start), while in the
benchmark it had streamed the whole corpus first — its vote is weaker here,
and the gate weights that vote accordingly. observe warms it up.
"""

import json
import math
import os
import sys

import numpy as np
import torch

from gate_lm import FeatMaker
from mixer_lm import build
import nn_expert

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("HEDGE_RESULTS",
                         os.path.join(HERE, "benchmarks", "results"))


def _gate_npz():
    """The headline router's weights (written by gate_hybrid.py): the
    valid-tuned one if the money table has run, else the train-trained one."""
    fair = os.path.join(RESULTS, "gate_fair.json")
    if os.path.exists(fair):
        with open(fair) as f:
            return os.path.join(RESULTS, f"gate_fair_h{json.load(f)['h']}.npz")
    from gate_hybrid import selected_h
    return os.path.join(RESULTS, f"gate_mlp_h{selected_h(128)}.npz")


GATE_NPZ = _gate_npz()
NAMES = ["kn3", "lag2", "cache", "lag3", "uni", "gru"]
W1 = b1 = W2 = b2 = None


def ensure_gate():
    """Load cached gate weights, or train + save them (one time)."""
    global W1, b1, W2, b2
    if os.path.exists(GATE_NPZ):
        z = np.load(GATE_NPZ)
        W1, b1, W2, b2 = z["W1"], z["b1"], z["W2"], z["b2"]
        return
    from gate_hybrid import fair_router, load_split, nll_pp
    (Yva, Lva) = load_split("valid")
    pred, kb, h, cv, _curve, wts = fair_router(Yva, Lva, nexp=Lva.shape[1])
    te_pp = nll_pp(pred(load_split("test")[0]), load_split("test")[1])
    W1, b1, W2, b2 = wts
    np.savez(GATE_NPZ, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"valid-tuned router h={h}: CV valid PP {cv:.2f}, test PP "
          f"{te_pp:.2f} ({kb:.1f} KB) — saved {GATE_NPZ}", flush=True)


def _predict_np(X):
    h = np.maximum(X @ W1 + b1, 0.0)
    z = h @ W2 + b2
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Hybrid:
    def __init__(self):
        from nn_expert import GRULM, OUT
        self.streams, self.mixer = build()
        self.tri = self.mixer.tri_model
        self.e_kn3 = self.mixer.experts[0]
        self.e_lag2 = self.mixer.experts[1]
        self.e_cache = self.mixer.experts[2]
        self.e_lag3 = self.mixer.experts[3]
        self.e_uni = self.mixer.experts[4]
        self.w2i = self.mixer.w2i
        self.i2w = self.mixer.i2w
        self.unk = self.w2i["unk"]
        self.fm = FeatMaker(self.mixer)

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        torch.manual_seed(42)
        self.V = len(self.i2w) + 1
        nn_expert.EOS = self.V - 1
        self.gru = GRULM(self.V)
        self.gru.load_state_dict(torch.load(
            os.path.join(OUT, "nn_gru_best.pt")))
        self.gru.eval()
        self.h = None

    def features(self, ctx_ids):
        """Causal, CANDIDATE-INDEPENDENT gate input (gate_lm.FeatMaker).
        The router sees the context only, never the word being scored, so the
        mixture below is a normalized distribution over the vocabulary.
        The prompt is the sentence so far, so i = len(ctx_ids). Returned as a
        (1, NF) row so it feeds straight into _predict_np."""
        return self.fm.features(list(ctx_ids), len(ctx_ids)).reshape(1, -1)

    def gru_next(self, ids):
        """Next-token log-probs after streaming ids (carries h)."""
        if not ids:
            ids = [self.unk]
        with torch.no_grad():
            logits, self.h = self.gru(
                torch.tensor(ids, dtype=torch.int64).view(1, -1), self.h)
        return torch.log_softmax(logits[0, -1], dim=-1)

    def suggest(self, text, topk=8):
        scored = self._mixture_rank(text.split())
        out = []
        for mix, cid, a in scored[:topk]:
            w = self.i2w[cid] if cid < len(self.i2w) else "<eos>"
            top2 = sorted(zip(NAMES, a), key=lambda kv: -kv[1])[:2]
            out.append((w, math.exp(mix), mix, top2))
        return out

    def feed(self, text):
        for w in text.split():
            wid = self.w2i.get(w, self.unk)
            self.e_cache.update(wid)
            self.gru_next([wid])

    def _mixture_rank(self, ctx_words, extra_cids=()):
        """Gate-mixture ranking over the benchmark's candidate pool.
        Returns [(logP_mixture, cid, gate_weights)] sorted best-first.
        Does NOT advance model state."""
        ctx_ids = [self.w2i.get(w, self.unk) for w in ctx_words]
        h_save = self.h
        gru_lp = self.gru_next(ctx_ids)
        self.h = h_save

        cands = set(extra_cids)
        for pool in (self.e_kn3.topk(ctx_words, 40),
                     self.e_lag2.topk(ctx_words, 16),
                     self.e_cache.topk(ctx_words, 30),
                     self.e_lag3.topk(ctx_words, 16),
                     self.e_uni.topk(ctx_words, 30)):
            cands.update(self.w2i.get(w, self.unk) for w, _ in pool)
        cands.update(gru_lp.topk(40).indices.tolist())

        a = _predict_np(self.features(ctx_ids)).ravel()   # once per context
        scored = []
        for cid in cands:
            votes = np.array([math.log(max(e.prob(cid, ctx_ids), 1e-10))
                              for e in (self.e_kn3, self.e_lag2,
                                        self.e_cache, self.e_lag3,
                                        self.e_uni)]
                             + [float(gru_lp[cid])])
            mix = math.log(float(np.sum(a * np.exp(votes))) + 1e-300)
            scored.append((mix, cid, a))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return scored

    def cloze(self, text):
        """The honest live test: walk real text, predict each next word,
        count top-1/top-3 hits (the benchmark metric, made tangible).
        Streams state causally (cache + GRU advance on the true words)."""
        words = text.split()
        m = len(words)
        n = hit1 = hit3 = 0
        nll = 0.0
        marks = []
        for i in range(1, m):
            ctx_words = words[max(0, i - 3):i]
            gold = words[i]
            gold_id = self.w2i.get(gold, self.unk)
            scored = self._mixture_rank(ctx_words, extra_cids=(gold_id,))
            lp_of = {cid: s for s, cid, _a in scored}
            gold_lp = lp_of[gold_id]
            nll -= gold_lp
            rank = [cid for _s, cid, _a in scored].index(gold_id)
            n += 1
            hit1 += rank == 0
            hit3 += rank < 3
            if rank == 0:
                marks.append(gold)
            elif rank < 3:
                marks.append(f"[{gold}]")
            else:
                best = (self.i2w[scored[0][1]]
                        if scored[0][1] < len(self.i2w) else "<eos>")
                marks.append(f"{gold}*({best})")
            self.feed(gold)               # causal state advance
        pp = math.exp(nll / max(n, 1))
        print("  " + " ".join(marks))
        print(f"  gold = word  [gold] = in top-3  gold*(model's guess) = miss")
        print(f"  top-1 {hit1/max(n,1):.1%}  |  top-3 {hit3/max(n,1):.1%}  |  "
              f"text PP {pp:.1f}")
        return hit1, hit3, n

    def _next_counts(self):
        """Lazy bigram->next-word counts over the TRAIN stream (for why)."""
        if not hasattr(self, "_nxt"):
            from collections import Counter
            nxt = {}
            for sent in self.streams["train"]:
                for i in range(2, len(sent)):
                    key = (sent[i - 2], sent[i - 1])
                    if key not in nxt:
                        nxt[key] = Counter()
                    nxt[key][sent[i]] += 1
            self._nxt = nxt
        return self._nxt

    def why(self, target, ctx_words):
        """Transparency: raw corpus counts vs each expert's estimate vs the
        final gate mixture probability of `target` after `ctx_words`."""
        ctx = tuple(ctx_words)
        key = tuple(ctx[-2:])
        nxt = self._next_counts().get(key, {})
        total = sum(nxt.values())
        cid = self.w2i.get(target, self.unk)
        ctx_ids = [self.w2i.get(w, self.unk) for w in ctx_words]
        h_save = self.h
        gru_lp = self.gru_next(ctx_ids)
        self.h = h_save
        print(f"  after {' '.join(key)!r} — train corpus: {total} occurrences")
        for w, c in nxt.most_common(8):
            print(f"    {w:<12} {c:4d}  ({c/total:.1%})")
        print(f"  estimates for {target!r}:")
        est = [("train count", f"{nxt.get(target, 0)} ({nxt.get(target, 0)/max(total,1):.1%})")]
        for name, e in zip(NAMES[:-1], (self.e_kn3, self.e_lag2,
                                        self.e_cache, self.e_lag3,
                                        self.e_uni)):
            est.append((name, f"{e.prob(cid, ctx_ids):.5f}"))
        est.append(("gru", f"{math.exp(float(gru_lp[cid])):.5f}"))
        for name, v in est:
            print(f"    {name:<12} {v}")
        a = _predict_np(self.features(ctx_ids)).ravel()
        votes = np.array([math.log(max(e.prob(cid, ctx_ids), 1e-10))
                          for e in (self.e_kn3, self.e_lag2, self.e_cache,
                                    self.e_lag3, self.e_uni)]
                         + [float(gru_lp[cid])])
        print(f"    FINAL P    {math.exp(math.log(float(a @ np.exp(votes)) + 1e-300)):.5f}"
              f"   (gate: " + " ".join(f"{n} {x:.2f}" for n, x in
                                       sorted(zip(NAMES, a), key=lambda kv: -kv[1])[:3]) + ")")

    def reset(self):
        self.e_cache.clear()
        self.h = None


def show(sug):
    print(f"{'word':<16} {'P(word)':>9}  {'logP':>8}  gate weight (top-2)")
    for w, p, lp, top2 in sug:
        wts = "  ".join(f"{n} {a:.2f}" for n, a in top2)
        print(f"{w:<16} {p:9.5f}  {lp:8.2f}  {wts}")


def repl(H):
    print("hybrid REPL — suggest <text> | complete <text> | cloze <text> | "
          "why <word> <context...> | observe <text> | reset | quit")
    while True:
        try:
            line = input("hybrid> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        cmd, _, arg = line.partition(" ")
        arg = arg.strip().strip('"').strip("'")
        if cmd == "quit":
            break
        if cmd == "suggest":
            show(H.suggest(arg))
        elif cmd == "complete":
            words = arg.split()
            for _ in range(10):
                top = H.suggest(" ".join(words), topk=1)
                if not top:
                    break
                words.append(top[0][0])
                H.feed(top[0][0])
            print("  " + " ".join(words))
        elif cmd == "cloze":
            H.cloze(arg)
        elif cmd == "why":
            parts = arg.split()
            if len(parts) < 2:
                print("  usage: why <word> <context words...>")
            else:
                H.why(parts[0], parts[1:])
        elif cmd == "observe":
            H.feed(arg)
            print("  fed", len(arg.split()), "tokens (cache + GRU context)")
        elif cmd == "reset":
            H.reset()
            print("  cache + GRU state cleared")
        else:
            print("  ? command")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "repl"
    ensure_gate()
    H = Hybrid()
    if mode == "suggest":
        show(H.suggest(" ".join(sys.argv[2:])))
    elif mode == "complete":
        words = sys.argv[2:]
        for _ in range(10):
            top = H.suggest(" ".join(words), topk=1)
            words.append(top[0][0])
            H.feed(top[0][0])
        print(" ".join(words))
    elif mode == "cloze":
        H.cloze(" ".join(sys.argv[2:]))
    elif mode == "why":
        H.why(sys.argv[2], sys.argv[3:])
    else:
        repl(H)


if __name__ == "__main__":
    main()
