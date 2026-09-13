"""
mixer_lm.py — modern statistical LM: online mixture of cheap experts.

The idea (borrowed from the world's best lossless compressors, PAQ/zpaq
lineage): instead of ONE smoothed n-gram, run several small independent
experts in parallel and blend their probabilities with weights that are
LEARNED ONLINE and CONDITIONED ON CONTEXT EVIDENCE.

Experts (all pure-Python, all O(1) per query):
  E1  modified Kneser-Ney trigram  (this repo's engine, discount_scale=1.15)
  E2  lag-2 bigram  P(w | w_{t-2})        (dirichlet-smoothed, p_cont base)
  E3  cache window unigram P(w | last 512 tokens) (causal, boosts repeats)
  E4  lag-3 bigram P(w | w_{t-3})         (dirichlet-smoothed, p_cont base)
  E5  continuation unigram P_cont(w)      (static anchor)
  E6  personalization unigram (optional; learns from user-typed text)

Mixer:
  alpha_i(ctx) = softmax over experts of beta[i][bucket(ctx)]
  P(w|ctx)     = sum_i alpha_i * p_i(w|ctx)
  online update after each token (gradient on log P):
      beta[i][bucket] += eta * (q_i - alpha_i),
      q_i = alpha_i * p_i / P_mix     (responsibility)

beta has n_experts x 4 context-evidence buckets. Bucket = token mass of the
trigram's bigram context: 0 unseen / 1 / 2-5 / 6+. This is the *learned*
version of the repo's hand-designed trust curve.

Top-k suggestions: each expert nominates candidates (trigram top-40, cache
top-30, lag bigrams top-16 each, unigram/user top-30); the union is scored
with the exact mixed distribution and re-ranked.

Benchmark (WikiText-2, fixed 217,004-position test stream, weights warmed on
valid then FROZEN; cache updates causally):
  MKN alone 300.3 | repo BEAST 292.4 | static mix 263.9
  MIXER 199.5     | hindsight oracle 89.1

Usage:
  python mixer_lm.py            # benchmark + top-k accuracy
  python mixer_cli.py demo      # end-to-end tour with real prompts
"""

import heapq
import math
import os
import pickle
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ZipfNextWordPredictor import ZipfNextWordPredictor  # noqa: E402

_RES = os.environ.get("HEDGE_RESULTS",
                      os.path.join(HERE, "benchmarks", "results"))
GOLD = os.path.join(_RES, "gold_streams.pkl")
TRI_PKL = os.path.join(_RES, "beast_trigram.pkl")
FLOOR = 1e-12
EXPERT_FLOOR = 1e-10
CACHE_WINDOW = 512
CACHE_MU = 10.0
BIGRAM_MU = 2.0
USER_MU = 4.0
N_BUCKETS = 4  # 0: unseen bigram-ctx, 1: count==1, 2: 2-5, 3: 6+


class KnesserNeyExpert:
    """E1: exact MKN probabilities via the repo's _p_kn."""

    def __init__(self, model, discount_scale=1.15):
        self.m = model
        self.m.discount_scale = discount_scale
        self.m._recompute_discounts()
        self.m.zipf_prior_weight = 0.0
        self.w2i = model._w2i
        self.i2w = model._i2w
        self.unk = self.w2i["unk"]
        self.name = "kn3"

    def prob(self, wid, ctx_ids):
        p = self.m._p_kn(wid, tuple(ctx_ids[-2:]))
        return p if p > EXPERT_FLOOR else EXPERT_FLOOR

    def topk(self, ctx_words, n):
        return self.m._topk_kn(list(ctx_words), n)


class UnigramExpert:
    """E5: the KN continuation distribution (also the bigrams' base)."""

    def __init__(self, kn_expert):
        self.w2i = kn_expert.w2i
        self.i2w = kn_expert.i2w
        self.unk = kn_expert.unk
        self.p = kn_expert.m._pcont
        self.name = "uni"
        self._leaders = None

    def prob(self, wid, ctx_ids):
        p = self.p.get(wid, 0.0)
        return p if p > EXPERT_FLOOR else EXPERT_FLOOR

    def topk(self, ctx_words, n):
        if self._leaders is None:
            self._leaders = heapq.nlargest(30, self.p.items(),
                                           key=lambda kv: kv[1])
        return [(self.i2w[wid], p) for wid, p in self._leaders[:n]]


class DirichletBigramExpert:
    """E2/E4: P(w | w_{t-lag}), dirichlet-smoothed toward P_cont."""

    def __init__(self, unigram_expert, train_sents, lag, name):
        self.base = unigram_expert.p
        self.w2i = unigram_expert.w2i
        self.i2w = unigram_expert.i2w
        self.unk = unigram_expert.unk
        self.lag = lag
        self.name = name
        self.counts = defaultdict(int)
        self.totals = defaultdict(int)
        self.top = defaultdict(list)          # ctx -> top-16 [(count, wid)]
        for sent in train_sents:
            ids = [self.w2i.get(w, self.unk) for w in sent]
            for i in range(lag, len(ids)):
                ctx = ids[i - lag]
                wid = ids[i]
                self.counts[(ctx, wid)] += 1
                self.totals[ctx] += 1
                lst = self.top[ctx]
                if len(lst) < 16:
                    heapq.heappush(lst, (self.counts[(ctx, wid)], wid))
                    if len(lst) == 16:
                        heapq.heapify(lst)
                elif self.counts[(ctx, wid)] > lst[0][0]:
                    heapq.heapreplace(lst, (self.counts[(ctx, wid)], wid))

    def prob(self, wid, ctx_ids):
        if len(ctx_ids) < self.lag:
            p = self.base.get(wid, 0.0)
            return p if p > EXPERT_FLOOR else EXPERT_FLOOR
        ctx = ctx_ids[-self.lag]
        t = self.totals.get(ctx, 0)
        c = self.counts.get((ctx, wid), 0)
        p = (c + BIGRAM_MU * self.base.get(wid, 0.0)) / (t + BIGRAM_MU)
        return p if p > EXPERT_FLOOR else EXPERT_FLOOR

    def topk(self, ctx_words, n):
        if len(ctx_words) < self.lag:
            return []
        ctx = self.w2i.get(ctx_words[-self.lag], self.unk)
        return [(self.i2w[wid], c) for c, wid in
                heapq.nlargest(n, self.top.get(ctx, []))]


class CacheExpert:
    """E3: empirical distribution over the last CACHE_WINDOW tokens."""

    def __init__(self, unigram_expert):
        self.base = unigram_expert.p
        self.i2w = unigram_expert.i2w
        self.name = "cache"
        self.win = []
        self.counts = defaultdict(int)

    def prob(self, wid, ctx_ids):
        t = len(self.win)
        c = self.counts.get(wid, 0)
        p = (c + CACHE_MU * self.base.get(wid, 0.0)) / (t + CACHE_MU)
        return p if p > EXPERT_FLOOR else EXPERT_FLOOR

    def topk(self, ctx_words, n):
        return [(self.i2w[wid], c) for wid, c in
                heapq.nlargest(n, self.counts.items(), key=lambda kv: kv[1])]

    def update(self, wid):
        self.win.append(wid)
        self.counts[wid] += 1
        if len(self.win) > CACHE_WINDOW:
            old = self.win.pop(0)
            self.counts[old] -= 1
            if self.counts[old] <= 0:
                del self.counts[old]

    def snapshot(self):
        return (list(self.win), defaultdict(int, self.counts))

    def restore(self, state):
        self.win = list(state[0])
        self.counts = defaultdict(int, state[1])


class UserExpert(CacheExpert):
    """E6: personalization unigram over everything the user typed.
    Keyed by word STRING so out-of-vocabulary user words still count."""

    def __init__(self, unigram_expert):
        super().__init__(unigram_expert)
        self.name = "user"
        self.mu = USER_MU
        self.i2w = unigram_expert.i2w        # list: id -> word

    def prob(self, wid, ctx_ids):
        w = self.i2w[wid]                    # id -> word for string lookup
        t = len(self.win)
        c = self.counts.get(w, 0)
        p = (c + self.mu * self.base.get(wid, 0.0)) / (t + self.mu)
        return p if p > EXPERT_FLOOR else EXPERT_FLOOR

    def topk(self, ctx_words, n):
        return [(w, c) for w, c in
                heapq.nlargest(n, self.counts.items(), key=lambda kv: kv[1])]

    def observe_text(self, words):
        for w in words:
            self.update(w)


def bucket_of(tri_model, ctx_ids2):
    payload = tri_model._counts[2].get(ctx_ids2) if len(ctx_ids2) == 2 else None
    if payload is None:
        return 0
    t = payload[1]
    return 1 if t == 1 else (2 if t <= 5 else 3)


class Mixer:
    def __init__(self, experts, kn_expert, eta=0.1):
        self.experts = experts
        self.kn = kn_expert
        self.tri_model = kn_expert.m
        self.w2i = kn_expert.w2i
        self.i2w = kn_expert.i2w
        self.unk = kn_expert.unk
        self.eta = eta
        self.trainable = eta > 0
        self.beta = [[0.0] * N_BUCKETS for _ in experts]

    # -- internals ------------------------------------------------------- #

    def _alphas(self, bkt):
        a = [math.exp(row[bkt]) for row in self.beta]
        z = sum(a)
        return [v / z for v in a]

    def _probs(self, wid, ctx):
        return [e.prob(wid, ctx) for e in self.experts]

    # -- scoring --------------------------------------------------------- #

    def run(self, sents, collect_experts=False, oracle=False, freeze=False,
            update_state=True):
        """Mixed PP over gold streams; optionally per-expert PP + oracle.
        update_state=False: learn beta but don't advance the cache/user
        state (used for replay passes over already-absorbed text)."""
        n_exp = len(self.experts)
        log_sum = 0.0
        n = 0
        exp_ls = [0.0] * n_exp if collect_experts else None
        ora_ls = 0.0
        w2i, unk = self.w2i, self.unk
        for sent in sents:
            ids = [w2i.get(w, unk) for w in sent]
            for i in range(1, len(ids)):
                wid = ids[i]
                ctx = tuple(ids[max(0, i - 3):i])
                bkt = bucket_of(self.tri_model, ctx[-2:])
                p_list = self._probs(wid, ctx)
                a = self._alphas(bkt)
                pm = sum(ai * pi for ai, pi in zip(a, p_list))
                if pm <= 0.0:
                    pm = FLOOR
                log_sum += math.log(pm)
                n += 1
                if collect_experts:
                    for j, pj in enumerate(p_list):
                        exp_ls[j] += math.log(pj)
                if oracle:
                    ora_ls += max(math.log(pj) for pj in p_list)
                if self.trainable and not freeze:
                    row = bkt
                    for j in range(n_exp):
                        self.beta[j][row] += self.eta * \
                            ((a[j] * p_list[j]) / pm - a[j])
                for e in self.experts:
                    if update_state and e.name == "cache":
                        e.update(wid)
        out = {"pp_mix": math.exp(-log_sum / n), "n": n}
        if collect_experts:
            out["experts"] = {self.experts[j].name:
                              math.exp(-exp_ls[j] / n) for j in range(n_exp)}
        if oracle:
            out["pp_oracle_best_expert"] = math.exp(-ora_ls / n)
        return out

    # -- suggestions ------------------------------------------------------ #

    def _pool(self, ctx_words):
        pool = set()
        for w, _p in self.kn.topk(ctx_words, 40):
            wid = self.w2i.get(w)
            if wid is not None:
                pool.add(wid)
        for e in self.experts:
            if e.name in ("cache", "user"):
                n = 30 if e.name == "cache" else 20
            elif e.name in ("lag2", "lag3"):
                n = 16
            else:
                n = 30
            for w, _s in e.topk(ctx_words, n):
                wid = self.w2i.get(w)
                if wid is not None:
                    pool.add(wid)
        return pool

    def topk(self, ctx_words, k):
        """Mixed top-k suggestions for the given context words."""
        ctx_ids = tuple(self.w2i.get(w, self.unk) for w in ctx_words)
        bkt = bucket_of(self.tri_model, ctx_ids[-2:])
        a = self._alphas(bkt)
        scored = []
        for wid in self._pool(ctx_words):
            p_list = self._probs(wid, ctx_ids)
            scored.append((sum(ai * pi for ai, pi in zip(a, p_list)),
                           self.i2w[wid]))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(w, p) for p, w in scored[:k]]

    def complete(self, ctx_words, n_words=6):
        """Greedy completion: repeatedly append the top-1 suggestion."""
        words = list(ctx_words)
        out = []
        for _ in range(n_words):
            cands = self.topk(words, 1)
            if not cands:
                break
            w = cands[0][0]
            out.append(w)
            words.append(w)
        return out

    def observe(self, text, replay=True):
        """Absorb user text into the user/cache experts, then (optionally)
        replay-score it a few times so the mixer re-weights toward the
        experts that now explain the user's words (causal, user-local)."""
        sents = [[w for w in s.split() if w]
                 for s in text.lower().replace(".", " . ").split(".")]
        sents = [s for s in sents if len(s) >= 2]
        flat = [w for s in sents for w in s]
        for e in self.experts:
            if e.name == "user":
                e.observe_text(flat)
            elif e.name == "cache":
                for w in flat:
                    e.update(self.w2i.get(w, self.unk))
        if replay and self.trainable:
            for _ in range(5):
                self.run(sents, update_state=False)

    def alphas_for(self, ctx_words):
        """Blend weights the mixer currently assigns at this context."""
        ctx_ids = tuple(self.w2i.get(w, self.unk) for w in ctx_words)
        bkt = bucket_of(self.tri_model, ctx_ids[-2:])
        a = self._alphas(bkt)
        return {e.name: round(ai, 3) for e, ai in zip(self.experts, a)}


# ------------------------------------------------------------------------- #

def build(with_user=False, verbose=True):
    """with_user=False reproduces the benchmarked 5-expert config exactly."""
    with open(GOLD, "rb") as f:
        streams = pickle.load(f)
    tri_model = ZipfNextWordPredictor.load(TRI_PKL)
    tri_model.personalization = 0.0
    e1 = KnesserNeyExpert(tri_model, discount_scale=1.15)
    e5 = UnigramExpert(e1)
    if verbose:
        print("building lag-2 / lag-3 bigram experts ...")
    e2 = DirichletBigramExpert(e5, streams["train"], lag=2, name="lag2")
    e4 = DirichletBigramExpert(e5, streams["train"], lag=3, name="lag3")
    e3 = CacheExpert(e5)
    experts = [e1, e2, e3, e4, e5]
    if with_user:
        experts.append(UserExpert(e5))
    return streams, Mixer(experts, e1)


def warm_up(mixer, valid_sents, etas=(0.02, 0.1, 0.3, 1.0), verbose=True):
    """Pick eta on valid, warm up beta, return (eta, snapshot of beta)."""
    cache_snap = mixer.experts[2].snapshot()
    best = (None, 1e9)
    for eta in etas:
        mixer.experts[2].restore(cache_snap)
        mixer.eta = eta
        mixer.trainable = eta > 0
        r = mixer.run(valid_sents)
        if verbose:
            print(f"  eta={eta:4.2f}  valid PP={r['pp_mix']:.2f}")
        if r["pp_mix"] < best[1]:
            best = (eta, r["pp_mix"])
    eta_b = best[0]
    mixer.experts[2].restore(cache_snap)
    mixer.eta = eta_b
    mixer.run(valid_sents)
    return eta_b, [row[:] for row in mixer.beta]


def main():
    streams, mixer = build()
    valid = streams["valid"][:2500]
    test = streams["test"]

    print("== PP benchmark (protocol as benchmarks/) ==")
    r0 = mixer.run(valid, collect_experts=True, oracle=True)
    print(f"  single experts (valid): "
          f"{ {k: round(v, 2) for k, v in r0['experts'].items()} }")
    print(f"  equal-weight mix (valid): {r0['pp_mix']:.2f}   "
          f"oracle: {r0['pp_oracle_best_expert']:.2f}")

    eta_b, beta_frozen = warm_up(mixer, valid)
    mixer.beta = beta_frozen
    mixer.eta = 0.0                       # frozen weights for the test pass
    mixer.trainable = False
    r = mixer.run(test, collect_experts=True, oracle=True)
    print(f"== TEST ==")
    print(f"  experts: { {k: round(v, 2) for k, v in r['experts'].items()} }")
    print(f"  MIXER (eta={eta_b}, frozen): PP={r['pp_mix']:.2f}")
    print(f"  hindsight oracle: {r['pp_oracle_best_expert']:.2f}")
    print("  anchors: MKN 300.3 | repo BEAST 292.4 | prior result 199.5")

    print("== top-k accuracy (first 2,000 test sentences, pool-based) ==")
    mixer.eta = 0.0
    sub = test[:2000]
    hits = {"mix1": 0, "mix3": 0, "kn1": 0, "kn3": 0}
    n = 0
    in_pool = 0
    for sent in sub:
        ids = sent
        for i in range(1, len(ids)):
            ctx_words = ids[max(0, i - 3):i]
            gold = ids[i]
            pool = mixer._pool(ctx_words)
            wid = mixer.w2i.get(gold, mixer.unk)
            in_pool += wid in pool
            mixed = mixer.topk(ctx_words, 3)
            kn = [w for w, _p in mixer.kn.topk(ctx_words, 3)]
            top = [w for w, _p in mixed]
            hits["mix1"] += gold == top[0] if top else False
            hits["mix3"] += gold in top
            hits["kn1"] += gold == kn[0] if kn else False
            hits["kn3"] += gold in kn
            n += 1
    print(f"  positions scored: {n:,}   gold-in-pool: {in_pool/n:.1%}")
    print(f"  top-1: mixer {hits['mix1']/n:.1%} vs kn3 {hits['kn1']/n:.1%}")
    print(f"  top-3: mixer {hits['mix3']/n:.1%} vs kn3 {hits['kn3']/n:.1%}")
    print("  (repo's exact-pool protocol on full test: kn3 19.2% / 32.6%)")


if __name__ == "__main__":
    main()
