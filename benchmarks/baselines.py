"""Baseline n-gram models for the WikiText-2 benchmark.

Every baseline trains on the same gold token sentences and is scored by the
same harness loop (benchmarks/common.evaluate_harness) as the Zipf family
models. All top-k lists are computed efficiently (no full-vocabulary scans):
observed continuations merged with precomputed unigram leaders.
"""

import pickle
import re
from collections import defaultdict


def _dd_int():
    return defaultdict(int)


def _dd_nested():
    return defaultdict(_dd_int)


class _Base:
    name = "base"

    def train(self, sents):
        raise NotImplementedError

    def logprob(self, target, ctx):
        raise NotImplementedError

    def topk(self, ctx, k):
        raise NotImplementedError

    def predict_next_words(self, context, num_predictions=5, temperature=1.0):
        """Compat shim so the latency probe can call every model uniformly."""
        if isinstance(context, str):
            ctx = tuple(re.findall(r"\b\w+\b|[.,!?;]", context.lower()))
        else:
            ctx = tuple(context)
        return [(w, 0.0) for w in self.topk(ctx, num_predictions)]

    def size_probe(self):
        """(raw_pickle_bytes, zlib_bytes) for the size metric."""
        raw = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        import zlib
        return len(raw), len(zlib.compress(raw, 6))


class UnigramMLE(_Base):
    """P(w) = c(w) / N — the floor every LM should beat."""

    name = "Unigram MLE (floor)"

    def train(self, sents):
        self.uni = defaultdict(int)
        n = 0
        for sent in sents:
            for w in sent:
                self.uni[w] += 1
                n += 1
        self.total = n
        self.vocab = len(self.uni)
        self._sorted = sorted(self.uni.items(), key=lambda kv: (-kv[1], kv[0]))
        self._top_cache = {}

    def logprob(self, target, ctx):
        return self.uni.get(target, 0) / self.total

    def topk(self, ctx, k):
        return [w for w, _ in self._sorted[:k]]


class BigramLaplace(_Base):
    """Textbook add-1 (Laplace) bigram: P(w|prev) = (c+1) / (c(prev) + V)."""

    name = "Bigram add-1 (Laplace)"

    def train(self, sents):
        self.bi = defaultdict(_dd_int)
        self.prev_total = defaultdict(int)
        self.uni = defaultdict(int)
        vocab = set()
        for sent in sents:
            for w in sent:
                self.uni[w] += 1
                vocab.add(w)
            for i in range(1, len(sent)):
                self.bi[sent[i - 1]][sent[i]] += 1
                self.prev_total[sent[i - 1]] += 1
        self.V = len(vocab)
        self._sorted_uni = sorted(self.uni.items(), key=lambda kv: (-kv[1], kv[0]))
        self._topk_memo = {}

    def logprob(self, target, ctx):
        prev = ctx[-1] if ctx else None
        c = self.bi.get(prev, {}).get(target, 0) if prev is not None else 0
        denom = (self.prev_total.get(prev, 0) if prev is not None else 0) + self.V
        return (c + 1) / denom

    def topk(self, ctx, k):
        prev = ctx[-1] if ctx else None
        memo = self._topk_memo.setdefault(prev, {})
        out = memo.get(k)
        if out is None:
            conts = self.bi.get(prev, {})
            out = [w for w, _ in
                   sorted(conts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]
            memo[k] = out
        if len(out) < k:
            # all unobserved continuations tie at 1/(c(prev)+V); pad by
            # unigram frequency (deterministic tie-break)
            for w, _ in self._sorted_uni:
                if w not in out:
                    out.append(w)
                    if len(out) == k:
                        break
        return out


class BigramInterpolated(_Base):
    """Markov-chain-class model: bigram MLE interpolated with the unigram
    (fixed 0.75 / 0.25 Jelinek-Mercer)."""

    name = "Bigram interp (Markov-style)"

    LAMBDA = 0.75

    def train(self, sents):
        self.bi = defaultdict(_dd_int)
        self.prev_total = defaultdict(int)
        self.uni = defaultdict(int)
        for sent in sents:
            for w in sent:
                self.uni[w] += 1
            for i in range(1, len(sent)):
                self.bi[sent[i - 1]][sent[i]] += 1
                self.prev_total[sent[i - 1]] += 1
        self.total = sum(self.uni.values())
        self._sorted_uni = sorted(self.uni.items(), key=lambda kv: (-kv[1], kv[0]))
        self._puni = {w: c / self.total for w, c in self.uni.items()}
        self._pool_memo = {}

    def _p(self, target, prev):
        puni = self._puni.get(target, 0.0)
        if prev is None:
            return puni
        ptot = self.prev_total.get(prev, 0)
        if ptot == 0:
            return puni
        c = self.bi.get(prev, {}).get(target, 0)
        return self.LAMBDA * (c / ptot) + (1.0 - self.LAMBDA) * puni

    def logprob(self, target, ctx):
        return self._p(target, ctx[-1] if ctx else None)

    def topk(self, ctx, k):
        prev = ctx[-1] if ctx else None
        pool = self._pool_memo.get(prev)
        if pool is None:
            conts = self.bi.get(prev, {})
            if len(conts) > 40:
                top_conts = sorted(conts.items(),
                                   key=lambda kv: (-kv[1], kv[0]))[:40]
            else:
                top_conts = list(conts.items())
            pool = [w for w, _ in top_conts]
            pool += [w for w, _ in self._sorted_uni[:40]]
            self._pool_memo[prev] = pool
        cands = {w: self._p(w, prev) for w in set(pool)}
        ranked = sorted(cands.items(), key=lambda kv: (-kv[1], kv[0]))
        return [w for w, _ in ranked[:k]]


class WittenBellTrigram(_Base):
    """Textbook Witten-Bell trigram (Witten & Bell 1994; Manning & Schutze
    6.2.2, count view). Per context h with token count T and distinct
    continuation types D, the D/(T+D) reserved mass backs off and seen
    continuations are divided by (T + D):

        P(w|h) = c(w,h) / (T + D)  +  D/(T+D) * P(w|h')

    The distribution sums to 1 exactly at every level (T/(T+D) + D/(T+D)),
    with the unigram MLE as the bottom level.

    Note: an earlier edition of this baseline divided seen continuations by
    T alone while ALSO passing D/(T+D) to the lower level, which made the
    distribution sum to more than 1 on seen contexts and deflated its
    perplexity. Corrected here; the report uses the corrected numbers.
    """

    name = "Trigram Witten-Bell"

    def train(self, sents):
        self.uni = defaultdict(int)
        self.bi = defaultdict(_dd_int)
        self.tri = defaultdict(_dd_int)
        for sent in sents:
            for w in sent:
                self.uni[w] += 1
            for i in range(1, len(sent)):
                self.bi[sent[i - 1]][sent[i]] += 1
            for i in range(2, len(sent)):
                self.tri[tuple(sent[i - 2:i])][sent[i]] += 1
        self.total = sum(self.uni.values())
        self._puni = {w: c / self.total for w, c in self.uni.items()}
        self._bi_meta = {ctx: (sum(d.values()), len(d))
                         for ctx, d in self.bi.items()}
        self._tri_meta = {ctx: (sum(d.values()), len(d))
                          for ctx, d in self.tri.items()}
        self._sorted_uni = sorted(self.uni.items(), key=lambda kv: (-kv[1], kv[0]))
        self._pool_memo = {}

    def _wb(self, d, meta, ctx, w, p_bo):
        """One Witten-Bell level: c/(T+D) for seen, D/(T+D) to the backoff."""
        total, types = meta.get(ctx, (0, 0))
        denom = total + types
        if denom == 0:
            return p_bo
        return d.get(ctx, {}).get(w, 0) / denom + (types / denom) * p_bo

    def _p(self, w, c2, c1):
        # P(w | c1 c2) = WB3(w|c1c2, backoff=WB2(w|c2, backoff=P_uni(w)))
        if c2 is not None:
            p2 = self._wb(self.bi, self._bi_meta, (c2,), w,
                          self._puni.get(w, 0.0))
        else:
            p2 = self._puni.get(w, 0.0)
        if c1 is not None and c2 is not None:
            return self._wb(self.tri, self._tri_meta, (c1, c2), w, p2)
        return p2

    def logprob(self, target, ctx):
        c2 = ctx[-1] if len(ctx) >= 1 else None
        c1 = ctx[-2] if len(ctx) >= 2 else None
        return self._p(target, c2, c1)

    def topk(self, ctx, k):
        c2 = ctx[-1] if len(ctx) >= 1 else None
        c1 = ctx[-2] if len(ctx) >= 2 else None
        memo = self._pool_memo.setdefault((c1, c2), {})
        pool = memo.get(k)
        if pool is None:
            pool = []
            if c1 is not None and c2 is not None:
                tri = self.tri.get((c1, c2), {})
                pool += [w for w, _ in
                         sorted(tri.items(), key=lambda kv: (-kv[1], kv[0]))[:40]]
            if c2 is not None:
                bi = self.bi.get((c2,), {})
                pool += [w for w, _ in
                         sorted(bi.items(), key=lambda kv: (-kv[1], kv[0]))[:40]]
            pool += [w for w, _ in self._sorted_uni[:40]]
            memo[k] = pool
        cands = {w: self._p(w, c2, c1) for w in set(pool)}
        ranked = sorted(cands.items(), key=lambda kv: (-kv[1], kv[0]))
        return [w for w, _ in ranked[:k]]
