"""
Distribution-validity tests for the learned router (the gate).

Regression guard, same family as tests/test_baselines.py: the first version of
gate_lm.collect() fed `log p_expert(w_gold)` in as the router's "vote"
features — the labels themselves. The router could then read the answer, its
"mixture" summed to ~2.7 over the vocabulary instead of 1, and every gate
perplexity in the money table was inflated by that factor while the
normalized baselines (KN3, GRU, fixed interpolation) were not.

These tests pin the invariant down from three directions:
  1. features depend on the CONTEXT only, never on the word being scored;
  2. every expert defines a normalized distribution over the vocabulary;
  3. therefore any context-only mixture of them is normalized too (this is
     what makes gate perplexity comparable to baseline perplexity);
  4. the cached .npz artifacts contain no feature column equal to a label
     column (skipped when the artifacts are absent).

Run:  python -m unittest tests.test_gate -v
"""

import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mixer_lm                                       # noqa: E402
from ZipfNextWordPredictor import ZipfNextWordPredictor  # noqa: E402
from gate_lm import NEXP, NF, FeatMaker, collect      # noqa: E402

CORPUS = [s.split() for s in [
    "the king said the queen was right .",
    "the queen said the king was wrong .",
    "the princess read the old book in the tower .",
    "a dragon guarded the tower and the book .",
    "the king rode to the tower with the queen .",
    "the dragon slept in the old tower .",
    "the book told of a king a queen and a dragon .",
    "the princess fed the dragon and read the book .",
    "the old king feared the dragon in the tower .",
    "the queen loved the old book more than the king .",
    "an unk token stands for a word the model has never seen .",
]]
CONTEXTS = [("the", "king"), ("said", "the"), ("the",), ("tower",),
            ("dragon", "the"), ("old", "book")]


def small_mixer():
    m = ZipfNextWordPredictor(corpus_texts=[" ".join(s) for s in CORPUS],
                              n_gram_size=3, smoothing="beast")
    m.personalization = 0.0
    e1 = mixer_lm.KnesserNeyExpert(m, discount_scale=1.15)
    e5 = mixer_lm.UnigramExpert(e1)
    e2 = mixer_lm.DirichletBigramExpert(e5, CORPUS, lag=2, name="lag2")
    e4 = mixer_lm.DirichletBigramExpert(e5, CORPUS, lag=3, name="lag3")
    e3 = mixer_lm.CacheExpert(e5)
    for w in ("the king said the queen was right . "
              "the princess read the old book in the tower").split():
        e3.update(e1.w2i.get(w, e1.unk))
    return mixer_lm.Mixer([e1, e2, e3, e4, e5], e1)


class TestFeaturesAreCausal(unittest.TestCase):
    def setUp(self):
        self.mixer = small_mixer()
        self.fm = FeatMaker(self.mixer)
        self.w2i, self.unk = self.mixer.w2i, self.mixer.unk

    def ids(self, words):
        return [self.w2i.get(w, self.unk) for w in words]

    def test_features_do_not_depend_on_the_scored_word(self):
        """Same context, different futures -> identical feature rows."""
        shared = "the king said the".split()
        stream_a = [shared + "queen was right .".split()]
        stream_b = [shared + "dragon slept old books .".split()]  # same length
        clean = self.mixer.experts[2].snapshot()
        Xa, La = collect(self.mixer, stream_a, "a")
        self.mixer.experts[2].restore(clean)
        Xb, Lb = collect(self.mixer, stream_b, "b")
        self.mixer.experts[2].restore(clean)
        k = len(shared) - 1                     # position predicting the fork
        self.assertTrue(np.allclose(Xa[k], Xb[k], atol=0, rtol=0),
                        f"features diverged for the same context:\n{Xa[k]}\n{Xb[k]}")
        self.assertFalse(np.allclose(La[k], Lb[k]),
                         "labels must differ (different gold words) — otherwise "
                         "the test is not exercising anything")

    def test_feature_width_and_no_nan(self):
        for ctx_words in CONTEXTS:
            x = self.fm.features(self.ids(ctx_words), len(ctx_words))
            self.assertEqual(x.shape, (NF,))
            self.assertTrue(np.isfinite(x).all())

    def test_labels_are_the_target_side_only(self):
        ctx = self.ids(("the", "king"))
        l1 = self.fm.labels(ctx, self.w2i.get("queen", self.unk))
        l2 = self.fm.labels(ctx, self.w2i.get("dragon", self.unk))
        self.assertFalse(np.allclose(l1, l2))
        x = self.fm.features(ctx, 2)
        for j in range(NF):                     # no feature column is a label
            self.assertFalse(np.isclose(x[j], l1[0]) and np.isclose(x[j], l2[0]))


class TestMixtureIsNormalized(unittest.TestCase):
    """The reason the perplexities are comparable at all."""

    def setUp(self):
        self.mixer = small_mixer()
        self.experts = self.mixer.experts
        self.vocab = list(range(len(self.mixer.i2w)))

    def test_each_expert_is_a_distribution(self):
        for ctx_words in CONTEXTS:
            ctx = tuple(self.mixer.w2i.get(w, self.mixer.unk)
                        for w in ctx_words)
            for e in self.experts:
                mass = math.fsum(e.prob(w, ctx) for w in self.vocab)
                self.assertAlmostEqual(mass, 1.0, delta=2e-3,
                                       msg=f"{e.name} mass {mass} after {ctx}")

    def test_context_only_mixture_is_a_distribution(self):
        rng = np.random.default_rng(0)
        for ctx_words in CONTEXTS:
            ctx = tuple(self.mixer.w2i.get(w, self.mixer.unk)
                        for w in ctx_words)
            a = rng.random(NEXP)
            a /= a.sum()
            mass = math.fsum(sum(ai * e.prob(w, ctx)
                                 for ai, e in zip(a, self.experts))
                             for w in self.vocab)
            self.assertAlmostEqual(mass, 1.0, delta=2e-3,
                                   msg=f"mixture mass {mass} after {ctx}")

    def test_top1_is_a_real_expert_pick(self):
        for ctx_words in CONTEXTS:
            ctx = tuple(self.mixer.w2i.get(w, self.mixer.unk)
                        for w in ctx_words)
            for e in self.experts:
                wid, p = e.top1(ctx)
                self.assertIn(wid, set(self.vocab))
                best = max(e.prob(w, ctx) for w in self.vocab)
                self.assertLessEqual(p, best + 1e-9,
                                     f"{e.name}.top1 p={p} > max prob {best}")
                self.assertGreater(p, 0.0)


class TestCacheTop1Incremental(unittest.TestCase):
    def test_matches_rescan_over_a_stream(self):
        mixer = small_mixer()
        cache = mixer.experts[2]
        rng = np.random.default_rng(1)
        vocab = list(range(len(mixer.i2w)))
        for step in range(400):
            wid = int(rng.choice(vocab)) if rng.random() < 0.3 else \
                int(rng.choice(vocab[:8]))
            cache.update(wid)
            if step % 7 == 0:
                w, p = cache.top1(())
                true_w, true_c = max(cache.counts.items(), key=lambda kv: kv[1])
                self.assertEqual(cache.counts[w], true_c,
                                 f"step {step}: top1 {w} count != {true_c}")
                t = len(cache.win)
                want = (true_c + mixer_lm.CACHE_MU * cache.base.get(w, 0.0)) \
                       / (t + mixer_lm.CACHE_MU)
                self.assertAlmostEqual(p, want, delta=1e-12)


class TestCachedArtifactsHaveNoLeak(unittest.TestCase):
    """Skipped unless the collected .npz artifacts are present."""

    def test_no_feature_column_equals_a_label_column(self):
        res = os.environ.get("HEDGE_RESULTS",
                             os.path.join(ROOT, "benchmarks", "results"))
        paths = [os.path.join(res, f"gate_{k}.npz")
                 for k in ("train", "valid", "test")]
        if not all(os.path.exists(p) for p in paths):
            self.skipTest(f"no gate npz artifacts under {res}")
        for p in paths:
            d = np.load(p)
            X, L = d["X"], d["L"]
            self.assertEqual(X.shape[1], NF, f"{p}: stale feature width")
            for j in range(X.shape[1]):
                for k in range(L.shape[1]):
                    self.assertFalse(np.array_equal(X[:, j], L[:, k]),
                                     f"{p}: feature {j} IS label {k} (leak)")
            # a leaked vote feature correlates ~1.0 with its label column.
            # A CONSTANT feature (zero variance) has undefined correlation and
            # cannot carry label information, so it is skipped.
            for k in range(min(5, L.shape[1])):
                xk = X[:, k].astype(np.float64)
                lk = L[:, k].astype(np.float64)
                if xk.std() == 0.0 or lk.std() == 0.0:
                    continue
                c = float(np.corrcoef(xk, lk)[0, 1])
                self.assertLess(c, 0.99, f"{p}: feature {k} tracks label {k} "
                                         f"(corr {c:.4f})")


if __name__ == "__main__":
    unittest.main()
