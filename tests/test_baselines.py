"""
Distribution-validity tests for the benchmark baselines.

Regression guard: an earlier WittenBellTrigram implementation divided seen
continuations by T while ALSO passing D/(T+D) to the lower level, so its
distribution summed to more than 1 on seen contexts and deflated its
perplexity. Every baseline must define a proper distribution.

Run:  python -m unittest tests.test_baselines -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "benchmarks"))

from baselines import (BigramInterpolated, BigramLaplace,  # noqa: E402
                       UnigramMLE, WittenBellTrigram)

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
]]
VOCAB = sorted({w for s in CORPUS for w in s})
CONTEXTS = [("the", "king"), ("said", "the"), ("the",), ("tower",),
            ("dragon", "the"), ("unseen", "context"), ("the", "unseen")]


class TestBaselineDistributions(unittest.TestCase):
    def _models(self):
        models = []
        for cls in (UnigramMLE, BigramLaplace, BigramInterpolated,
                    WittenBellTrigram):
            m = cls()
            m.train(CORPUS)
            models.append(m)
        return models

    def test_every_baseline_sums_to_one(self):
        for m in self._models():
            for ctx in CONTEXTS:
                total = sum(m.logprob(w, tuple(ctx)) for w in VOCAB)
                self.assertTrue(abs(total - 1.0) < 1e-9,
                                f"{m.name} ctx={ctx} sum={total}")

    def test_witten_bell_seen_context_sums_to_one(self):
        """The exact failure mode of the old implementation: seen trigram
        contexts must not allocate more than the available mass."""
        m = WittenBellTrigram()
        m.train(CORPUS)
        seen = [ctx for ctx in m._tri_meta if ctx in
                [tuple(c) for c in CONTEXTS]]
        for ctx in seen:
            total = sum(m.logprob(w, ctx) for w in VOCAB)
            self.assertTrue(abs(total - 1.0) < 1e-9,
                            f"seen ctx={ctx} sum={total}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
