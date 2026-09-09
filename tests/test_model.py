"""
Sanity tests for the BEAST edition of ZipfNextWordPredictor.

Run:  python -m unittest tests.test_model -v
"""

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ZipfNextWordPredictor import ZipfNextWordPredictor  # noqa: E402

CORPUS = [
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
]

HELD_OUT = (
    "the king read the book in the tower . "
    "the queen saw the dragon near the old tower . "
    "the princess said the dragon was right ."
)


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


class TestAllEngines(unittest.TestCase):
    ENGINES = ["beast", "modified_kneser_ney", "kneser_ney", "zipf"]

    def _make(self, engine, **kw):
        return ZipfNextWordPredictor(corpus_texts=list(CORPUS),
                                     smoothing=engine, **kw)

    def test_predictions_structure_all_engines(self):
        for engine in self.ENGINES:
            m = self._make(engine)
            preds = m.predict_next_words("the king said the", num_predictions=3)
            self.assertIsInstance(preds, list, engine)
            self.assertLessEqual(len(preds), 3, engine)
            for word, prob in preds:
                self.assertIsInstance(word, str, engine)
                self.assertGreater(prob, 0.0, engine)
            # probabilities non-increasing
            probs = [p for _, p in preds]
            self.assertEqual(probs, sorted(probs, reverse=True), engine)

    def test_legacy_mode_matches_v4_on_predictions(self):
        """The 'zipf' engine must reproduce v4 behaviour bit-for-bit."""
        import importlib.util
        legacy_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "benchmarks", "legacy",
            "ZipfNextWordPredictor_v4.py")
        spec = importlib.util.spec_from_file_location("legacy_zipf_v4",
                                                      legacy_path)
        legacy_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy_mod)

        old = legacy_mod.ZipfNextWordPredictor(corpus_texts=list(CORPUS),
                                               n_gram_size=3, zipf_weight=0.7)
        new = self._make("zipf", n_gram_size=3, zipf_weight=0.7)
        for ctx in ["the king", "the queen said the", "dragon", ""]:
            self.assertEqual(old.predict_next_words(ctx, 5),
                             new.predict_next_words(ctx, 5),
                             f"legacy mismatch for context {ctx!r}")
        old_eval = old.evaluate(HELD_OUT, k=3)
        new_eval = new.evaluate(HELD_OUT, k=3)
        self.assertTrue(_close(old_eval["perplexity"],
                               new_eval["perplexity"], 1e-6))
        self.assertTrue(_close(old_eval["top_k_accuracy"],
                               new_eval["top_k_accuracy"], 1e-9))

    def test_kn_distribution_sums_to_one(self):
        """For every context the full KN distribution must sum to 1."""
        for engine in ["beast", "modified_kneser_ney", "kneser_ney"]:
            m = self._make(engine, zipf_prior_weight=0.2)
            for ctx_words in [("the",), ("the", "king"), ("dragon", "the"),
                              ("unseen", "context"), ()]:
                ctx = m._context_ids(list(ctx_words))
                total = 0.0
                for wid in range(len(m._i2w)):
                    total += m._p_kn(wid, ctx)
                self.assertTrue(_close(total, 1.0, 1e-9),
                                f"{engine} ctx={ctx_words} sum={total}")

    def test_topk_shortcut_matches_bruteforce(self):
        """The pruned-merge top-k must equal brute-force ranking."""
        for engine in ["beast", "modified_kneser_ney"]:
            m = self._make(engine, zipf_prior_weight=0.15)
            for ctx_words in [("the",), ("the", "king"), ("said", "the"),
                              ("unseen", "words")]:
                fast = m._topk_kn(list(ctx_words), 5)
                brute = m._kn_full_topk_exact(list(ctx_words), 5)
                self.assertEqual([w for w, _ in fast], [w for w, _ in brute],
                                 f"{engine} ctx={ctx_words}\nfast={fast}\n"
                                 f"brute={brute}")

    def test_evaluate_topk_matches_topk_kn(self):
        """evaluate()'s cached top-k path must agree with _topk_kn."""
        m = self._make("beast", zipf_prior_weight=0.15)
        text = " ".join(CORPUS)
        res = m.evaluate(text, k=3)
        # manually recompute top-3 hits over the same stream
        hits = 0
        tokens = 0
        for sentence in m.preprocess(text):
            words = [w.lower() for w in sentence]
            for i in range(1, len(words)):
                ctx = words[max(0, i - 2):i]
                target = words[i]
                topk = [w for w, _ in m._topk_kn(ctx, 3)]
                if target in topk:
                    hits += 1
                tokens += 1
        self.assertEqual(tokens, res["tokens"])
        self.assertTrue(abs(hits / tokens - res["top_k_accuracy"]) < 1e-9,
                        f"evaluate={res['top_k_accuracy']} "
                        f"manual={hits / tokens}")

    def test_user_personalization_changes_predictions(self):
        m = self._make("beast", personalization=0.9, personalization_halflife=20)
        before = m.predict_next_words("the princess decided to", 3)
        m.observe("the princess decided to write python code every day . "
                  "the princess said python is fun .")
        after = m.predict_next_words("the princess decided to", 3)
        self.assertTrue(after)
        self.assertNotEqual([w for w, _ in before], [w for w, _ in after])
        stats = m.user_stats()
        self.assertGreater(stats["words"], 0)
        self.assertGreater(stats["effective_alpha"], 0.0)
        m.forget()
        self.assertEqual(m.user_stats()["words"], 0)

    def test_save_load_roundtrip_all_engines(self):
        for engine in self.ENGINES:
            m = self._make(engine)
            m.observe("the princess likes python .")
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "model.pkl")
                m.save(path)
                m2 = ZipfNextWordPredictor.load(path)
                self.assertEqual(m2.smoothing, engine)
                self.assertEqual(
                    m.predict_next_words("the king said the", 5),
                    m2.predict_next_words("the king said the", 5))
                self.assertEqual(
                    m.predict_next_words("unseen context words", 5),
                    m2.predict_next_words("unseen context words", 5))
                # user layer survives
                self.assertEqual(m2.user_stats()["words"],
                                 m.user_stats()["words"])

    def test_compressed_save_load(self):
        m = self._make("beast")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.zf")
            m.save(path, compress=True)
            m2 = ZipfNextWordPredictor.load(path)
            self.assertEqual(m.predict_next_words("the old", 3),
                             m2.predict_next_words("the old", 3))

    def test_v4_pickle_upgrade_in_place(self):
        """A v4-style payload must load and gain beast-level scoring."""
        import importlib.util
        legacy_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "benchmarks", "legacy",
            "ZipfNextWordPredictor_v4.py")
        spec = importlib.util.spec_from_file_location("legacy_zipf_v4b",
                                                      legacy_path)
        legacy_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy_mod)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v4.pkl")
            old = legacy_mod.ZipfNextWordPredictor(corpus_texts=list(CORPUS),
                                                   n_gram_size=3)
            old.save(path)
            upgraded = ZipfNextWordPredictor.load(path)
            self.assertEqual(upgraded.smoothing, "beast")
            self.assertGreater(upgraded.total_words, 0)
            preds = upgraded.predict_next_words("the king said the", 5)
            self.assertTrue(preds)
            # KN engine properties hold on the upgraded model
            ctx = upgraded._context_ids(["the", "king"])
            total = sum(upgraded._p_kn(w, ctx) for w in range(len(upgraded._i2w)))
            self.assertTrue(_close(total, 1.0, 1e-9))

    def test_tune_improves_or_preserves_perplexity(self):
        for engine in ["beast", "zipf"]:
            m = self._make(engine)
            before = m.evaluate(HELD_OUT, k=3)["perplexity"]
            report = m.tune(HELD_OUT, max_tokens=500)
            after = m.evaluate(HELD_OUT, k=3)["perplexity"]
            self.assertLessEqual(after, before + 1e-9,
                                 f"{engine}: tune made perplexity worse "
                                 f"({before:.2f} -> {after:.2f})")
            self.assertIn("best", report)
            self.assertIn("trials", report)

    def test_train_from_files_and_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            f1 = os.path.join(tmp, "a.txt")
            with open(f1, "w", encoding="utf-8") as f:
                f.write(" ".join(CORPUS))
            m = ZipfNextWordPredictor(smoothing="beast")
            stats = m.train_from_files(tmp)
            self.assertEqual(stats["num_files"], 1)
            self.assertGreater(stats["words"], 0)
            self.assertGreater(stats["vocab"], 0)

    def test_completion_returns_text(self):
        m = self._make("beast")
        out = m.predict_completion("the king said", num_words=4)
        self.assertTrue(out.startswith("the king said"))

    def test_sentence_logprob_consistency(self):
        m = self._make("beast")
        res = m.sentence_logprob(HELD_OUT)
        ev = m.evaluate(HELD_OUT, k=3)
        self.assertEqual(res["tokens"], ev["tokens"])
        self.assertTrue(_close(res["perplexity"], ev["perplexity"], 1e-6))

    def test_model_info(self):
        m = self._make("beast")
        info = m.model_info()
        self.assertEqual(info["smoothing"], "beast")
        self.assertIn("discounts_per_order", info)
        self.assertIn("zipf_prior_weight", info)

    def test_higher_order_support(self):
        m5 = ZipfNextWordPredictor(corpus_texts=list(CORPUS) * 5,
                                   n_gram_size=5, smoothing="beast")
        preds = m5.predict_next_words("the old king said the queen was", 3)
        self.assertTrue(preds)
        ctx = m5._context_ids(["the", "old", "king", "said"])
        total = sum(m5._p_kn(w, ctx) for w in range(len(m5._i2w)))
        self.assertTrue(_close(total, 1.0, 1e-9))

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            ZipfNextWordPredictor(n_gram_size=1)
        with self.assertRaises(ValueError):
            ZipfNextWordPredictor(smoothing="quantum")
        with self.assertRaises(ValueError):
            ZipfNextWordPredictor(prior_mode="laplace")


class TestWittenBellPriorMode(unittest.TestCase):
    """Cross-corpus comparator: Witten-Bell confidence weighting in the same
    junction as the Zipf-continuation prior (backoff weight D/(T+D),
    parameter-free, no prior diversion)."""

    def _make(self, **kw):
        return ZipfNextWordPredictor(corpus_texts=list(CORPUS), **kw)

    def test_default_prior_mode_is_zipf(self):
        m = self._make(smoothing="beast")
        self.assertEqual(m.prior_mode, "zipf")

    def test_wb_distribution_sums_to_one(self):
        m = self._make(smoothing="beast", prior_mode="witten_bell",
                       zipf_prior_weight=0.0)
        for ctx_words in [("the",), ("the", "king"), ("dragon", "the"),
                          ("unseen", "context"), ()]:
            ctx = m._context_ids(list(ctx_words))
            total = sum(m._p_kn(wid, ctx) for wid in range(len(m._i2w)))
            self.assertTrue(_close(total, 1.0, 1e-9),
                            f"ctx={ctx_words} sum={total}")

    def test_wb_topk_matches_bruteforce(self):
        m = self._make(smoothing="beast", prior_mode="witten_bell",
                       zipf_prior_weight=0.0)
        for ctx_words in [("the",), ("the", "king"), ("said", "the"),
                          ("unseen", "words")]:
            fast = m._topk_kn(list(ctx_words), 5)
            brute = m._kn_full_topk_exact(list(ctx_words), 5)
            self.assertEqual([w for w, _ in fast], [w for w, _ in brute],
                             f"ctx={ctx_words}\nfast={fast}\nbrute={brute}")

    def test_wb_differs_from_zipf_mode(self):
        """The two regularizers must produce different predictions somewhere
        (otherwise the comparison would be vacuous)."""
        mz = self._make(smoothing="beast", prior_mode="zipf",
                        zipf_prior_weight=0.15)
        mw = self._make(smoothing="beast", prior_mode="witten_bell",
                        zipf_prior_weight=0.0)
        pairs = [
            (mz.predict_next_words("the king said the", 5),
             mw.predict_next_words("the king said the", 5)),
            (mz.predict_next_words("read the", 5),
             mw.predict_next_words("read the", 5)),
            (mz.predict_next_words("the dragon", 5),
             mw.predict_next_words("the dragon", 5)),
        ]
        self.assertTrue(any(a != b for a, b in pairs))

    def test_wb_mode_beats_nothing_assumption_not_encoded(self):
        """Sanity: WB mode is a valid engine configuration — evaluate() runs
        and returns finite perplexity. No quality assumption is encoded."""
        m = self._make(smoothing="beast", prior_mode="witten_bell",
                       zipf_prior_weight=0.0)
        res = m.evaluate(HELD_OUT, k=3)
        self.assertGreater(res["tokens"], 0)
        self.assertTrue(math.isfinite(res["perplexity"]))
        self.assertGreater(res["perplexity"], 0.0)

    def test_wb_tune_sweeps_only_discount_scale(self):
        m = self._make(smoothing="beast", prior_mode="witten_bell",
                       zipf_prior_weight=0.0)
        out = m.tune(HELD_OUT, discount_scales=[0.85, 1.0, 1.15])
        self.assertEqual(out["trials_run"], 3)  # ds grid x zw=[0] x adaptive=[False]
        for trial in out["trials"]:
            self.assertEqual(trial["zipf_prior_weight"], 0.0)
            self.assertEqual(trial["prior_mode"], "witten_bell")

    def test_wb_save_load_roundtrip(self):
        import tempfile
        m = self._make(smoothing="beast", prior_mode="witten_bell",
                       zipf_prior_weight=0.0)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wb_model.pkl")
            m.save(path)
            loaded = ZipfNextWordPredictor.load(path)
        self.assertEqual(getattr(loaded, "prior_mode", "zipf"),
                         "witten_bell")
        a = m.predict_next_words("the king said the", 5)
        b = loaded.predict_next_words("the king said the", 5)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
