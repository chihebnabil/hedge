"""
ZipfNextWordPredictor — BEAST edition.

A cheap, single-purpose statistical language model: next-word suggestions and
text completion for plain prose. Pure Python standard library. No neural
networks, no GPU, no dependencies, trains in seconds.

The BEAST engine ("smoothing='beast'") combines, in one zero-dependency
package:

1. Modified Kneser-Ney smoothing
   Discount parameters (D1, D2, D3+) estimated from the corpus itself via the
   Chen-Goodman method of moments, with recursive backoff and continuation
   counts (the "Kneser-Ney base": a word is scored by how many DISTINCT
   contexts it continues, not by raw frequency).

2. Zipf-continuation prior  (novel)
   The backoff mass that Kneser-Ney would push down the context hierarchy is
   partially diverted straight to a Zipf-shaped prior built over
   *continuation ranks*:  p_zipf(w) ~ 1 / rank_c(w)^s  where rank_c ranks
   words by how many distinct contexts they continue in. This regularizes
   exactly the sparse corner where Kneser-Ney is weakest: words with thin
   continuation statistics.

3. Confidence-adaptive prior diversion  (novel)
   The diversion strength is not global. It is gated per context by an
   evidence-aware trust curve:
       z(ctx) = z_max * theta^tau / (theta^tau + c(ctx)^tau)
   so contexts with plenty of evidence trust plain Kneser-Ney, while thin or
   unseen contexts lean on the Zipf-continuation prior. z_max = 0 recovers
   exact Modified Kneser-Ney; adaptive_prior = False makes z constant.

4. Self-calibration
   tune() sweeps the discount scale, the prior ceiling and the trust curve on
   held-out text and applies the winner. The model finds its own smoothing.

Legacy mode ("smoothing='zipf'") reproduces the historical v4 behaviour
exactly: interpolated n-grams whose distributions blend empirical counts with
a rank-based Zipf prior, plus the personalization user layer.

The user layer (observe / forget / user_stats / save_user_profile) is
unchanged and available in every engine: it learns from whatever the user
types without touching the base model.

Drop-in upgrade: save() writes format version 5; load() still reads version 4
pickles and upgrades them in place to the BEAST engine with zero retraining
(the Kneser-Ney statistics are derived from the stored raw counts).
"""

import glob
import heapq
import math
import os
import pickle
import random
import re
from collections import defaultdict, Counter

MODEL_FORMAT_VERSION = 5

SMOOTHING_ENGINES = ("beast", "modified_kneser_ney", "kneser_ney", "zipf")


def _dd_int():
    """Module-level factory so models stay picklable (no local lambdas)."""
    return defaultdict(int)

_KN_ENGINES = ("beast", "modified_kneser_ney", "kneser_ney")


class ZipfNextWordPredictor:
    def __init__(self, corpus_texts=None, n_gram_size=3, zipf_exponent=1.0,
                 zipf_weight=0.7, interp_weights=None,
                 personalization=0.3, personalization_halflife=200.0,
                 smoothing="beast", discount_scale=1.0,
                 zipf_prior_weight=0.15, adaptive_prior=True,
                 prior_mode="zipf",
                 trust_halflife=2.0, trust_exponent=1.0):
        """
        Initialize the next word predictor.

        Args:
            corpus_texts: List of texts to build the model from
            n_gram_size: Size of n-grams to use (default: 3 for trigrams;
                the BEAST engine supports 5-grams comfortably)
            zipf_exponent: Exponent s for Zipf's law (default: 1.0)
            zipf_weight: [legacy engine + user layer] Blend factor between
                empirical counts and the Zipf rank prior.
                P = zipf_weight * P_empirical + (1 - zipf_weight) * P_zipf.
            interp_weights: [legacy engine + user layer] Interpolation weights
                per context length (index 0 = unigram level).
            personalization: Max weight (0-1) given to the user layer when
                blending user-learned vs corpus distributions.
            personalization_halflife: User words needed for the
                personalization weight to reach half its maximum.
            smoothing: Base engine:
                "beast"               -> Modified Kneser-Ney + Zipf-continuation
                                          prior + confidence-adaptive diversion
                "modified_kneser_ney" -> plain Modified Kneser-Ney
                "kneser_ney"          -> simple Kneser-Ney (one discount/order)
                "zipf"                -> legacy v4 interpolated Zipf blend
            discount_scale: Multiplier applied to the auto-estimated
                Kneser-Ney discounts (tunable; default 1.0 = as estimated).
            zipf_prior_weight: z_max — ceiling of the Zipf-continuation prior
                diversion for the beast engine (0 disables the prior).
            adaptive_prior: If True (default), the diversion strength follows
                the per-context trust curve; if False it is constant z_max.
            prior_mode: What regularizes the sparse-region backoff mass at
                the same junction (KN-family engines only):
                "zipf"        -> the novel Zipf-continuation prior (default;
                                  matches every pre-5.1 model exactly)
                "witten_bell" -> Witten-Bell's own confidence weighting:
                                  backoff weight lambda = D/(T+D) per context
                                  (T = context token count, D = distinct
                                  continuation types); parameter-free, no
                                  prior diversion. Comparator baseline for
                                  the cross-corpus study.
            trust_halflife: theta — context evidence (token mass) at which the
                adaptive diversion reaches half its ceiling.
            trust_exponent: tau — sharpness of the trust curve.
        """
        if n_gram_size < 2:
            raise ValueError("n_gram_size must be >= 2")
        if smoothing not in SMOOTHING_ENGINES:
            raise ValueError(
                f"Unknown smoothing engine '{smoothing}'. "
                f"Choose one of {SMOOTHING_ENGINES}"
            )

        self.n_gram_size = n_gram_size
        self.zipf_exponent = zipf_exponent
        self.zipf_weight = min(max(zipf_weight, 0.0), 1.0)
        self.interp_weights = (
            self._default_interp_weights() if interp_weights is None
            else self._normalize_weights(interp_weights)
        )
        self.personalization = min(max(personalization, 0.0), 1.0)
        self.personalization_halflife = max(float(personalization_halflife), 1.0)

        self.smoothing = smoothing
        self.discount_scale = max(float(discount_scale), 0.0)
        self.zipf_prior_weight = min(max(float(zipf_prior_weight), 0.0), 1.0)
        self.adaptive_prior = bool(adaptive_prior)
        if prior_mode not in ("zipf", "witten_bell"):
            raise ValueError(
                "Unknown prior_mode '%s'. Choose one of "
                "('zipf', 'witten_bell')" % prior_mode)
        self.prior_mode = prior_mode
        self.trust_halflife = max(float(trust_halflife), 1e-9)
        self.trust_exponent = max(float(trust_exponent), 1e-9)

        self._legacy_mode = (smoothing == "zipf")

        # Base (corpus) model components.
        self.total_words = 0
        if self._legacy_mode:
            self._ngram_legacy = defaultdict(_dd_int)
            self._corpus_freq_legacy = Counter()
            self._unigram_dist = {}
            self._unigram_top = None
            self._legacy_level_cache = {}
            self._legacy_top_cache = {}
            self._legacy_topk_cache = {}
        else:
            # id-interned backend:
            # self._counts[k][ctx_ids] = (counts_dict, total, n1, n2, n3p)
            self._w2i = {}
            self._i2w = []
            self._counts = [None] + [dict() for _ in range(n_gram_size - 1)]
            self._uni = {}
            self._ctx_stats = [None] + [dict() for _ in range(n_gram_size - 1)]
            self._cont = {}
            self._cont_total = 0
            self._pcont = {}
            self._pzipf = {}
            self._D = [None] + [(0.5, 1.0, 1.5)] * (n_gram_size - 1)
            self._coc = [None] + [(0, 0, 0, 0)] * (n_gram_size - 1)
            self._topk_pcont = []
            self._topk_pzipf = []

        # Lazily materialized legacy views of the id backend (compat only).
        self._legacy_view = None
        self._legacy_freq_view = None

        # Speed cache for the legacy interpolation path (results identical,
        # keyed by context + engine parameters; cleared on rebuild).
        self._base_dist_cache = {}

        # User layer: learns from observed text, never touches the base model
        self.user_n_gram_model = defaultdict(_dd_int)
        self.user_corpus_freq = Counter()
        self.user_total_words = 0
        self._user_unigram_dist = {}

        if corpus_texts:
            self.build_model(corpus_texts)

    # ------------------------------------------------------------------ #
    # Compat views: n_gram_model / corpus_freq behave exactly like v4
    # ------------------------------------------------------------------ #

    @property
    def n_gram_model(self):
        if self._legacy_mode:
            return self._ngram_legacy
        if self._legacy_view is None:
            self._legacy_view = self._materialize_legacy_view()
        return self._legacy_view

    @n_gram_model.setter
    def n_gram_model(self, value):
        self._ngram_legacy = value

    @property
    def corpus_freq(self):
        if self._legacy_mode:
            return self._corpus_freq_legacy
        if self._legacy_freq_view is None:
            self._legacy_freq_view = Counter(
                {self._i2w[w]: c for w, c in self._uni.items()}
            )
        return self._legacy_freq_view

    @corpus_freq.setter
    def corpus_freq(self, value):
        self._corpus_freq_legacy = value

    def _materialize_legacy_view(self):
        """Rebuild the v4-style {context_tuple(str): {word: count}} view."""
        view = defaultdict(dict)
        for k in range(1, len(self._counts)):
            for ctx, payload in self._counts[k].items():
                words = tuple(self._i2w[i] for i in ctx)
                view[words] = dict(payload[0])
        return dict(view)

    # ------------------------------------------------------------------ #
    # Tokenization
    # ------------------------------------------------------------------ #

    def simple_sent_tokenize(self, text):
        """Simple sentence tokenizer that doesn't rely on NLTK"""
        sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])$', text)
        return [s for s in sentences if s]

    def preprocess(self, text):
        """Convert text to lowercase and extract words while preserving sentence structure"""
        sentences = self.simple_sent_tokenize(text)
        processed_sentences = []

        for sentence in sentences:
            words = re.findall(r'\b\w+\b|[.,!?;]', sentence)
            if words:
                processed_sentences.append(words)

        return processed_sentences

    @staticmethod
    def _tokenize(text):
        return re.findall(r'\b\w+\b|[.,!?;]', text.lower())

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def build_model(self, corpus_texts):
        """Build the base model from corpus texts"""
        all_sentences = []
        for text in corpus_texts:
            all_sentences.extend(self.preprocess(text))
        self._build_from_sentences(all_sentences)

    def _build_from_sentences(self, sentences):
        self._base_dist_cache = {}
        self._legacy_view = None
        self._legacy_freq_view = None

        if self._legacy_mode:
            self._unigram_top = None
            self._legacy_level_cache = {}
            self._legacy_top_cache = {}
            self._legacy_topk_cache = {}
            self._ngram_legacy = defaultdict(_dd_int)
            self._corpus_freq_legacy = Counter()
            self._update_counts(sentences,
                                self._ngram_legacy, self._corpus_freq_legacy)
            self.total_words = sum(self._corpus_freq_legacy.values())
            self._recompute_unigram_dist()
        else:
            self._build_id_backend(sentences)

    def _build_id_backend(self, sentences):
        """Count n-grams with id-interned words and derive KN statistics."""
        w2i = self._w2i
        i2w = self._i2w
        n_max = self.n_gram_size - 1

        counts = [None] + [defaultdict(dict) for _ in range(n_max)]
        uni = defaultdict(int)

        for sentence in sentences:
            ids = []
            for w in sentence:
                wl = w.lower()
                wid = w2i.get(wl)
                if wid is None:
                    wid = len(i2w)
                    w2i[wl] = wid
                    i2w.append(wl)
                ids.append(wid)
            for wid in ids:
                uni[wid] += 1
            L = len(ids)
            for k in range(1, n_max + 1):
                ck = counts[k]
                for i in range(L - k):
                    ctx = tuple(ids[i:i + k])
                    t = ids[i + k]
                    d = ck[ctx]
                    d[t] = d.get(t, 0) + 1

        # Finalize: attach per-context stats (total, n1, n2, n3p) to each ctx.
        for k in range(1, n_max + 1):
            finalized = {}
            for ctx, d in counts[k].items():
                total = 0
                n1 = n2 = n3p = 0
                for c in d.values():
                    total += c
                    if c == 1:
                        n1 += 1
                    elif c == 2:
                        n2 += 1
                    else:
                        n3p += 1
                finalized[ctx] = (d, total, n1, n2, n3p)
            self._counts[k] = finalized

        self._uni = dict(uni)
        self.total_words = sum(self._uni.values())
        self._derive_kn_statistics()

    def _derive_kn_statistics(self):
        """Continuation counts, discounts, and the Zipf-continuation prior."""
        n_max = self.n_gram_size - 1

        # Continuation counts: number of DISTINCT left contexts per word
        # (bigram level) — the Kneser-Ney "continuation" evidence.
        cont = defaultdict(int)
        for d, total, _n1, _n2, _n3p in self._counts[1].values():
            for w in d:
                cont[w] += 1
        self._cont = dict(cont)
        self._cont_total = sum(cont.values())

        # Count-of-counts per order (for discount estimation).
        self._coc = [None] * (n_max + 1)
        for k in range(1, n_max + 1):
            n1 = n2 = n3 = n4 = 0
            for ctx, (d, _t, _a, _b, _c) in self._counts[k].items():
                for c in d.values():
                    if c == 1:
                        n1 += 1
                    elif c == 2:
                        n2 += 1
                    elif c == 3:
                        n3 += 1
                    else:
                        n4 += 1
            self._coc[k] = (n1, n2, n3, n4)

        self._recompute_discounts()
        self._build_prior_distributions()

    def _recompute_discounts(self):
        """
        Estimate Kneser-Ney discounts from the corpus count-of-counts
        (Chen & Goodman 1998, method of moments), scaled by discount_scale.

        Simple KN:      D = n1 / (n1 + 2*n2) per order.
        Modified KN:    Y = n1/(n1+2 n2);
                        D1 = 1 - 2 Y n2/n1; D2 = 2 - 3 Y n3/n2;
                        D3 = 3 - 4 Y n4/n3.
        """
        n_max = self.n_gram_size - 1
        s = self.discount_scale
        self._D = [None] * (n_max + 1)
        for k in range(1, n_max + 1):
            n1, n2, n3, n4 = self._coc[k]
            if self.smoothing == "kneser_ney":
                d = (n1 / (n1 + 2.0 * n2)) if (n1 + n2) > 0 else 0.5
                self._D[k] = (s * d, 0.0, 0.0)
                continue
            if n1 > 0 and n2 > 0 and n3 > 0 and n4 > 0:
                y = n1 / (n1 + 2.0 * n2)
                d1 = 1.0 - 2.0 * y * (n2 / n1)
                d2 = 2.0 - 3.0 * y * (n3 / n2)
                d3 = 3.0 - 4.0 * y * (n4 / n3)
            else:
                d1, d2, d3 = 0.5, 1.0, 1.5
            self._D[k] = (max(s * d1, 0.0), max(s * d2, 0.0), max(s * d3, 0.0))
        # The prior distributions do not depend on discounts; nothing else here.

    def _build_prior_distributions(self):
        """
        Build the two base distributions:

        p_cont: empirical Kneser-Ney continuation distribution
                P(w) = |{left contexts of w}| / |{left contexts}|
        p_zipf: Zipf prior over CONTINUATION ranks
                p_zipf(w) ~ 1 / rank_c(w)^s  (rank_c: rank by continuation
                count). This is the regularizing prior the beast engine
                diverts backoff mass to.
        """
        ct = self._cont_total
        if ct <= 0:
            self._pcont = {}
            self._pzipf = {}
            self._topk_pcont = []
            self._topk_pzipf = []
            return

        p_emp = {w: c / ct for w, c in self._cont.items()}

        ranked = sorted(self._cont.items(), key=lambda kv: (-kv[1], kv[0]))
        s_exp = self.zipf_exponent
        raw = {}
        for rank, (w, _c) in enumerate(ranked, start=1):
            raw[w] = 1.0 / (rank ** s_exp)
        zsum = sum(raw.values())
        p_zipf = {w: v / zsum for w, v in raw.items()}

        self._pcont = p_emp
        self._pzipf = p_zipf

        # Pre-sorted lists for exact lazy top-k merges (descending mass,
        # alphabetical tie-break — matches the final (-prob, word) sort).
        self._topk_pcont = sorted(p_emp.items(),
                                  key=lambda kv: (-kv[1], self._i2w[kv[0]]))
        self._topk_pzipf = sorted(p_zipf.items(),
                                  key=lambda kv: (-kv[1], self._i2w[kv[0]]))

    # ------------------------------------------------------------------ #
    # User layer (unchanged from v4 — works with every engine)
    # ------------------------------------------------------------------ #

    def observe(self, text):
        """
        Learn from user text without touching the base corpus model.
        Call repeatedly with whatever the user types; predictions adapt.

        Returns:
            Dict with user layer stats: words, vocab, effective alpha.
        """
        sentences = self.preprocess(text)
        self._update_counts(sentences,
                            self.user_n_gram_model, self.user_corpus_freq)
        self.user_total_words = sum(self.user_corpus_freq.values())
        self._recompute_user_unigram_dist()
        return self.user_stats()

    def forget(self):
        """Erase everything learned from the user (privacy reset)."""
        self.user_n_gram_model = defaultdict(_dd_int)
        self.user_corpus_freq = Counter()
        self.user_total_words = 0
        self._user_unigram_dist = {}

    def user_stats(self):
        """Current user layer size and effective personalization weight."""
        return {
            "words": self.user_total_words,
            "vocab": len(self.user_corpus_freq),
            "effective_alpha": self._effective_alpha(),
        }

    def _update_counts(self, sentences, n_gram_model, freq):
        for sentence in sentences:
            for word in sentence:
                freq[word.lower()] += 1
            for n in range(2, self.n_gram_size + 1):
                for i in range(len(sentence) - n + 1):
                    context = tuple(w.lower() for w in sentence[i:i + n - 1])
                    next_word = sentence[i + n - 1].lower()
                    n_gram_model[context][next_word] += 1

    def train_from_files(self, path):
        """
        Train the model from .txt file(s).

        Args:
            path: Path to a .txt file or a directory (searched recursively
                for *.txt files).

        Returns:
            Dict with training stats: files, words, vocab.
        """
        files = self._collect_text_files(path)
        texts = []
        for file_path in files:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                texts.append(f.read())

        self.build_model(texts)
        return {
            "files": files,
            "num_files": len(files),
            "words": self.total_words,
            "vocab": len(self.corpus_freq),
        }

    @staticmethod
    def _collect_text_files(path):
        if os.path.isfile(path):
            return [path]
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True))
            if not files:
                raise ValueError(f"No .txt files found under directory: {path}")
            return files
        raise ValueError(f"Path does not exist: {path}")

    # ------------------------------------------------------------------ #
    # Probability math (legacy engine): Zipf blend + interpolated smoothing
    # ------------------------------------------------------------------ #

    def _default_interp_weights(self):
        raw = [2.0 ** i for i in range(self.n_gram_size)]
        total = sum(raw)
        return [r / total for r in raw]

    def _normalize_weights(self, weights):
        weights = [float(w) for w in weights]
        if len(weights) != self.n_gram_size:
            raise ValueError(
                f"interp_weights must have length n_gram_size ({self.n_gram_size})"
            )
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("interp_weights must be non-negative and not all zero")
        total = sum(weights)
        return [w / total for w in weights]

    def zipf_probability(self, rank):
        """Calculate Zipf's probability for a given rank"""
        return 1.0 / (rank ** self.zipf_exponent)

    def _blended_distribution(self, counts):
        """
        Turn raw counts into a distribution blending empirical evidence with
        the Zipf rank prior:
            P(word) proportional to zipf_weight * (count/total)
                                 + (1 - zipf_weight) * (1/rank^s)
        """
        total = sum(counts.values())
        if not total:
            return {}

        ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        dist = {}
        for rank, (word, count) in enumerate(ranked, start=1):
            p_empirical = count / total
            p_zipf = self.zipf_probability(rank)
            dist[word] = self.zipf_weight * p_empirical + (1.0 - self.zipf_weight) * p_zipf

        norm = sum(dist.values())
        return {w: p / norm for w, p in dist.items()}

    def _recompute_unigram_dist(self):
        self._unigram_dist = self._blended_distribution(self._corpus_freq_legacy)
        self._unigram_top = None
        self._legacy_topk_cache = {}

    def _recompute_user_unigram_dist(self):
        self._user_unigram_dist = self._blended_distribution(self.user_corpus_freq)

    def _effective_alpha(self):
        """Personalization weight scaled by how much user evidence exists."""
        if self.personalization <= 0 or self.user_total_words == 0:
            return 0.0
        u = self.user_total_words
        h = self.personalization_halflife
        return self.personalization * (u / (u + h))

    def _base_interp_scores(self, context, cache_key):
        """Interpolated corpus-level scores for the legacy engine (cached)."""
        cached = self._base_dist_cache.get(cache_key)
        if cached is not None:
            return cached

        scores = defaultdict(float)
        for ctx_len, weight in enumerate(self.interp_weights):
            if weight <= 0:
                continue
            if ctx_len == 0:
                dist = self._unigram_dist
            elif ctx_len <= len(context):
                counts = self._ngram_legacy.get(context[-ctx_len:])
                if not counts:
                    continue
                dist = self._blended_distribution(counts)
            else:
                continue
            if not dist:
                continue
            for word, p in dist.items():
                scores[word] += weight * p

        result = dict(scores)
        if len(self._base_dist_cache) < 400000:
            self._base_dist_cache[cache_key] = result
        return result

    def _interp_scores(self, context, n_gram_model, unigram_dist):
        """Interpolated scores for one layer (used for the user layer)."""
        scores = defaultdict(float)

        for ctx_len, weight in enumerate(self.interp_weights):
            if weight <= 0:
                continue
            if ctx_len == 0:
                dist = unigram_dist
            elif ctx_len <= len(context):
                counts = n_gram_model.get(context[-ctx_len:])
                if not counts:
                    continue
                dist = self._blended_distribution(counts)
            else:
                continue
            if not dist:
                continue
            for word, p in dist.items():
                scores[word] += weight * p

        return dict(scores)

    def _blended_cached(self, ctx_len, ctx_suffix):
        """Cached Zipf-blended distribution for one (order, context).
        Keyed by zipf_weight: tune() sweeps the weight, so stale entries
        would silently corrupt the sweep."""
        key = (ctx_len, ctx_suffix, self.zipf_weight)
        d = self._legacy_level_cache.get(key)
        if d is None:
            counts = self._ngram_legacy.get(ctx_suffix)
            d = self._blended_distribution(counts) if counts else {}
            if len(self._legacy_level_cache) < 800000:
                self._legacy_level_cache[key] = d
        return d

    def _legacy_levels(self, context):
        """Per-level (weight, blended distribution), cached (legacy engine).
        Each level's distribution sums to 1, so interpolated scores over any
        candidate set are already normalized."""
        out = []
        for ctx_len, weight in enumerate(self.interp_weights):
            if weight <= 0:
                continue
            if ctx_len == 0:
                out.append((weight, self._unigram_dist))
            elif ctx_len <= len(context):
                d = self._blended_cached(ctx_len, context[-ctx_len:])
                if d:
                    out.append((weight, d))
        return out

    def _legacy_p(self, context, target):
        """Legacy-engine probability of one target word (user layer blended
        when active). Renormalizes by the weights of the levels that actually
        contributed mass — exactly what _score_distribution + evaluate()'s
        norm division produce."""
        alpha = self._effective_alpha()
        p = 0.0
        w_sum = 0.0
        for weight, d in self._legacy_levels(context):
            p += weight * d.get(target, 0.0)
            w_sum += weight
        if alpha <= 0:
            return p / w_sum if w_sum > 0 else 0.0
        up = 0.0
        uw_sum = 0.0
        for ctx_len, weight in enumerate(self.interp_weights):
            if weight <= 0:
                continue
            if ctx_len == 0:
                up += weight * self._user_unigram_dist.get(target, 0.0)
                uw_sum += weight
            elif ctx_len <= len(context):
                counts = self.user_n_gram_model.get(context[-ctx_len:])
                if not counts:
                    continue
                up += weight * self._blended_distribution(counts).get(target, 0.0)
                uw_sum += weight
        num = (1.0 - alpha) * p + alpha * up
        den = (1.0 - alpha) * w_sum + alpha * uw_sum
        return num / den if den > 0 else 0.0

    def _legacy_top_k(self, context, k):
        """Exact-on-small-vocab, pooled top-k for the legacy engine (the
        per-level candidate pool mirrors the classical baselines' protocol).
        The user layer's vocabulary is always included in full."""
        key = (tuple(context), k, self.zipf_weight, tuple(self.interp_weights),
               self._effective_alpha())
        hit = self._legacy_topk_cache.get(key)
        if hit is not None:
            return hit
        alpha = self._effective_alpha()
        pool = {}
        for ctx_len, weight in enumerate(self.interp_weights):
            if weight <= 0:
                continue
            if ctx_len == 0:
                if self._unigram_top is None:
                    self._unigram_top = sorted(
                        self._unigram_dist.items(),
                        key=lambda kv: (-kv[1], kv[0]))[:80]
                items = self._unigram_top
            elif ctx_len <= len(context):
                counts = self._ngram_legacy.get(context[-ctx_len:])
                if not counts:
                    continue
                top_key = (ctx_len, context[-ctx_len:], self.zipf_weight)
                items = self._legacy_top_cache.get(top_key)
                if items is None:
                    # blend value is monotone in (count desc, word asc), so
                    # the top-80 by raw count equal the top-80 by blend value
                    if len(counts) > 80:
                        d = self._blended_cached(ctx_len, context[-ctx_len:])
                        top_counts = heapq.nsmallest(
                            80, counts.items(), key=lambda kv: (-kv[1], kv[0]))
                        items = [(w, d[w]) for w, _ in top_counts]
                    else:
                        items = list(self._blended_cached(
                            ctx_len, context[-ctx_len:]).items())
                    if len(self._legacy_top_cache) < 800000:
                        self._legacy_top_cache[top_key] = items
            else:
                continue
            for w, v in items:
                pool[w] = pool.get(w, 0.0) + weight * v
        if alpha > 0:
            uwords = set(self._user_unigram_dist)
            for ctx_len, weight in enumerate(self.interp_weights):
                if weight <= 0:
                    continue
                if ctx_len == 0:
                    for w in uwords:
                        pool[w] = pool.get(w, 0.0)
                elif ctx_len <= len(context):
                    counts = self.user_n_gram_model.get(context[-ctx_len:])
                    if counts:
                        for w in self._blended_distribution(counts):
                            pool.setdefault(w, pool.get(w, 0.0))
            for w in list(pool):
                uw = 0.0
                if w in self._user_unigram_dist:
                    uw += self.interp_weights[0] * self._user_unigram_dist[w]
                for ctx_len, weight in enumerate(self.interp_weights):
                    if weight <= 0 or ctx_len == 0 or ctx_len > len(context):
                        continue
                    counts = self.user_n_gram_model.get(context[-ctx_len:])
                    if counts:
                        uw += weight * self._blended_distribution(counts).get(w, 0.0)
                base = pool[w]
                pool[w] = (1.0 - alpha) * base + alpha * uw
        out = [w for w, _ in heapq.nsmallest(k, pool.items(),
                                             key=lambda kv: (-kv[1], kv[0]))]
        if len(self._legacy_topk_cache) < 300000:
            self._legacy_topk_cache[key] = out
        return out

    def _score_distribution(self, context):
        """
        Final distribution for the legacy engine: interpolated corpus scores
        blended with the user layer scores using the adaptive alpha.
        """
        context = tuple(context)
        cache_key = (context, self.zipf_weight, tuple(self.interp_weights))
        base = self._base_interp_scores(context, cache_key)

        alpha = self._effective_alpha()
        if alpha <= 0:
            return base

        user = self._interp_scores(context, self.user_n_gram_model,
                                   self._user_unigram_dist)
        if not user:
            return base

        blended = {}
        for word in set(base) | set(user):
            blended[word] = (1.0 - alpha) * base.get(word, 0.0) + alpha * user.get(word, 0.0)
        return blended

    @staticmethod
    def _apply_temperature(dist, temperature):
        if temperature == 1.0 or not dist:
            return dict(dist)
        t = max(float(temperature), 1e-6)
        adjusted = {w: p ** (1.0 / t) for w, p in dist.items()}
        norm = sum(adjusted.values())
        if norm <= 0:
            return dict(dist)
        return {w: p / norm for w, p in adjusted.items()}

    # ------------------------------------------------------------------ #
    # Probability math (BEAST engine): Kneser-Ney + Zipf-continuation prior
    # ------------------------------------------------------------------ #

    def _context_ids(self, context_words):
        """Map context words to ids, trimmed to the longest stored order."""
        n_max = self.n_gram_size - 1
        ids = []
        for w in context_words:
            wid = self._w2i.get(w)
            if wid is not None:
                ids.append(wid)
        return tuple(ids[-n_max:]) if n_max else ()

    def _trust(self, evidence_mass):
        """
        Confidence-adaptive prior strength for one context:
            z = z_max * theta^tau / (theta^tau + c^tau)
        Thin contexts -> z ~ z_max (lean on the Zipf prior);
        fat contexts  -> z ~ 0    (trust plain Kneser-Ney).
        """
        if not self.adaptive_prior:
            return self.zipf_prior_weight
        if self.zipf_prior_weight <= 0.0:
            return 0.0
        theta_t = self.trust_halflife ** self.trust_exponent
        c_t = evidence_mass ** self.trust_exponent
        return self.zipf_prior_weight * theta_t / (theta_t + c_t)

    def _bucket_discount(self, count, order):
        d1, d2, d3 = self._D[order]
        if count <= 1:
            return d1
        if count == 2:
            return d2
        return d3

    def _p_kn(self, wid, ctx):
        """
        Probability of word `wid` after context `ctx` (tuple of word ids,
        length <= n_gram_size-1) under the Kneser-Ney family, with the beast
        prior diversion:
            P(w|h) = p_hi(w|h)
                     + (1 - z) * lambda(h) * P(w|h[:-1])
                     + z * lambda(h) * p_zipf(w)
        """
        k = len(ctx)
        if k == 0:
            return self._pcont.get(wid, 0.0)

        payload = self._counts[k].get(ctx)
        if payload is None:
            return self._p_kn(wid, ctx[1:])

        d, total, n1, n2, n3p = payload
        c = d.get(wid, 0)
        p_hi = 0.0
        if c > 0:
            dc = self._bucket_discount(c, k)
            if c > dc:
                p_hi = (c - dc) / total
        d1, d2, d3 = self._D[k]
        lam = (d1 * n1 + d2 * n2 + d3 * n3p) / total
        if getattr(self, "prior_mode", "zipf") == "witten_bell":
            # Textbook Witten-Bell (Witten & Bell 1994; Manning & Schutze
            # 6.2.2), in the same junction the KN path interpolates:
            #   P(w|h) = c(w,h) / (T + D)  +  D/(T+D) * P(w|h')
            # (T = context token count, D = distinct continuation types).
            # Seen words are rescaled by T/(T+D); the D/(T+D) reserved mass
            # goes to the lower-order model. Parameter-free, no discounts,
            # no prior diversion; sums to 1 by construction.
            n_types = len(d)
            denom = total + n_types
            return (c / denom) + (n_types / denom) * self._p_kn(wid, ctx[1:])
        if lam <= 0.0:
            return p_hi

        p_bo = self._p_kn(wid, ctx[1:])
        z = self._trust(total)
        if z <= 0.0:
            return p_hi + lam * p_bo
        return p_hi + (1.0 - z) * lam * p_bo + z * lam * self._pzipf.get(wid, 0.0)

    _OBSERVED_POOL = 64  # per-level candidate cap (matches baseline pools)

    def _walk_levels(self, ctx):
        """
        Walk the context hierarchy once, returning:
          observed: dict wid -> accumulated highest-order contributions
          A: coefficient of the empirical continuation base p_cont
          B: coefficient of the Zipf-continuation prior p_zipf
        such that for any word w in the candidate pool:
            P(w|ctx) = observed.get(w, 0) + A * p_cont(w) + B * p_zipf(w)

        Per level, at most _OBSERVED_POOL words (top by count) contribute
        explicit highest-order terms; deeper tail words are still ranked by
        their exact base terms, mirroring the candidate-pool protocol used by
        the classical baselines. Log-probabilities (_p_kn) remain exact for
        every vocabulary word.
        """
        observed = {}
        A = 1.0
        B = 0.0
        scale = 1.0

        for k in range(len(ctx), 0, -1):
            if k >= len(self._counts):
                continue
            payload = self._counts[k].get(ctx[-k:])
            if payload is None:
                continue  # unseen at this order: level is transparent
            d, total, n1, n2, n3p = payload
            d1, d2, d3 = self._D[k]
            lam = (d1 * n1 + d2 * n2 + d3 * n3p) / total
            # highest-order contributions from this level (capped pool)
            if len(d) > self._OBSERVED_POOL:
                items = heapq.nsmallest(self._OBSERVED_POOL, d.items(),
                                        key=lambda kv: (-kv[1], kv[0]))
            else:
                items = d.items()
            if getattr(self, "prior_mode", "zipf") == "witten_bell":
                # Textbook Witten-Bell: seen words get c/(T+D); the D/(T+D)
                # reserved mass passes to the next-lower level. B stays 0
                # (no prior target in the top-k merge).
                n_types = len(d)
                denom = total + n_types
                for wid, c in items:
                    if c > 0:
                        observed[wid] = observed.get(wid, 0.0) + \
                            scale * c / denom
                scale *= n_types / denom
            else:
                for wid, c in items:
                    if c > 0:
                        dc = self._bucket_discount(c, k)
                        if c > dc:
                            observed[wid] = observed.get(wid, 0.0) + \
                                scale * (c - dc) / total
                z = self._trust(total)
                if z > 0.0:
                    B += scale * z * lam
                    scale *= (1.0 - z) * lam
                else:
                    scale *= lam
            if scale <= 0.0:
                break

        A = scale
        return observed, A, B

    def _top_unobserved(self, A, B, exclude, k):
        """
        Exact top-k words NOT in `exclude`, ranked by A*p_cont + B*p_zipf.

        Pruned parallel scan over the two pre-sorted lists (descending mass,
        alphabetical ties). Any unexplored word is bounded by
        A*p_frontier + B*z_frontier, so scanning stops once that bound falls
        below the current k-th best score. Replacement keeps the canonical
        (-score, word) order so results match brute-force ranking exactly.
        """
        if A + B <= 0.0 or k <= 0:
            return []
        lp = self._topk_pcont
        lz = self._topk_pzipf
        n_p, n_z = len(lp), len(lz)
        i = j = 0
        best = []       # sorted by (-score, word); worst item at best[-1]
        seen = set()

        while True:
            if best and len(best) >= k:
                s0 = best[-1][0]
                bound = (A * lp[i][1] if i < n_p else 0.0) + \
                        (B * lz[j][1] if j < n_z else 0.0)
                if bound < s0:
                    break
            if i >= n_p and j >= n_z:
                break
            if i < n_p:
                wid, p = lp[i]
                i += 1
                if wid not in exclude and wid not in seen:
                    seen.add(wid)
                    score = A * p + B * self._pzipf.get(wid, 0.0)
                    word = self._i2w[wid]
                    if len(best) < k:
                        best.append((score, wid, word))
                        if len(best) == k:
                            best.sort(key=lambda t: (-t[0], t[2]))
                    elif (-score, word) < (-best[-1][0], best[-1][2]):
                        best[-1] = (score, wid, word)
                        best.sort(key=lambda t: (-t[0], t[2]))
            if j < n_z:
                wid, zq = lz[j]
                j += 1
                if wid not in exclude and wid not in seen:
                    seen.add(wid)
                    score = A * self._pcont.get(wid, 0.0) + B * zq
                    word = self._i2w[wid]
                    if len(best) < k:
                        best.append((score, wid, word))
                        if len(best) == k:
                            best.sort(key=lambda t: (-t[0], t[2]))
                    elif (-score, word) < (-best[-1][0], best[-1][2]):
                        best[-1] = (score, wid, word)
                        best.sort(key=lambda t: (-t[0], t[2]))
        return [(score, wid) for score, wid, _word in best]

    def _topk_kn(self, ctx_words, k):
        """
        Exact top-k next-word distribution for the Kneser-Ney family,
        including the user-layer blend when personalization is active.

        Returns a list of (word, probability) sorted by decreasing probability.
        """
        ctx = self._context_ids(ctx_words)
        observed, A, B = self._walk_levels(ctx)

        alpha = self._effective_alpha()
        user_dist = None
        if alpha > 0:
            user_dist = self._interp_scores(
                tuple(w.lower() for w in ctx_words),
                self.user_n_gram_model, self._user_unigram_dist)

        # Candidate pool: observed words + exact top unobserved + user picks.
        # (merge-scan scores are re-derived below; only membership matters here)
        pool = dict(observed)
        for _score, wid in self._top_unobserved(A, B, pool, k):
            pool.setdefault(wid, 0.0)
        if user_dist:
            for w in sorted(user_dist, key=user_dist.get, reverse=True)[:k]:
                uwid = self._w2i.get(w)
                if uwid is not None and uwid not in pool:
                    pool[uwid] = 0.0

        cands = []
        for wid, obs_p in pool.items():
            p = obs_p + A * self._pcont.get(wid, 0.0) + B * self._pzipf.get(wid, 0.0)
            w = self._i2w[wid]
            if user_dist:
                p = (1.0 - alpha) * p + alpha * user_dist.get(w, 0.0)
            cands.append((w, p))
        if user_dist:
            # user-layer favourites outside the base vocabulary are legitimate
            # suggestions (v4 behaviour): add their blended mass directly
            added = 0
            for w in sorted(user_dist, key=user_dist.get, reverse=True):
                if w in self._w2i:
                    continue
                cands.append((w, alpha * user_dist[w]))
                added += 1
                if added >= k:
                    break
        cands.sort(key=lambda x: (-x[1], x[0]))
        return cands[:k]

    def _score_token_kn(self, ctx_words, target, k):
        """
        Score one token for evaluation: exact probability of `target` and the
        exact top-k candidate list, in a single pass over the context walk.

        Returns (probability, top_k_words).
        """
        ctx = self._context_ids(ctx_words)
        wid = self._w2i.get(target)
        observed, A, B = self._walk_levels(ctx)

        alpha = self._effective_alpha()
        user_dist = None
        if alpha > 0:
            user_dist = self._interp_scores(
                tuple(w.lower() for w in ctx_words),
                self.user_n_gram_model, self._user_unigram_dist)

        if wid is not None:
            p = observed.get(wid, 0.0) + A * self._pcont.get(wid, 0.0) \
                + B * self._pzipf.get(wid, 0.0)
        else:
            p = 0.0
        if user_dist:
            p = (1.0 - alpha) * p + alpha * user_dist.get(target, 0.0)

        pool = dict(observed)
        for _score, bwid in self._top_unobserved(A, B, pool, k):
            pool.setdefault(bwid, 0.0)
        if user_dist:
            for w in sorted(user_dist, key=user_dist.get, reverse=True)[:k]:
                uwid = self._w2i.get(w)
                if uwid is not None and uwid not in pool:
                    pool[uwid] = 0.0

        cands = []
        for wid2, obs_p in pool.items():
            pp = obs_p + A * self._pcont.get(wid2, 0.0) + B * self._pzipf.get(wid2, 0.0)
            w2 = self._i2w[wid2]
            if user_dist:
                pp = (1.0 - alpha) * pp + alpha * user_dist.get(w2, 0.0)
            cands.append((w2, pp))
        if user_dist:
            added = 0
            for w in sorted(user_dist, key=user_dist.get, reverse=True):
                if w in self._w2i:
                    continue
                cands.append((w, alpha * user_dist[w]))
                added += 1
                if added >= k:
                    break
        cands.sort(key=lambda x: (-x[1], x[0]))
        return p, [word for word, _pp in cands[:k]]

    def _kn_full_topk_exact(self, ctx_words, k):
        """Brute-force exact top-k over the full vocabulary (for tests)."""
        ctx = self._context_ids(ctx_words)
        alpha = self._effective_alpha()
        user_dist = None
        if alpha > 0:
            user_dist = self._interp_scores(
                tuple(w.lower() for w in ctx_words),
                self.user_n_gram_model, self._user_unigram_dist)
        cands = []
        for wid in range(len(self._i2w)):
            p = self._p_kn(wid, ctx)
            if user_dist:
                p = (1.0 - alpha) * p + alpha * user_dist.get(self._i2w[wid], 0.0)
            cands.append((self._i2w[wid], p))
        cands.sort(key=lambda x: (-x[1], x[0]))
        return cands[:k]

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict_next_words(self, context, num_predictions=5, temperature=1.0):
        """
        Predict multiple possible next words based on context.

        Args:
            context: String or list/tuple of preceding words
            num_predictions: Number of predictions to return
            temperature: Controls randomness (higher = more random)

        Returns:
            List of (word, probability) tuples sorted by decreasing probability
        """
        if isinstance(context, str):
            context = self._tokenize(context)
        else:
            context = [str(w).lower() for w in context]

        max_ctx = self.n_gram_size - 1
        context = tuple(context[-max_ctx:]) if max_ctx else ()

        if self._legacy_mode:
            scores = self._score_distribution(context)
            if not scores:
                return []
            dist = self._apply_temperature(scores, temperature)
            ranked = sorted(dist.items(), key=lambda x: (-x[1], x[0]))
            return ranked[:num_predictions]

        ranked = self._topk_kn(list(context), max(int(num_predictions), 1))
        if not ranked:
            return []
        dist = self._apply_temperature(dict(ranked), temperature)
        ranked = sorted(dist.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:num_predictions]

    def predict_completion(self, input_text, num_words=3, temperature=1.2):
        """
        Complete the given input text with a specified number of words

        Args:
            input_text: The text to complete
            num_words: Number of words to generate
            temperature: Controls randomness (higher = more random)

        Returns:
            Original text with completion appended
        """
        words = re.findall(r'\b\w+\b|[.,!?;]', input_text)
        result = input_text
        max_ctx = self.n_gram_size - 1

        for _ in range(num_words):
            context = words[-max_ctx:] if max_ctx else ()
            predictions = self.predict_next_words(context, num_predictions=10,
                                                  temperature=temperature)
            if not predictions:
                break

            words_list, probs_list = zip(*predictions)
            next_word = random.choices(words_list, weights=probs_list, k=1)[0]

            if next_word in ".,!?;":
                result += next_word
            else:
                if result and not result[-1].isspace():
                    result += " "
                result += next_word

            words.append(next_word)

        return result

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(self, text, k=3):
        """
        Evaluate the model on held-out text.

        Args:
            text: Held-out text to score
            k: Top-k for accuracy measurement

        Returns:
            Dict with top_k_accuracy, perplexity, tokens
        """
        hits = 0
        tokens = 0
        log_prob_sum = 0.0

        if self._legacy_mode:
            for sentence in self.preprocess(text):
                words = [w.lower() for w in sentence]
                for i in range(1, len(words)):
                    context = tuple(words[max(0, i - (self.n_gram_size - 1)):i])
                    target = words[i]

                    p = self._legacy_p(context, target)
                    if p > 0.0:
                        if target in self._legacy_top_k(context, k):
                            hits += 1
                    log_prob_sum += math.log(max(p, 1e-12))
                    tokens += 1
        else:
            n_ctx = self.n_gram_size - 1
            alpha = self._effective_alpha()
            use_cache = alpha <= 0  # cached top-k valid only without user layer
            ctx_cache = {}
            for sentence in self.preprocess(text):
                words = [w.lower() for w in sentence]
                for i in range(1, len(words)):
                    ctx_words = words[max(0, i - n_ctx):i]
                    target = words[i]

                    if use_cache:
                        wid = self._w2i.get(target)
                        if wid is not None:
                            ctx = self._context_ids(ctx_words)
                            p = self._p_kn(wid, ctx)
                        else:
                            p = 0.0
                        key = tuple(ctx_words)
                        cached = ctx_cache.get(key)
                        if cached is None:
                            cached = self._eval_topk_cached(key, k)
                            ctx_cache[key] = cached
                        if target in cached:
                            hits += 1
                    else:
                        p, topk_words = self._score_token_kn(ctx_words, target, k)
                        if target in topk_words:
                            hits += 1
                    log_prob_sum += math.log(max(p, 1e-12))
                    tokens += 1

        if tokens == 0:
            return {"top_k_accuracy": 0.0, "perplexity": float("inf"), "tokens": 0}

        return {
            "top_k_accuracy": hits / tokens,
            "perplexity": math.exp(-log_prob_sum / tokens),
            "tokens": tokens,
        }

    def _eval_topk_cached(self, ctx_tuple, k):
        """Exact top-k word list for one context (no user layer). Used by
        evaluate() with a per-call cache; results identical to _topk_kn."""
        observed, A, B = self._walk_levels(self._context_ids(list(ctx_tuple)))
        pool = dict(observed)
        for _score, bwid in self._top_unobserved(A, B, pool, k):
            pool.setdefault(bwid, 0.0)
        cands = []
        for wid2, obs_p in pool.items():
            pp = obs_p + A * self._pcont.get(wid2, 0.0) \
                + B * self._pzipf.get(wid2, 0.0)
            cands.append((self._i2w[wid2], pp))
        cands.sort(key=lambda x: (-x[1], x[0]))
        return [word for word, _pp in cands[:k]]

    def sentence_logprob(self, text):
        """
        Fast log-probability scoring of held-out text (base engine only,
        personalization included when active). Additive API.

        Returns:
            Dict with log_probability (natural log), tokens, perplexity.
        """
        log_prob_sum = 0.0
        tokens = 0

        if self._legacy_mode:
            for sentence in self.preprocess(text):
                words = [w.lower() for w in sentence]
                for i in range(1, len(words)):
                    context = tuple(words[max(0, i - (self.n_gram_size - 1)):i])
                    scores = self._score_distribution(context)
                    p = 0.0
                    if scores:
                        norm = sum(scores.values())
                        if norm > 0:
                            p = scores.get(words[i], 0.0) / norm
                    log_prob_sum += math.log(max(p, 1e-12))
                    tokens += 1
        else:
            n_ctx = self.n_gram_size - 1
            for sentence in self.preprocess(text):
                words = [w.lower() for w in sentence]
                for i in range(1, len(words)):
                    ctx_words = words[max(0, i - n_ctx):i]
                    wid = self._w2i.get(words[i])
                    if wid is None:
                        p = 0.0
                    else:
                        ctx = self._context_ids(ctx_words)
                        p = self._p_kn(wid, ctx)
                    log_prob_sum += math.log(max(p, 1e-12))
                    tokens += 1

        if tokens == 0:
            return {"log_probability": 0.0, "tokens": 0, "perplexity": float("inf")}
        return {
            "log_probability": log_prob_sum,
            "tokens": tokens,
            "perplexity": math.exp(-log_prob_sum / tokens),
        }

    # ------------------------------------------------------------------ #
    # Self-tuning
    # ------------------------------------------------------------------ #

    def _interp_profiles(self):
        """Named interpolation weight profiles, index 0 = unigram level."""
        n = self.n_gram_size
        return {
            "long_context": [2.0 ** i for i in range(n)],
            "aggressive_long": [4.0 ** i for i in range(n)],
            "balanced": [float(i + 1) for i in range(n)],
            "uniform": [1.0] * n,
        }

    def tune(self, held_out_text, zipf_weights=None, interp_profiles=None,
             discount_scales=None, zipf_prior_weights=None,
             adaptive_options=None, max_tokens=None):
        """
        Sweep hyperparameters on held-out text and apply the best combination
        (lowest perplexity) to this model.

        Legacy engine ("zipf"): sweeps 11 zipf_weight values across 4
        interpolation profiles, exactly like v4.

        Kneser-Ney family engines: sweeps the discount scale, the
        Zipf-continuation prior ceiling and the adaptive-prior switch —
        the model calibrates its own smoothing.

        Args:
            held_out_text: Text NOT used for training, used to score settings
            zipf_weights: [legacy] grid of zipf_weight values
            interp_profiles: [legacy] interpolation profiles to try
            discount_scales: [KN family] grid of discount scale multipliers
            zipf_prior_weights: [KN family] grid of prior ceilings z_max
            adaptive_options: [KN family] list of adaptive_prior booleans
            max_tokens: Optional cap on held-out tokens per trial (speed)

        Returns:
            Dict with "best", "trials", "trials_run".
        """
        if self._legacy_mode:
            return self._tune_legacy(held_out_text, zipf_weights, interp_profiles)

        if discount_scales is None:
            discount_scales = [0.7, 0.85, 1.0, 1.15, 1.3]
        if zipf_prior_weights is None:
            zipf_prior_weights = [0.0, 0.05, 0.1, 0.15, 0.25, 0.4]
        if adaptive_options is None:
            adaptive_options = [True, False]

        if self.smoothing != "beast" or \
                getattr(self, "prior_mode", "zipf") == "witten_bell":
            # Plain KN engines: the prior is not part of their identity.
            # Witten-Bell mode: the weight is parameter-free by design.
            zipf_prior_weights = [0.0]
            adaptive_options = [False]

        original = (self.discount_scale, self.zipf_prior_weight,
                    self.adaptive_prior)

        score_text = held_out_text
        if max_tokens is not None:
            sentences = self.preprocess(held_out_text)
            budget = max_tokens
            kept = []
            for s in sentences:
                if budget <= 0:
                    break
                kept.append(s)
                budget -= len(s)
            score_text = " ".join(" ".join(s) for s in kept)

        best = None
        trials = []
        try:
            for adaptive in adaptive_options:
                self.adaptive_prior = bool(adaptive)
                for ds in discount_scales:
                    self.discount_scale = max(float(ds), 0.0)
                    self._recompute_discounts()
                    for zw in zipf_prior_weights:
                        self.zipf_prior_weight = min(max(float(zw), 0.0), 1.0)
                        result = self.evaluate(score_text, k=3)
                        trial = {
                            "smoothing": self.smoothing,
                            "prior_mode": getattr(self, "prior_mode", "zipf"),
                            "discount_scale": self.discount_scale,
                            "zipf_prior_weight": self.zipf_prior_weight,
                            "adaptive_prior": self.adaptive_prior,
                            "interp_profile": "kneser_ney",
                            "perplexity": result["perplexity"],
                            "top_k_accuracy": result["top_k_accuracy"],
                        }
                        trials.append(trial)
                        if best is None or trial["perplexity"] < best["perplexity"]:
                            best = trial
        finally:
            if best is not None:
                self.discount_scale = best["discount_scale"]
                self.zipf_prior_weight = best["zipf_prior_weight"]
                self.adaptive_prior = best["adaptive_prior"]
            else:
                (self.discount_scale, self.zipf_prior_weight,
                 self.adaptive_prior) = original
            self._recompute_discounts()

        return {"best": best, "trials": trials, "trials_run": len(trials)}

    def _tune_legacy(self, held_out_text, zipf_weights, interp_profiles):
        """Original v4 sweep: zipf_weight grid x interpolation profiles."""
        if zipf_weights is None:
            zipf_weights = [round(i / 10, 1) for i in range(11)]
        if interp_profiles is None:
            interp_profiles = list(self._interp_profiles().keys())

        profiles = self._interp_profiles()
        unknown = [p for p in interp_profiles if p not in profiles]
        if unknown:
            raise ValueError(f"Unknown interp profiles: {unknown}")

        original = (self.zipf_weight, self.interp_weights)
        best = None
        trials = []

        try:
            for profile_name in interp_profiles:
                self.interp_weights = self._normalize_weights(profiles[profile_name])
                for zw in zipf_weights:
                    self.zipf_weight = min(max(float(zw), 0.0), 1.0)
                    self._recompute_unigram_dist()
                    result = self.evaluate(held_out_text, k=3)
                    trial = {
                        "zipf_weight": self.zipf_weight,
                        "interp_profile": profile_name,
                        "perplexity": result["perplexity"],
                        "top_k_accuracy": result["top_k_accuracy"],
                    }
                    trials.append(trial)
                    if best is None or trial["perplexity"] < best["perplexity"]:
                        best = trial
        finally:
            if best is not None:
                self.zipf_weight = best["zipf_weight"]
                self.interp_weights = self._normalize_weights(
                    profiles[best["interp_profile"]])
            else:
                self.zipf_weight, self.interp_weights = original
            self._recompute_unigram_dist()

        return {"best": best, "trials": trials, "trials_run": len(trials)}

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path, compress=False):
        """
        Serialize the trained model (including the user layer) to a file.
        KN-family engines write the id-interned backend; the legacy engine
        writes the v4-style structures. Set compress=True for a
        zlib-compressed container (roughly 3-5x smaller).
        """
        if self._legacy_mode:
            payload = {
                "format_version": MODEL_FORMAT_VERSION,
                "engine": "zipf",
                "params": {
                    "n_gram_size": self.n_gram_size,
                    "zipf_exponent": self.zipf_exponent,
                    "zipf_weight": self.zipf_weight,
                    "interp_weights": self.interp_weights,
                    "personalization": self.personalization,
                    "personalization_halflife": self.personalization_halflife,
                    "smoothing": self.smoothing,
                },
                "n_gram_model": {ctx: dict(counts) for ctx, counts
                                 in self._ngram_legacy.items()},
                "corpus_freq": dict(self._corpus_freq_legacy),
                "total_words": self.total_words,
                "user_n_gram_model": {ctx: dict(counts) for ctx, counts
                                      in self.user_n_gram_model.items()},
                "user_corpus_freq": dict(self.user_corpus_freq),
            }
        else:
            payload = {
                "format_version": MODEL_FORMAT_VERSION,
                "engine": self.smoothing,
                "params": {
                    "n_gram_size": self.n_gram_size,
                    "zipf_exponent": self.zipf_exponent,
                    "zipf_weight": self.zipf_weight,
                    "interp_weights": self.interp_weights,
                    "personalization": self.personalization,
                    "personalization_halflife": self.personalization_halflife,
                    "smoothing": self.smoothing,
                    "discount_scale": self.discount_scale,
                    "zipf_prior_weight": self.zipf_prior_weight,
                    "adaptive_prior": self.adaptive_prior,
                    "prior_mode": getattr(self, "prior_mode", "zipf"),
                    "trust_halflife": self.trust_halflife,
                    "trust_exponent": self.trust_exponent,
                },
                "vocab": list(self._i2w),
                "counts": {
                    k: {ctx: (dict(d), total, n1, n2, n3p)
                        for ctx, (d, total, n1, n2, n3p) in self._counts[k].items()}
                    for k in range(1, len(self._counts))
                },
                "uni": dict(self._uni),
                "coc": list(self._coc),
                "D": list(self._D),
                "cont": dict(self._cont),
                "cont_total": self._cont_total,
                "total_words": self.total_words,
                "user_n_gram_model": {ctx: dict(counts) for ctx, counts
                                      in self.user_n_gram_model.items()},
                "user_corpus_freq": dict(self.user_corpus_freq),
            }

        if compress:
            import zlib
            raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            with open(path, "wb") as f:
                f.write(b"ZFCM")
                f.write(zlib.compress(raw, level=6))
        else:
            with open(path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_user_profile(self, path):
        """
        Serialize just the user layer to a standalone profile file,
        so personal data can be stored, moved, or deleted separately.
        """
        payload = {
            "profile_version": 1,
            "n_gram_size": self.n_gram_size,
            "user_n_gram_model": {ctx: dict(counts) for ctx, counts
                                  in self.user_n_gram_model.items()},
            "user_corpus_freq": dict(self.user_corpus_freq),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_user_profile(self, path):
        """Replace the user layer with one saved via save_user_profile()"""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        if not isinstance(payload, dict) or "profile_version" not in payload:
            raise ValueError(f"Not a valid user profile file: {path}")
        if payload.get("n_gram_size") != self.n_gram_size:
            raise ValueError(
                f"Profile n_gram_size ({payload.get('n_gram_size')}) does not "
                f"match model ({self.n_gram_size})"
            )

        self.user_n_gram_model = defaultdict(
            lambda: defaultdict(int),
            {ctx: defaultdict(int, counts)
             for ctx, counts in payload["user_n_gram_model"].items()},
        )
        self.user_corpus_freq = Counter(payload["user_corpus_freq"])
        self.user_total_words = sum(self.user_corpus_freq.values())
        self._recompute_user_unigram_dist()
        return self.user_stats()

    @classmethod
    def load(cls, path):
        """Load a model previously saved with save() (formats 4 and 5)."""
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic == b"ZFCM":
            import zlib
            with open(path, "rb") as f:
                f.seek(4)
                payload = pickle.loads(zlib.decompress(f.read()))
        else:
            with open(path, "rb") as f:
                payload = pickle.load(f)

        if not isinstance(payload, dict) or "format_version" not in payload:
            raise ValueError(f"Not a valid model file: {path}")
        version = payload["format_version"]
        if version not in (4, 5):
            raise ValueError(
                f"Unsupported model format version {version} "
                f"(expected 4 or 5). Retrain the model."
            )

        params = payload["params"]
        engine = payload.get("engine", "zipf") if version == 5 else "zipf"

        if version == 5 and engine != "zipf":
            model = cls(
                n_gram_size=params["n_gram_size"],
                zipf_exponent=params["zipf_exponent"],
                zipf_weight=params["zipf_weight"],
                interp_weights=params.get("interp_weights"),
                personalization=params.get("personalization", 0.3),
                personalization_halflife=params.get(
                    "personalization_halflife", 200.0),
                smoothing=params.get("smoothing", engine),
                discount_scale=params.get("discount_scale", 1.0),
                zipf_prior_weight=params.get("zipf_prior_weight", 0.15),
                adaptive_prior=params.get("adaptive_prior", True),
                prior_mode=params.get("prior_mode", "zipf"),
                trust_halflife=params.get("trust_halflife", 2.0),
                trust_exponent=params.get("trust_exponent", 1.0),
            )
            model._restore_id_backend(payload)
        else:
            # Legacy payload (v4, or v5 saved in zipf mode).
            model = cls(
                n_gram_size=params["n_gram_size"],
                zipf_exponent=params["zipf_exponent"],
                zipf_weight=params["zipf_weight"],
                interp_weights=params.get("interp_weights"),
                personalization=params.get("personalization", 0.3),
                personalization_halflife=params.get(
                    "personalization_halflife", 200.0),
                smoothing="zipf" if version == 5 else "beast",
            )
            model.n_gram_model = defaultdict(
                lambda: defaultdict(int),
                {ctx: defaultdict(int, counts)
                 for ctx, counts in payload["n_gram_model"].items()},
            )
            model.corpus_freq = Counter(payload["corpus_freq"])
            model.total_words = payload["total_words"]
            if model._legacy_mode:
                model._recompute_unigram_dist()
            else:
                # Upgrade a v4 pickle in place to the BEAST engine.
                model._ingest_legacy_structures(model._ngram_legacy,
                                                model._corpus_freq_legacy)

        model.user_n_gram_model = defaultdict(
            lambda: defaultdict(int),
            {ctx: defaultdict(int, counts)
             for ctx, counts in payload.get("user_n_gram_model", {}).items()},
        )
        model.user_corpus_freq = Counter(payload.get("user_corpus_freq", {}))
        model.user_total_words = sum(model.user_corpus_freq.values())
        model._recompute_user_unigram_dist()
        return model

    def _restore_id_backend(self, payload):
        """Rebuild the id backend from a v5 KN-family payload."""
        self._w2i = {w: i for i, w in enumerate(payload["vocab"])}
        self._i2w = list(payload["vocab"])
        n_max = self.n_gram_size - 1
        self._counts = [None] + [dict() for _ in range(n_max)]
        for k in range(1, n_max + 1):
            src = payload["counts"].get(k, {})
            self._counts[k] = {
                tuple(ctx): (d, total, n1, n2, n3p)
                for ctx, (d, total, n1, n2, n3p) in src.items()
            }
        self._uni = {int(w): c for w, c in payload["uni"].items()}
        self._coc = [None if v is None else tuple(v) for v in payload["coc"]]
        self._D = [None if v is None else tuple(v) for v in payload["D"]]
        self._cont = {int(w): c for w, c in payload["cont"].items()}
        self._cont_total = payload["cont_total"]
        self.total_words = payload["total_words"]
        self._build_prior_distributions()
        self._legacy_view = None
        self._legacy_freq_view = None
        self._base_dist_cache = {}

    def _ingest_legacy_structures(self, ngram_legacy, corpus_freq):
        """
        Convert v4-style str-keyed structures into the id backend and derive
        all Kneser-Ney statistics — the zero-retraining upgrade path.
        """
        n_max = self.n_gram_size - 1
        w2i, i2w = self._w2i, self._i2w

        def intern(w):
            wid = w2i.get(w)
            if wid is None:
                wid = len(i2w)
                w2i[w] = wid
                i2w.append(w)
            return wid

        counts = [None] + [defaultdict(dict) for _ in range(n_max)]
        for ctx, d in ngram_legacy.items():
            k = len(ctx)
            if k < 1 or k > n_max:
                continue
            ctx_ids = tuple(intern(w) for w in ctx)
            dst = counts[k][ctx_ids]
            for w, c in d.items():
                wid = intern(w)
                dst[wid] = dst.get(wid, 0) + c

        for w, c in corpus_freq.items():
            intern(w)
        self._uni = {w2i[w]: c for w, c in corpus_freq.items()}
        self.total_words = sum(self._uni.values())

        for k in range(1, n_max + 1):
            finalized = {}
            for ctx, d in counts[k].items():
                total = 0
                n1 = n2 = n3p = 0
                for c in d.values():
                    total += c
                    if c == 1:
                        n1 += 1
                    elif c == 2:
                        n2 += 1
                    else:
                        n3p += 1
                finalized[ctx] = (d, total, n1, n2, n3p)
            self._counts[k] = finalized

        self._derive_kn_statistics()
        self._legacy_view = None
        self._legacy_freq_view = None
        self._base_dist_cache = {}

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def model_info(self):
        """Summary of engine configuration and learned statistics."""
        info = {
            "smoothing": self.smoothing,
            "n_gram_size": self.n_gram_size,
            "total_words": self.total_words,
            "vocab": len(self.corpus_freq),
        }
        if not self._legacy_mode:
            info["contexts_per_order"] = {
                k: len(self._counts[k]) for k in range(1, len(self._counts))
            }
            info["discounts_per_order"] = {
                k: self._D[k] for k in range(1, len(self._counts))
            }
            info["zipf_prior_weight"] = self.zipf_prior_weight
            info["adaptive_prior"] = self.adaptive_prior
            info["prior_mode"] = getattr(self, "prior_mode", "zipf")
            info["trust_halflife"] = self.trust_halflife
            info["trust_exponent"] = self.trust_exponent
            info["discount_scale"] = self.discount_scale
        return info


