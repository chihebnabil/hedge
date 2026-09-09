import glob
import math
import os
import pickle
import random
import re
from collections import defaultdict, Counter

MODEL_FORMAT_VERSION = 4


class ZipfNextWordPredictor:
    def __init__(self, corpus_texts=None, n_gram_size=3, zipf_exponent=1.0,
                 zipf_weight=0.7, interp_weights=None,
                 personalization=0.3, personalization_halflife=200.0):
        """
        Initialize the next word predictor with Zipf's law parameters

        Args:
            corpus_texts: List of texts to build the model from
            n_gram_size: Size of n-grams to use (default: 3 for trigrams)
            zipf_exponent: Exponent for Zipf's law (default: 1.0)
            zipf_weight: Blend factor between empirical counts and the Zipf
                rank prior. P = zipf_weight * P_empirical + (1 - zipf_weight) * P_zipf.
                1.0 = pure counts, 0.0 = pure rank-based Zipf (legacy behavior).
            interp_weights: Interpolation weights per context length
                (index 0 = unigram/corpus level, index n-1 = longest context).
                Defaults to exponentially favoring longer contexts.
            personalization: Max weight (0-1) given to the user layer when
                blending user-learned vs corpus distributions.
            personalization_halflife: User words needed for the personalization
                weight to reach half its maximum (cold-start protection).
        """
        if n_gram_size < 2:
            raise ValueError("n_gram_size must be >= 2")

        self.n_gram_size = n_gram_size
        self.zipf_exponent = zipf_exponent
        self.zipf_weight = min(max(zipf_weight, 0.0), 1.0)
        self.interp_weights = (
            self._default_interp_weights() if interp_weights is None
            else self._normalize_weights(interp_weights)
        )
        self.personalization = min(max(personalization, 0.0), 1.0)
        self.personalization_halflife = max(float(personalization_halflife), 1.0)

        # Base (corpus) model components
        self.n_gram_model = defaultdict(lambda: defaultdict(int))
        self.corpus_freq = Counter()
        self.total_words = 0
        self._unigram_dist = {}

        # User layer: learns from observed text, never touches the base model
        self.user_n_gram_model = defaultdict(lambda: defaultdict(int))
        self.user_corpus_freq = Counter()
        self.user_total_words = 0
        self._user_unigram_dist = {}

        if corpus_texts:
            self.build_model(corpus_texts)

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
        """Build n-gram model and frequency distributions from corpus texts"""
        all_sentences = []
        for text in corpus_texts:
            all_sentences.extend(self.preprocess(text))
        self._update_counts(all_sentences,
                            self.n_gram_model, self.corpus_freq)
        self.total_words = sum(self.corpus_freq.values())
        self._recompute_unigram_dist()

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
        self.user_n_gram_model = defaultdict(lambda: defaultdict(int))
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
    # Probability math: Zipf blend + interpolated smoothing
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
        self._unigram_dist = self._blended_distribution(self.corpus_freq)

    def _recompute_user_unigram_dist(self):
        self._user_unigram_dist = self._blended_distribution(self.user_corpus_freq)

    def _effective_alpha(self):
        """Personalization weight scaled by how much user evidence exists."""
        if self.personalization <= 0 or self.user_total_words == 0:
            return 0.0
        u = self.user_total_words
        h = self.personalization_halflife
        return self.personalization * (u / (u + h))

    def _interp_scores(self, context, n_gram_model, unigram_dist):
        """Interpolated scores for one layer (base or user)."""
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

    def _score_distribution(self, context):
        """
        Final distribution: interpolated corpus scores blended with the
        user layer scores using the adaptive personalization weight.
        """
        context = tuple(context)
        base = self._interp_scores(context, self.n_gram_model, self._unigram_dist)

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

        scores = self._score_distribution(context)
        if not scores:
            return []

        dist = self._apply_temperature(scores, temperature)
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

        for sentence in self.preprocess(text):
            words = [w.lower() for w in sentence]
            for i in range(1, len(words)):
                context = tuple(words[max(0, i - (self.n_gram_size - 1)):i])
                target = words[i]

                scores = self._score_distribution(context)
                if scores:
                    norm = sum(scores.values())
                    if norm > 0:
                        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
                        top_k = [word for word, _ in ranked[:k]]
                        if target in top_k:
                            hits += 1
                        log_prob_sum += math.log(max(scores.get(target, 0.0) / norm, 1e-12))
                else:
                    log_prob_sum += math.log(1e-12)
                tokens += 1

        if tokens == 0:
            return {"top_k_accuracy": 0.0, "perplexity": float("inf"), "tokens": 0}

        return {
            "top_k_accuracy": hits / tokens,
            "perplexity": math.exp(-log_prob_sum / tokens),
            "tokens": tokens,
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

    def tune(self, held_out_text, zipf_weights=None, interp_profiles=None):
        """
        Sweep hyperparameters on held-out text and apply the best combination
        (lowest perplexity) to this model. The model finds its own Zipf curve.

        Args:
            held_out_text: Text NOT used for training, used to score each setting
            zipf_weights: Grid of zipf_weight values to try
                (default: 0.0 to 1.0 in steps of 0.1)
            interp_profiles: Names of interpolation profiles to try
                (default: all of them)

        Returns:
            Dict with "best" (dict with zipf_weight, interp_profile, perplexity,
            top_k_accuracy) and "trials" (every combination tried).
        """
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
                self.interp_weights = self._normalize_weights(profiles[best["interp_profile"]])
            else:
                self.zipf_weight, self.interp_weights = original
            self._recompute_unigram_dist()

        return {"best": best, "trials": trials, "trials_run": len(trials)}

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path):
        """Serialize the trained model (including the user layer) to a file"""
        payload = {
            "format_version": MODEL_FORMAT_VERSION,
            "params": {
                "n_gram_size": self.n_gram_size,
                "zipf_exponent": self.zipf_exponent,
                "zipf_weight": self.zipf_weight,
                "interp_weights": self.interp_weights,
                "personalization": self.personalization,
                "personalization_halflife": self.personalization_halflife,
            },
            "n_gram_model": {ctx: dict(counts) for ctx, counts in self.n_gram_model.items()},
            "corpus_freq": dict(self.corpus_freq),
            "total_words": self.total_words,
            "user_n_gram_model": {ctx: dict(counts) for ctx, counts in self.user_n_gram_model.items()},
            "user_corpus_freq": dict(self.user_corpus_freq),
        }
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
            "user_n_gram_model": {ctx: dict(counts) for ctx, counts in self.user_n_gram_model.items()},
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
        """Load a model previously saved with save()"""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        if not isinstance(payload, dict) or "format_version" not in payload:
            raise ValueError(f"Not a valid model file: {path}")
        if payload["format_version"] != MODEL_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported model format version {payload['format_version']} "
                f"(expected {MODEL_FORMAT_VERSION}). Retrain the model."
            )

        params = payload["params"]
        model = cls(
            n_gram_size=params["n_gram_size"],
            zipf_exponent=params["zipf_exponent"],
            zipf_weight=params["zipf_weight"],
            interp_weights=params.get("interp_weights"),
            personalization=params.get("personalization", 0.3),
            personalization_halflife=params.get("personalization_halflife", 200.0),
        )
        model.n_gram_model = defaultdict(
            lambda: defaultdict(int),
            {ctx: defaultdict(int, counts) for ctx, counts in payload["n_gram_model"].items()},
        )
        model.corpus_freq = Counter(payload["corpus_freq"])
        model.total_words = payload["total_words"]
        model._recompute_unigram_dist()

        model.user_n_gram_model = defaultdict(
            lambda: defaultdict(int),
            {ctx: defaultdict(int, counts)
             for ctx, counts in payload.get("user_n_gram_model", {}).items()},
        )
        model.user_corpus_freq = Counter(payload.get("user_corpus_freq", {}))
        model.user_total_words = sum(model.user_corpus_freq.values())
        model._recompute_user_unigram_dist()
        return model
