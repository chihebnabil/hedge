"""
Run the WikiText-2 benchmark — staged edition.

The environment reaps long-running detached processes, so the benchmark is
split into short foreground stages that persist state under benchmarks/results/:

    --stage data          tokenize corpus, save gold streams
    --stage baselines     unigram / add-1 / markov-style / Witten-Bell
    --stage zipf_mle      interpolated trigram MLE (zipf engine, w=1.0)
    --stage zipf_legacy   v4 model (BEFORE) + its self-tuning
    --stage beast_train   train trigram id-backend, save beast.pkl
    --stage beast_eval    load backend, self-tune, eval BEAST (AFTER)
    --stage beast_ablate  load backend, eval MKN and fixed-prior ablations
    --stage five          bonus: 5-gram beast
    --stage report        render RESULTS.md from accumulated results.json

Each stage appends its rows to results.json. Metrics: test perplexity,
top-1 / top-3 accuracy, train time, model size, suggestion latency.
"""

import argparse
import json
import os
import pickle
import sys
import time
import tracemalloc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from ZipfNextWordPredictor import ZipfNextWordPredictor  # noqa: E402
import common  # noqa: E402
from baselines import (UnigramMLE, BigramLaplace, BigramInterpolated,  # noqa: E402
                       WittenBellTrigram)

RESULTS_DIR = os.path.join(HERE, "results")
GOLD_PKL = os.path.join(RESULTS_DIR, "gold_streams.pkl")
BEAST_PKL = os.path.join(RESULTS_DIR, "beast_trigram.pkl")
BEAST5_PKL = os.path.join(RESULTS_DIR, "beast_5gram.pkl")
RESULTS_JSON = os.path.join(RESULTS_DIR, "results.json")
TUNE_TOKENS = 40000
LATENCY_QUERIES = 300


# --------------------------------------------------------------------- #
# state helpers
# --------------------------------------------------------------------- #

def load_gold():
    with open(GOLD_PKL, "rb") as f:
        return pickle.load(f)


def load_results():
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            return json.load(f).get("results", [])
    return []


def save_results(results, protocol=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {"results": results}
    if protocol:
        payload["protocol"] = protocol
    common.json_dump(payload, RESULTS_JSON)


class PredictorAdapter:
    """Score a predictor through the shared harness loop on the gold stream.

    KN-family: logprob via _p_kn (O(orders) per token); top-k via the exact
    memoized context walk.

    Legacy zipf: the interpolated distribution spans the whole vocabulary
    (the unigram level alone has ~33k entries), so both logprob and top-k are
    computed from the per-level blended components. logprob is O(levels);
    top-k merges observed continuations at each level with the global unigram
    leaders -- exact, because any word outside that candidate set scores
    strictly below the unigram leaders (each level's blended distribution
    sums to 1, so the interpolated scores need no re-normalization; v4's own
    evaluate() normalization was a floating-point no-op).
    """

    def __init__(self, model):
        self.m = model
        self.kn = not model._legacy_mode
        self._topk_cache = {}
        self._level_cache = {}
        self._level_top_cache = {}
        self._uni_leaders = None

    def _levels(self, ctx):
        """Per-level (weight, blended distribution) with caching (legacy)."""
        m = self.m
        out = []
        for ctx_len, weight in enumerate(m.interp_weights):
            if weight <= 0:
                continue
            if ctx_len == 0:
                out.append((weight, m._unigram_dist))
            elif ctx_len <= len(ctx):
                key = (ctx_len, ctx[-ctx_len:])
                d = self._level_cache.get(key)
                if d is None:
                    counts = m._ngram_legacy.get(key[1])
                    d = m._blended_distribution(counts) if counts else {}
                    if len(self._level_cache) < 400000:
                        self._level_cache[key] = d
                if d:
                    out.append((weight, d))
        return out

    def _leaders(self):
        if self._uni_leaders is None:
            d = self.m._unigram_dist
            self._uni_leaders = [w for w, _ in
                                 sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:80]]
        return self._uni_leaders

    def logprob(self, target, ctx):
        m = self.m
        if self.kn:
            wid = m._w2i.get(target)
            if wid is None:
                return 0.0
            return m._p_kn(wid, m._context_ids(list(ctx)))
        p = 0.0
        for weight, d in self._levels(ctx):
            p += weight * d.get(target, 0.0)
        return p

    _POOL = 64  # per-level candidate cap (matches Witten-Bell baseline's 40)

    def _levels_top(self, ctx):
        """Per-level (weight, top items) with caching (legacy engine)."""
        m = self.m
        out = []
        for ctx_len, weight in enumerate(m.interp_weights):
            if weight <= 0:
                continue
            if ctx_len == 0:
                if self._uni_leaders is None:
                    self._uni_leaders = sorted(
                        [(w, p) for w, p in m._unigram_dist.items()],
                        key=lambda kv: (-kv[1], kv[0]))[:80]
                out.append((weight, self._uni_leaders))
            elif ctx_len <= len(ctx):
                key = (ctx_len, ctx[-ctx_len:])
                entry = self._level_top_cache.get(key)
                if entry is None:
                    counts = m._ngram_legacy.get(key[1])
                    d = m._blended_distribution(counts) if counts else {}
                    if d and len(d) > self._POOL:
                        items = sorted(d.items(),
                                       key=lambda kv: (-kv[1], kv[0]))[:self._POOL]
                    else:
                        items = list(d.items()) if d else []
                    entry = items
                    if len(self._level_top_cache) < 300000:
                        self._level_top_cache[key] = entry
                if entry:
                    out.append((weight, entry))
        return out

    def topk(self, ctx, k):
        key = (ctx, k)
        hit = self._topk_cache.get(key)
        if hit is not None:
            return hit
        if self.kn:
            out = self.m._eval_topk_cached(tuple(ctx), k)
        else:
            cands = {}
            for weight, items in self._levels_top(ctx):
                for w, v in items:
                    cands[w] = cands.get(w, 0.0) + weight * v
            ranked = sorted(cands.items(), key=lambda kv: (-kv[1], kv[0]))
            out = [w for w, _ in ranked[:k]]
        if len(self._topk_cache) < 500000:
            self._topk_cache[key] = out
        return out


def subsample(sents, max_tokens):
    out, budget = [], max_tokens
    for s in sents:
        if budget <= 0:
            break
        out.append(s)
        budget -= len(s)
    return out


def tune_text(valid_sents):
    return "\n".join(" ".join(s) for s in subsample(valid_sents, TUNE_TOKENS))


def subsample_text_25k(valid_sents):
    return "\n".join(" ".join(s) for s in subsample(valid_sents, 25000))


def suggestion_latency(model, contexts, k=5):
    t0 = time.perf_counter()
    n = 0
    for ctx in contexts[:LATENCY_QUERIES]:
        model.predict_next_words(ctx, num_predictions=k)
        n += 1
    return (time.perf_counter() - t0) / max(n, 1) * 1000.0


def eval_predictor(model, test_sents, valid_sents):
    adapter = PredictorAdapter(model)
    tr = common.evaluate_harness(adapter.logprob, adapter.topk, test_sents)
    va = common.evaluate_harness(adapter.logprob, adapter.topk,
                                 subsample(valid_sents, TUNE_TOKENS))
    tr["valid_perplexity"] = va["perplexity"]
    return tr


def eval_baseline(model, valid_sents, test_sents):
    tr = common.evaluate_harness(model.logprob, model.topk, test_sents)
    va = common.evaluate_harness(model.logprob, model.topk,
                                 subsample(valid_sents, TUNE_TOKENS))
    tr["valid_perplexity"] = va["perplexity"]
    return tr


def record(name, engine, train_s, metrics, model_obj=None, note="",
           model_path=None):
    results = load_results()
    raw_b = z_b = 0
    if model_obj is not None:
        raw_b, z_b = common.compressed_size_bytes(model_obj)
    elif model_path and os.path.exists(model_path):
        raw_b = common.file_size(model_path)
    row = {
        "model": name, "engine": engine,
        "train_seconds": round(train_s, 2),
        "model_bytes": raw_b, "model_bytes_compressed": z_b,
        "suggestion_ms": metrics.get("suggestion_ms"),
        "note": note,
    }
    row.update({k: v for k, v in metrics.items() if k != "suggestion_ms"})
    results = [r for r in results if r["model"] != name] + [row]
    save_results(results)
    print(f"  -> {name}: PP={row['perplexity']:.1f} "
          f"top1={row['top1_accuracy']*100:.1f}% "
          f"top3={row['top3_accuracy']*100:.1f}% "
          f"train={row['train_seconds']}s size={raw_b/1e6:.1f}MB")


def latency_contexts(test_sents):
    ctxs = []
    for sent in test_sents:
        for i in range(2, len(sent)):
            ctxs.append(" ".join(sent[i - 2:i]))
    step = max(1, len(ctxs) // (LATENCY_QUERIES * 3))
    return ctxs[::step]


# --------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------- #

def stage_data():
    print("== stage: data ==")
    train, valid, test = common.load_wikitext2()
    with open(GOLD_PKL, "wb") as f:
        pickle.dump({"train": train, "valid": valid, "test": test}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    n_train = sum(len(s) for s in train)
    print(f"  saved gold streams: train {n_train:,} tokens, "
          f"valid {sum(len(s) for s in valid):,}, test {sum(len(s) for s in test):,}")

    # tokenization identity check (predictor re-tokenizes joined lines identically)
    probe_lines = [" ".join(s) for s in train[:400]]
    probe = ZipfNextWordPredictor()
    flat = [w.lower() for s in probe.preprocess("\n".join(probe_lines)) for w in s]
    assert flat == [w for s in train[:400] for w in s], "identity check FAILED"
    print("  tokenization identity check: OK")


def stage_baselines(only=None):
    print("== stage: baselines ==")
    g = load_gold()
    lat = latency_contexts(g["test"])
    models = {
        "unigram": (UnigramMLE, "Unigram MLE (floor)"),
        "laplace": (BigramLaplace, "Bigram add-1 (Laplace)"),
        "markov": (BigramInterpolated, "Bigram interp (Markov-style)"),
        "wittenbell": (WittenBellTrigram, "Trigram Witten-Bell"),
    }
    for key, (cls, name) in models.items():
        if only and key not in only:
            continue
        print(f"== {name} ==")
        m = cls()
        t0 = time.perf_counter()
        m.train(g["train"])
        train_s = time.perf_counter() - t0
        metrics = eval_baseline(m, g["valid"], g["test"])
        metrics["suggestion_ms"] = suggestion_latency(m, lat)
        record(name, "count-based", train_s, metrics, model_obj=m)
        del m


def stage_zipf_mle():
    print("== stage: zipf_mle ==")
    g = load_gold()
    train_lines = [" ".join(s) for s in g["train"]]
    t0 = time.perf_counter()
    m = ZipfNextWordPredictor(corpus_texts=train_lines, n_gram_size=3,
                              smoothing="zipf", zipf_weight=1.0)
    train_s = time.perf_counter() - t0
    metrics = eval_predictor(m, g["test"], g["valid"])
    metrics["suggestion_ms"] = suggestion_latency(m, latency_contexts(g["test"]))
    record("Interpolated trigram MLE", "zipf(w=1.0)", train_s, metrics,
           model_obj=m)


def stage_zipf_legacy():
    print("== stage: zipf_legacy ==")
    g = load_gold()
    train_lines = [" ".join(s) for s in g["train"]]
    t0 = time.perf_counter()
    m = ZipfNextWordPredictor(corpus_texts=train_lines, n_gram_size=3,
                              smoothing="zipf", zipf_weight=0.7)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    rep = m.tune(tune_text(g["valid"]))
    tune_s = time.perf_counter() - t0
    print(f"  tune: {rep['trials_run']} combos in {tune_s:.0f}s -> "
          f"zipf_weight={rep['best']['zipf_weight']} "
          f"profile={rep['best']['interp_profile']}")
    metrics = eval_predictor(m, g["test"], g["valid"])
    metrics["suggestion_ms"] = suggestion_latency(m, latency_contexts(g["test"]))
    record("Zipf v4 (BEFORE, tuned)", "zipf(w=tuned)", train_s + tune_s,
           metrics, model_obj=m,
           note=f"tuned zipf_weight={rep['best']['zipf_weight']}")


def stage_beast_train():
    print("== stage: beast_train ==")
    g = load_gold()
    train_lines = [" ".join(s) for s in g["train"]]
    t0 = time.perf_counter()
    m = ZipfNextWordPredictor(corpus_texts=train_lines, n_gram_size=3,
                              smoothing="beast")
    train_s = time.perf_counter() - t0
    info = m.model_info()
    print(f"  vocab={info['vocab']:,} contexts={info['contexts_per_order']} "
          f"D={ {k: tuple(round(x, 3) for x in v) for k, v in info['discounts_per_order'].items()} }")
    m.save(BEAST_PKL)
    with open(os.path.join(RESULTS_DIR, "backend_meta.json"), "w") as f:
        json.dump({"train_seconds": round(train_s, 2),
                   "vocab": info["vocab"],
                   "contexts": info["contexts_per_order"]}, f)
    print(f"  trained in {train_s:.1f}s, saved {BEAST_PKL} "
          f"({common.file_size(BEAST_PKL)/1e6:.1f} MB)")


def _load_beast():
    m = ZipfNextWordPredictor.load(BEAST_PKL)
    m.personalization = 0.0
    return m


def stage_beast_eval():
    print("== stage: beast_eval ==")
    g = load_gold()
    m = _load_beast()
    m.smoothing = "beast"
    m.zipf_prior_weight = 0.15
    m.adaptive_prior = True
    m.discount_scale = 1.0
    m._recompute_discounts()
    t0 = time.perf_counter()
    rep = m.tune(subsample_text_25k(g["valid"]),
                 discount_scales=[0.85, 1.0, 1.15],
                 zipf_prior_weights=[0.0, 0.05, 0.1, 0.2, 0.3],
                 adaptive_options=[True, False])
    tune_s = time.perf_counter() - t0
    b = rep["best"]
    print(f"  tune: {rep['trials_run']} combos in {tune_s:.0f}s -> "
          f"scale={b['discount_scale']} z={b['zipf_prior_weight']} "
          f"adaptive={b['adaptive_prior']}")
    backend_train = 0.0
    meta_path = os.path.join(RESULTS_DIR, "backend_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            backend_train = json.load(f).get("train_seconds", 0.0)
    metrics = eval_predictor(m, g["test"], g["valid"])
    metrics["suggestion_ms"] = suggestion_latency(m, latency_contexts(g["test"]))
    record("BEAST (AFTER, tuned)", "beast", tune_s + backend_train, metrics,
           model_path=BEAST_PKL,
           note=(f"scale={b['discount_scale']}, z={b['zipf_prior_weight']}, "
                 f"adaptive={b['adaptive_prior']}, discounts auto-estimated"))


def stage_beast_ablate():
    print("== stage: beast_ablate ==")
    g = load_gold()
    lat = latency_contexts(g["test"])
    m = _load_beast()

    m.smoothing = "modified_kneser_ney"
    m.zipf_prior_weight = 0.0
    m.adaptive_prior = False
    m.discount_scale = 1.0
    m._recompute_discounts()
    metrics = eval_predictor(m, g["test"], g["valid"])
    metrics["suggestion_ms"] = suggestion_latency(m, lat)
    record("Modified Kneser-Ney (no prior)", "modified_kneser_ney", 0.0,
           metrics, model_path=BEAST_PKL)

    m.smoothing = "beast"
    m.zipf_prior_weight = 0.15
    m.adaptive_prior = False
    m._recompute_discounts()
    metrics = eval_predictor(m, g["test"], g["valid"])
    metrics["suggestion_ms"] = suggestion_latency(m, lat)
    record("MKN + fixed Zipf prior", "beast(z fixed)", 0.0,
           metrics, model_path=BEAST_PKL)

    m.smoothing = "kneser_ney"
    m.zipf_prior_weight = 0.0
    m._recompute_discounts()
    metrics = eval_predictor(m, g["test"], g["valid"])
    metrics["suggestion_ms"] = suggestion_latency(m, lat)
    record("Simple Kneser-Ney", "kneser_ney", 0.0,
           metrics, model_path=BEAST_PKL)


def stage_five():
    print("== stage: five ==")
    g = load_gold()
    train_s = 0.0
    peak_note = ""
    if os.path.exists(BEAST5_PKL):
        print("  reusing saved 5-gram backend")
        train_s = float(json.load(open(RESULTS_JSON))["results"][-1]
                        .get("train_seconds", 0) or 0) \
            if os.path.exists(RESULTS_JSON) else 0.0
    else:
        train_lines = [" ".join(s) for s in g["train"]]
        tracemalloc.start()
        t0 = time.perf_counter()
        m = ZipfNextWordPredictor(corpus_texts=train_lines, n_gram_size=5,
                                  smoothing="beast")
        train_s = time.perf_counter() - t0
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        m.save(BEAST5_PKL)
        peak_note = f", peak RAM ~{peak/1e9:.2f} GB"
        print(f"  trained 5-gram in {train_s:.1f}s{peak_note}")

    m = ZipfNextWordPredictor.load(BEAST5_PKL)
    m.personalization = 0.0
    t0 = time.perf_counter()
    rep = m.tune(tune_text(g["valid"]), discount_scales=[1.0],
                 zipf_prior_weights=[0.0, 0.1, 0.2], adaptive_options=[True])
    tune_s = time.perf_counter() - t0
    metrics = eval_predictor(m, g["test"], g["valid"])
    metrics["suggestion_ms"] = suggestion_latency(m, latency_contexts(g["test"]))
    record("BEAST 5-gram", "beast(n=5)", train_s + tune_s, metrics,
           model_path=BEAST5_PKL,
           note=(f"{peak_note}, z={rep['best']['zipf_prior_weight']}"))
    os.remove(BEAST5_PKL)


def stage_report():
    with open(GOLD_PKL, "rb") as f:
        g = pickle.load(f)
    n_test = sum(len(s) - 1 for s in g["test"])
    results = sorted(load_results(), key=lambda r: r.get("perplexity", 9e9)
                     if isinstance(r.get("perplexity"), (int, float)) else 9e9)
    md = render_markdown(results, n_test)
    with open(os.path.join(HERE, "..", "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(RESULTS_DIR, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


def render_markdown(results, n_test_tokens):
    lines = [
        "# Benchmark results (WikiText-2)",
        "",
        f"Test tokens: {n_test_tokens:,} · one shared gold token stream · "
        f"one harness loop for every model · trigram order unless noted · "
        f"pure-stdlib models, single core",
        "",
        "Context: all rows are classical n-gram models; the best widely reported "
        "neural result on this benchmark (AWD-LSTM ~65.8, Merity et al., 2018) "
        "is far below this class. No transformer number is cited because "
        "transformer LM papers report on WikiText-103/PTB/enwik8, not WikiText-2.",
        "",
        "| Model | Test PP ↓ | Top-1 ↑ | Top-3 ↑ | Train+tune (s) | Size (MB) | ms/sugg |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if not isinstance(r.get("perplexity"), (int, float)):
            lines.append(f"| {r['model']} | {r.get('skipped', 'n/a')} | - | - | - | - | - |")
            continue
        size = r.get("model_bytes", 0) / 1e6
        ms = r.get("suggestion_ms")
        ms_s = f"{ms:.1f}" if ms else "-"
        lines.append(
            f"| {r['model']} | {r['perplexity']:.1f} | "
            f"{r['top1_accuracy']*100:.1f}% | {r['top3_accuracy']*100:.1f}% | "
            f"{r['train_seconds']} | {size:.1f} | {ms_s} |")
    lines += ["",
              "PP = word-level perplexity, natural log (lower is better).",
              "Top-k = next-word suggestion accuracy on the test stream (higher is better).",
              "Size = raw pickle of the fitted model (what save() writes, before compression)."]
    return "\n".join(lines)


STAGES = {
    "data": stage_data,
    "baselines": stage_baselines,
    "zipf_mle": stage_zipf_mle,
    "zipf_legacy": stage_zipf_legacy,
    "beast_train": stage_beast_train,
    "beast_eval": stage_beast_eval,
    "beast_ablate": stage_beast_ablate,
    "five": stage_five,
    "report": stage_report,
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of baseline keys")
    args = ap.parse_args()
    if args.stage == "baselines":
        stage_baselines(only=args.only)
    else:
        STAGES[args.stage]()
