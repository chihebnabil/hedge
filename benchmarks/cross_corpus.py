r"""Cross-corpus validation of the Zipf-continuation prior.

Question (raised in external review): does the prior's measured gain over
plain Modified Kneser-Ney hold outside WikiText-2, and does it beat another
named sparse-region regularizer - Witten-Bell's confidence weighting - when
both are placed in the same junction of the same backbone?

Design
------
* Four corpora, different sizes and domains:
    - wikitext2   : Wikipedia articles, ~1.93M train tokens (official split)
    - ptb         : Penn Treebank (Wall Street Journal news), ~0.89M tokens
                    (Mikolov split, tomsercu/lstm)
    - brown       : Brown corpus (15 genres, tagged -> detagged),
                    deterministic seeded file split 80/10/10
    - wt103prefix : WikiText-103 official splits, first ~8M train tokens,
                    first 60k valid/test tokens (compute-budget prefix)
* One shared gold token stream per corpus (the predictor's own tokenizer,
  exactly like the main benchmark), OOV -> "unk".
* One trained trigram per corpus; three variants tuned and scored on it:
    1. mkn          : Modified Kneser-Ney (prior off) - discount scale tuned
    2. witten_bell  : textbook Witten-Bell in the same junction
                      (P = c/(T+D) + D/(T+D) * P_bo) - parameter-free
    3. zipf_prior   : the novel Zipf-continuation prior - discount scale,
                      prior ceiling and adaptive switch tuned
  All variants see the same tune grid and the same capped tune text.
* Test perplexity + top-1/top-3 from the model's own exact scorer.

Usage:
    python benchmarks/cross_corpus.py --corpus ptb [--quick]
Results: benchmarks/cross_corpus/results/<corpus>.json
"""

import argparse
import json
import os
import pickle
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ZipfNextWordPredictor import ZipfNextWordPredictor  # noqa: E402
import common  # noqa: E402

_HEADER_RE = re.compile(r"^\s*=.*=\s*$")


# --------------------------------------------------------------------------- #
# Corpus loaders: every loader returns (train_sents, valid_sents, test_sents)
# where sentences are lists of lowercase tokens sharing the predictor's
# tokenizer, with valid/test OOV mapped to "unk".
# --------------------------------------------------------------------------- #

def _finish_stream_split(train, valid_raw, test_raw):
    vocab = set()
    for sent in train:
        vocab.update(sent)
    unk = "unk"
    valid, test = [], []
    for split, store in (("valid", valid), ("test", test)):
        src = valid_raw if split == "valid" else test_raw
        for sent in src:
            store.append([w if w in vocab else unk for w in sent])
    return train, valid, test


def load_ptb(data_dir):
    paths = {split: os.path.join(data_dir, f"ptb.{split}.txt")
             for split in ("train", "valid", "test")}
    streams = {}
    for split, path in paths.items():
        sents = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                sents.extend(common.tokenize_line(line.strip()))
        streams[split] = sents
        print(f"  ptb {split}: {len(sents)} sents "
              f"({sum(len(s) for s in sents):,} tokens)")
    return _finish_stream_split(streams["train"], streams["valid"],
                                streams["test"])


def load_brown(data_dir, seed=42):
    brown_dir = os.path.join(data_dir, "brown", "brown")
    if not os.path.isdir(brown_dir):
        brown_dir = os.path.join(data_dir, "brown")
    files = sorted(f for f in os.listdir(brown_dir)
                   if re.match(r"^c[a-r]\d\d$", f))
    if len(files) < 100:
        raise RuntimeError(f"Brown corpus looks incomplete: {len(files)} files")
    rng = random.Random(seed)
    rng.shuffle(files)  # genre-clustered file order -> seeded interleave
    n = len(files)
    splits = {
        "train": files[:int(n * 0.8)],
        "valid": files[int(n * 0.8):int(n * 0.9)],
        "test": files[int(n * 0.9):],
    }
    streams = {}
    for split, names in splits.items():
        sents = []
        for name in names:
            with open(os.path.join(brown_dir, name), "r",
                      encoding="latin-1") as f:
                for line in f:
                    # word/TAG pairs -> raw text for the shared tokenizer
                    words = [tok.rsplit("/", 1)[0] for tok in line.split()
                             if "/" in tok]
                    if words:
                        sents.extend(common.tokenize_line(" ".join(words)))
        streams[split] = sents
        print(f"  brown {split}: {len(sents)} sents "
              f"({sum(len(s) for s in sents):,} tokens)")
    return _finish_stream_split(streams["train"], streams["valid"],
                                streams["test"])


def _wt_style_stream(path, token_cap=None):
    """WikiText-style file -> tokenized sentences (headers dropped)."""
    sents = []
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or _HEADER_RE.match(line):
                continue
            got = common.tokenize_line(line)
            sents.extend(got)
            total += sum(len(s) for s in got)
            if token_cap is not None and total >= token_cap:
                break
    return sents


def load_wt103_prefix(data_dir, train_tokens=8_000_000, eval_tokens=60_000):
    base = os.path.join(data_dir, "wikitext-103")
    paths = {split: os.path.join(base, f"wiki.{split}.tokens")
             for split in ("train", "valid", "test")}
    cache = os.path.join(data_dir, f"wt103prefix_cache_{train_tokens}_"
                         f"{eval_tokens}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            train, valid, test = pickle.load(f)
        print("  wt103prefix: loaded tokenized streams from cache")
    else:
        train = _wt_style_stream(paths["train"], token_cap=train_tokens)
        valid = _wt_style_stream(paths["valid"], token_cap=eval_tokens)
        test = _wt_style_stream(paths["test"], token_cap=eval_tokens)
        with open(cache, "wb") as f:
            pickle.dump((train, valid, test), f, protocol=4)
    print(f"  wt103prefix train: {len(train)} sents "
          f"({sum(len(s) for s in train):,} tokens)")
    print(f"  wt103prefix valid/test: "
          f"{sum(len(s) for s in valid):,} / {sum(len(s) for s in test):,} tokens")
    return _finish_stream_split(train, valid, test)


LOADERS = {
    "wikitext2": lambda dd: common.load_wikitext2(dd),
    "ptb": load_ptb,
    "brown": load_brown,
    "wt103prefix": load_wt103_prefix,
}


# --------------------------------------------------------------------------- #
# Experiment protocol
# --------------------------------------------------------------------------- #

def tune_and_score(model, name, variant, test_text, args):
    """Tune one variant on valid, score on test. Returns a result dict.

    Variant grids are EXPLICIT so the sweeps cannot bleed into each other:
      mkn         -> discount scale only (prior structurally off)
      witten_bell -> nothing (parameter-free; single identity trial)
      zipf_prior  -> discount scale x prior ceiling x adaptive switch
    """
    model.prior_mode = {"mkn": "zipf", "witten_bell": "witten_bell",
                        "zipf_prior": "zipf"}[variant]
    if variant == "mkn":
        tune_kw = {"discount_scales": args.ds_grid,
                   "zipf_prior_weights": [0.0],
                   "adaptive_options": [False]}
    elif variant == "witten_bell":
        tune_kw = {"discount_scales": [1.0],
                   "zipf_prior_weights": [0.0],
                   "adaptive_options": [False]}
    else:
        tune_kw = {"discount_scales": args.ds_grid,
                   "zipf_prior_weights": args.z_grid,
                   "adaptive_options": [True, False]}

    t0 = time.perf_counter()
    out = model.tune(args.tune_text, max_tokens=args.tune_cap, **tune_kw)
    tune_s = time.perf_counter() - t0
    best = out["best"]

    t0 = time.perf_counter()
    res3 = model.evaluate(test_text, k=3)
    res1 = model.evaluate(test_text, k=1)
    eval_s = time.perf_counter() - t0

    result = {
        "variant": variant,
        "best_params": {k: best[k] for k in
                        ("discount_scale", "zipf_prior_weight",
                         "adaptive_prior", "prior_mode")},
        "valid_perplexity": best["perplexity"],
        "test_perplexity": res3["perplexity"],
        "test_top1": res1["top_k_accuracy"],
        "test_top3": res3["top_k_accuracy"],
        "test_tokens": res3["tokens"],
        "trials_run": out["trials_run"],
        "tune_seconds": tune_s,
        "eval_seconds": eval_s,
    }
    print(f"  [{name}] {variant:<12} PP {result['test_perplexity']:.1f} "
          f"top1 {result['test_top1']*100:.1f}% "
          f"top3 {result['test_top3']*100:.1f}% "
          f"(best z={result['best_params']['zipf_prior_weight']}, "
          f"ds={result['best_params']['discount_scale']}, "
          f"adaptive={result['best_params']['adaptive_prior']}, "
          f"{out['trials_run']} trials, tune {tune_s:.0f}s, eval {eval_s:.0f}s)")
    return result


def run_corpus(corpus, args):
    print(f"== corpus: {corpus} ==")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "cross_corpus", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{corpus}.json")

    print(f"== corpus: {corpus} ==")
    train, valid, test = LOADERS[corpus](args.data_dir)
    n_train = sum(len(s) for s in train)
    n_valid = sum(len(s) for s in valid)
    n_test = sum(len(s) for s in test)
    print(f"  totals: train {n_train:,} / valid {n_valid:,} / test {n_test:,}")

    train_text = common.streams_to_text(train)
    args.tune_text = common.streams_to_text(valid)
    test_text = common.streams_to_text(test)

    t0 = time.perf_counter()
    model = ZipfNextWordPredictor(corpus_texts=[train_text], n_gram_size=3,
                                  smoothing="beast")
    train_s = time.perf_counter() - t0
    print(f"  trained trigram in {train_s:.1f}s "
          f"(vocab {len(model.corpus_freq):,})")

    raw_b, zlib_b = common.compressed_size_bytes(model)
    print(f"  serialized size: {raw_b/1e6:.1f} MB raw / {zlib_b/1e6:.1f} MB zlib")

    extra = None  # variant grids are fully explicit in tune_and_score

    results = {
        "corpus": corpus,
        "train_tokens": n_train,
        "valid_tokens": n_valid,
        "test_tokens": n_test,
        "train_seconds": train_s,
        "raw_bytes": raw_b,
        "zlib_bytes": zlib_b,
        "tune_cap_tokens": args.tune_cap,
        "ds_grid": args.ds_grid,
        "z_grid": args.z_grid,
        "variants": {},
    }
    for variant in args.variants.split(","):
        results["variants"][variant] = tune_and_score(
            model, corpus, variant, test_text, args)

    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("tune_cap_tokens") == results["tune_cap_tokens"]:
            merged = dict(prev)
            merged["variants"].update(results["variants"])
            results = merged
            print("  merged with existing results")

    common.json_dump(results, out_path)
    print("  wrote", out_path)
    return results


def main():
    global LOADERS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--corpus", required=True, choices=sorted(LOADERS))
    ap.add_argument("--data-dir", default="/home/z/my-project/data/corpora")
    ap.add_argument("--tune-cap", type=int, default=30000,
                    help="valid tokens per tune trial (all variants equal)")
    ap.add_argument("--ds-grid", default="0.7,0.85,1.0,1.15,1.3",
                    help="discount-scale grid (comma-separated)")
    ap.add_argument("--z-grid", default="0.0,0.05,0.1,0.15,0.25,0.4",
                    help="Zipf prior ceiling grid (comma-separated)")
    ap.add_argument("--wt103-train-tokens", type=int, default=8_000_000)
    ap.add_argument("--wt103-eval-tokens", type=int, default=60_000)
    ap.add_argument("--quick", action="store_true",
                    help="smoke mode: tiny tune grids and caps")
    ap.add_argument("--variants", default="mkn,witten_bell,zipf_prior",
                    help="comma-separated subset of variants to run")
    args = ap.parse_args()

    if args.quick:
        args.tune_cap = 5000
        args.ds_grid = "1.0"
        args.z_grid = "0.0,0.15"

    args.ds_grid = [float(x) for x in args.ds_grid.split(",")]
    args.z_grid = [float(x) for x in args.z_grid.split(",")]

    if args.corpus == "wt103prefix":
        # thread the prefix sizes through a partial
        LOADERS = dict(LOADERS)
        LOADERS["wt103prefix"] = (
            lambda dd: load_wt103_prefix(dd, args.wt103_train_tokens,
                                         args.wt103_eval_tokens))

    run_corpus(args.corpus, args)


if __name__ == "__main__":
    main()
