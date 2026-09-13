r"""Shared benchmark utilities: WikiText-2 loading, shared tokenization,
<unk> mapping, timing and size measurement.

Benchmark protocol
------------------
* Corpus: WikiText-2 (word level, pre-tokenized by Salesforce).
* Header lines ("= Title =") and blank lines are dropped.
* The gold token stream for ALL models is produced by the predictor's own
  tokenizer (re.findall(r"\b\w+\b|[.,!?;]"), lowercased), so every model —
  baselines included — is scored on the exact same stream.
* Tokens unseen in the training stream are mapped to "unk" (the tokenized
  form of WikiText's literal "<unk>") in valid/test.
* All models see the same training text; models that consume token streams
  get the gold stream, models that consume raw text get the exact
  re-joined gold stream (identity-checked in run_benchmark).
"""

import gzip
import json
import os
import pickle
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ZipfNextWordPredictor import ZipfNextWordPredictor  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data")

WT2_URLS = [
    "https://wikitext.smerity.com/wikitext-2-v1.zip",
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
]

_HEADER_RE = re.compile(r"^\s*=.*=\s*$")

# Exact replica of the predictor's tokenization (simple_sent_tokenize +
# preprocess, lowercased): the shared gold stream for every model.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")
_TOKEN_RE = re.compile(r"\b\w+\b|[.,!?;]")


def tokenize_line(line):
    """One raw line -> list of lowercase token sentences (predictor-compatible)."""
    out = []
    for s in _SENT_SPLIT_RE.split(line):
        if s:
            words = [w.lower() for w in _TOKEN_RE.findall(s)]
            if words:
                out.append(words)
    return out


def _fetch(url, zip_path):
    """Download with a browser User-Agent (bare urllib gets Cloudflare-blocked)."""
    import shutil
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(zip_path, "wb") as f:
        shutil.copyfileobj(r, f)


def ensure_wikitext2(data_dir=DATA_DIR):
    """Return paths to the three WikiText-2 token files, downloading if needed."""
    wt2 = os.path.join(data_dir, "wikitext-2")
    paths = {split: os.path.join(wt2, f"wiki.{split}.tokens")
             for split in ("train", "valid", "test")}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "wikitext-2-v1.zip")
    if not os.path.exists(zip_path):
        last_err = None
        for url in WT2_URLS:
            try:
                print(f"  downloading {url} ...")
                _fetch(url, zip_path)
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        if last_err:
            raise RuntimeError(
                f"Could not download WikiText-2: {last_err}\n"
                f"Manual fallback:\n"
                f"  curl -L -A 'Mozilla/5.0' -o {zip_path} {WT2_URLS[0]}")
    import zipfile
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    return paths


def load_wikitext2(data_dir=DATA_DIR):
    """
    Returns (train_sents, valid_sents, test_sents) where each sentence is a
    list of lowercase tokens sharing the predictor's tokenization, with
    valid/test OOV mapped to "unk".
    """
    paths = ensure_wikitext2(data_dir)

    def raw_lines(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or _HEADER_RE.match(line):
                    continue
                yield line

    def tokenize(line):
        return tokenize_line(line)

    def stream(path):
        out = []
        for line in raw_lines(path):
            out.extend(tokenize(line))
        return out

    print("  tokenizing train ...")
    train = stream(paths["train"])
    vocab = set()
    for sent in train:
        vocab.update(sent)

    unk = "unk"
    print("  tokenizing valid/test (+unk mapping) ...")
    valid, test = [], []
    for split, store in (("valid", valid), ("test", test)):
        for sent in stream(paths[split]):
            store.append([w if w in vocab else unk for w in sent])

    return train, valid, test


def load_ptb(data_dir=DATA_DIR):
    """
    Returns (train_sents, valid_sents, test_sents) with the predictor's
    tokenization, valid/test OOV mapped to "unk" (Mikolov PTB split,
    data/ptb/ptb.{split}.txt; one sentence per line, pre-tokenized).
    Note: the predictor tokenizer re-splits PTB tokens (e.g. "u.s." -> two
    tokens), so vocab and streams follow OUR tokenizer, not PTB's raw one.
    """
    ptb = os.path.join(data_dir, "ptb")
    paths = {split: os.path.join(ptb, f"ptb.{split}.txt")
             for split in ("train", "valid", "test")}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"PTB files missing: {missing}\n"
                           "Fetch the Mikolov split (tomsercu/lstm): "
                           "curl -L -o data/ptb/ptb.train.txt "
                           "https://raw.githubusercontent.com/tomsercu/lstm"
                           "/master/data/ptb.train.txt (and valid/test)")

    print("  tokenizing train ...")
    train = []
    with open(paths["train"], encoding="utf-8") as f:
        for line in f:
            train.extend(tokenize_line(line))
    vocab = set()
    for sent in train:
        vocab.update(sent)

    unk = "unk"
    print("  tokenizing valid/test (+unk mapping) ...")
    valid, test = [], []
    for split, store in (("valid", valid), ("test", test)):
        with open(paths[split], encoding="utf-8") as f:
            for line in f:
                for sent in tokenize_line(line):
                    store.append([w if w in vocab else unk for w in sent])

    return train, valid, test


def streams_to_text(sents):
    """Re-join token sentences into raw text the predictor tokenizes back
    identically."""
    return "\n".join(" ".join(sent) for sent in sents)


def file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def compressed_size_bytes(obj):
    """zlib-compressed pickle size (fair 'shipped size' metric)."""
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return len(raw), len(gzip.compress(raw, compresslevel=6))


def evaluate_harness(logprob_fn, topk_fn, sents, max_ctx=2, ks=(1, 3)):
    """
    Score a baseline model on gold token sentences.
    logprob_fn(target, ctx_tuple) -> float probability
    topk_fn(ctx_tuple, k) -> ranked list of words
    """
    hits = {k: 0 for k in ks}
    tokens = 0
    log_sum = 0.0
    t0 = time.perf_counter()
    for sent in sents:
        words = sent  # already lowercase
        for i in range(1, len(words)):
            ctx = tuple(words[max(0, i - max_ctx):i])
            target = words[i]
            p = logprob_fn(target, ctx)
            log_sum += __import__("math").log(max(p, 1e-12))
            for k in ks:
                if target in topk_fn(ctx, k):
                    hits[k] += 1
            tokens += 1
    dt = time.perf_counter() - t0
    out = {"tokens": tokens, "eval_seconds": dt}
    for k in ks:
        out[f"top{k}_accuracy"] = hits[k] / tokens if tokens else 0.0
    out["perplexity"] = float("inf") if tokens == 0 else \
        __import__("math").exp(-log_sum / tokens)
    return out


def json_dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
