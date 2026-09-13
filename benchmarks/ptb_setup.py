"""
PTB setup for Phase 5 (cross-corpus replication of the money table).

    python benchmarks/ptb_setup.py gold    # gold streams -> results/ptb/
    python benchmarks/ptb_setup.py beast   # trigram backbone -> results/ptb/

Mirrors run_benchmark.py --stage data / --stage beast_train, but writes only
under benchmarks/results/ptb/ (WT-2 artifacts are never touched) and uses
common.load_ptb. Protocol differences from WT-2 are none: same tokenizer,
same vocab-from-train + OOV->unk rule, same trigram config.
"""

import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import common  # noqa: E402
from ZipfNextWordPredictor import ZipfNextWordPredictor  # noqa: E402

RES = os.environ.get("HEDGE_RESULTS", os.path.join(HERE, "results", "ptb"))
GOLD_PKL = os.path.join(RES, "gold_streams.pkl")
BEAST_PKL = os.path.join(RES, "beast_trigram.pkl")


def stage_gold():
    os.makedirs(RES, exist_ok=True)
    train, valid, test = common.load_ptb()
    with open(GOLD_PKL, "wb") as f:
        pickle.dump({"train": train, "valid": valid, "test": test}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  saved gold streams: train {sum(map(len, train)):,} tokens, "
          f"valid {sum(map(len, valid)):,}, test {sum(map(len, test)):,}")

    probe_lines = [" ".join(s) for s in train[:400]]
    probe = ZipfNextWordPredictor()
    flat = [w.lower() for s in probe.preprocess("\n".join(probe_lines))
            for w in s]
    assert flat == [w for s in train[:400] for w in s], "identity check FAILED"
    print("  tokenization identity check: OK")


def stage_beast():
    with open(GOLD_PKL, "rb") as f:
        g = pickle.load(f)
    train_lines = [" ".join(s) for s in g["train"]]
    t0 = time.perf_counter()
    m = ZipfNextWordPredictor(corpus_texts=train_lines, n_gram_size=3,
                              smoothing="beast")
    train_s = time.perf_counter() - t0
    info = m.model_info()
    print(f"  vocab={info['vocab']:,} "
          f"contexts={info['contexts_per_order']} "
          f"D={ {k: tuple(round(x, 3) for x in v) for k, v in info['discounts_per_order'].items()} }")
    m.save(BEAST_PKL)
    with open(os.path.join(RES, "backend_meta.json"), "w") as f:
        json.dump({"train_seconds": round(train_s, 2),
                   "vocab": info["vocab"],
                   "contexts": info["contexts_per_order"]}, f)
    print(f"  trained in {train_s:.1f}s, saved {BEAST_PKL} "
          f"({common.file_size(BEAST_PKL)/1e6:.1f} MB)")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "gold":
        stage_gold()
    elif stage == "beast":
        stage_beast()
    else:
        print(__doc__)
