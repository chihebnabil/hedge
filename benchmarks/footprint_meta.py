"""Write nn_meta.json (unique GRU parameter count + expert pickles sizes) for
the active corpus, so every footprint number in the paper comes from one
measured source instead of a hardcoded constant.

    python benchmarks/footprint_meta.py
    HEDGE_RESULTS=benchmarks/results/ptb python benchmarks/footprint_meta.py
"""
import json
import os
import pickle
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import nn_expert                                     # noqa: E402
import mixer_lm                                       # noqa: E402
from gate_hybrid import OUT, COUNTS_PKL, BEST         # noqa: E402


def main():
    streams, mixer = mixer_lm.build(verbose=False)
    V = len(mixer.i2w) + 1
    m = nn_expert.GRULM(V)
    m.load_state_dict(torch.load(BEST))
    params = sum(p.numel() for p in m.parameters())      # dedupes tied weights
    e2, e3, e4, e5 = mixer.experts[1], mixer.experts[2], mixer.experts[3], mixer.experts[4]

    def pkl_bytes(obj):
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    meta = {
        "corpus": "PTB" if OUT.rstrip("/").endswith("ptb") else "WT-2",
        "vocab_gru": V,
        "params": params,
        "gru_mb_fp32": round(params * 4 / 1e6, 1),
        "gru_mb_fp16": round(params * 2 / 1e6, 1),
        "trigram_mb": round(os.path.getsize(COUNTS_PKL) / 1e6, 1),
        "lag2_mb": round(pkl_bytes({"t": dict(e2.totals), "c": dict(e2.counts)}) / 1e6, 1),
        "lag3_mb": round(pkl_bytes({"t": dict(e4.totals), "c": dict(e4.counts)}) / 1e6, 1),
        "cache_uni_mb": round(pkl_bytes({"c": dict(e5.p)}) / 1e6, 1),
    }
    meta["statistical_mb"] = round(meta["trigram_mb"] + meta["lag2_mb"]
                                   + meta["lag3_mb"] + meta["cache_uni_mb"], 1)
    meta["system_mb"] = round(meta["statistical_mb"] + meta["gru_mb_fp32"], 1)
    with open(os.path.join(OUT, "nn_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
