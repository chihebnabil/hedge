"""
Gate-size sweep (Phase 6): GATE(6) test PP vs gate hidden size / KB.

Runs the 6-expert gate at h in {24, 64, 128, 256} on the corpus selected by
HEDGE_RESULTS (default WT-2), reporting valid + test PP and gate size.
Uses the same protocol as the money table (train-trained, 12 ep, wd 1e-4,
seed 0). Run for both corpora:

    HEDGE_RESULTS=benchmarks/results     python benchmarks/gate_sweep.py
    HEDGE_RESULTS=benchmarks/results/ptb python benchmarks/gate_sweep.py
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from gate_hybrid import load_split, train_gate, nll_pp, OUT  # noqa: E402


def main():
    corpus = "PTB" if OUT.rstrip("/").endswith("ptb") else "WT-2"
    (Ytr, Ltr) = load_split("train")
    (Yva, Lva) = load_split("valid")
    (Yte, Lte) = load_split("test")
    nexp = Ltr.shape[1]
    print(f"== gate-size sweep: {corpus}, GATE({nexp}) train-trained, "
          f"{len(Yte):,} test positions ==", flush=True)
    for h in (24, 64, 128, 256):
        pred, kb = train_gate(Ytr, Ltr, nexp, hidden=h)
        va = nll_pp(pred(Yva), Lva)
        te = nll_pp(pred(Yte), Lte)
        print(f"  h={h:3d}  gate {kb:5.1f} KB   valid {va:7.2f}   "
              f"test {te:7.2f}", flush=True)


if __name__ == "__main__":
    main()
