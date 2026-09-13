#!/usr/bin/env python3
"""KenLM toolkit baseline under OUR aligned protocol.

Purpose (paper limitations item): compare our statistical experts against
a toolkit-grade modified-KN n-gram (KenLM defaults, no tuning) on the exact
gold streams and aligned positions used by the money table.

Protocol:
- Text: the gold token streams re-joined one sentence per line (the same
  streams every other system scores; unk-mapping already applied).
- Train: lmplz -o {3,4,5} on the train dump (default discounting).
- Score: kenlm full_scores per line, drop each line's first word (the
  sentence-initial target our protocol excludes; eos separators are never
  scored), sum natural-log probs.
- Rows: must equal sum(len(s)-1) per split (the gate npz row count).

Usage:
  python benchmarks/kenlm_baseline.py            # WT-2 (default results root)
  HEDGE_RESULTS=benchmarks/results/ptb python benchmarks/kenlm_baseline.py
"""
import math
import os
import pickle
import subprocess
import sys
import tempfile

RES = os.environ.get("HEDGE_RESULTS", "benchmarks/results")
CORPUS = "ptb" if RES.rstrip("/").endswith("ptb") else "wt2"
ORDERS = (3, 4, 5)


def main():
    with open(os.path.join(RES, "gold_streams.pkl"), "rb") as f:
        streams = pickle.load(f)

    expect = {k: sum(len(s) - 1 for s in v) for k, v in streams.items()}
    print(f"[{CORPUS}] sentences train/valid/test:",
          *[len(streams[k]) for k in ("train", "valid", "test")])

    tmp = tempfile.mkdtemp(prefix="kenlm_")
    paths = {}
    for split, sents in streams.items():
        p = os.path.join(tmp, f"{split}.txt")
        with open(p, "w") as f:
            f.write("\n".join(" ".join(s) for s in sents) + "\n")
        paths[split] = p

    results = {}
    for o in ORDERS:
        arpa = os.path.join(tmp, f"model{o}.arpa")
        subprocess.run(
            [find_lmplz(), "-o", str(o), "-S", "40%", "--text", paths["train"],
             "--arpa", arpa], check=True, stdout=subprocess.DEVNULL)
        import kenlm
        m = kenlm.Model(arpa)
        row = {}
        for split in ("train", "valid", "test"):
            n, s10 = 0, 0.0
            with open(paths[split]) as f:
                for line in f:
                    for i, (lp, _, _) in enumerate(m.full_scores(line, eos=False)):
                        if i == 0:
                            continue                     # sentence-initial
                        s10 += lp
                        n += 1
            assert n == expect[split], (split, n, expect[split])
            row[split] = math.exp(-s10 * math.log(10) / n)
        results[o] = row
        mb = os.path.getsize(arpa) / 1e6
        print(f"  {o}-gram  train {row['train']:.2f}  valid {row['valid']:.2f}"
              f"  test {row['test']:.2f}  ({mb:.1f} MB arpa)")

    with open(os.path.join(RES, "kenlm.log"), "w") as f:
        f.write(f"KenLM baseline (lmplz defaults), aligned protocol, "
                f"{CORPUS.upper()}\n")
        for o, row in results.items():
            f.write(f"{o}-gram: train {row['train']:.2f} valid {row['valid']:.2f}"
                    f" test {row['test']:.2f}\n")


def find_lmplz():
    for cand in ("lmplz", os.path.expanduser("~/.local/bin/lmplz")):
        if subprocess.run(["which", cand], capture_output=True).returncode == 0:
            return cand
    sys.exit("lmplz not found: build KenLM tools "
             "(mkdir build && cmake .. && make lmplz) or `pip install kenlm` "
             "only gives the scorer")


if __name__ == "__main__":
    main()
