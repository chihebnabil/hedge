#!/usr/bin/env python3
"""Selective GRU activation: the cost-quality frontier (realizable).

Motivation (oracle analysis, 2026-09-13): the GRU's value concentrates on a
minority of tokens — the selective-GRU oracle hits full oracle PP while
running the GRU on only ~50% of positions. This script measures the
REALIZABLE version:

  skip rule (fully causal): consult the GRU iff the trained 6-expert gate's
  own GRU weight alpha_gru(x) >= tau. The gate's input is the 5 cheap votes
  + 15 evidence features — it never reads the GRU's output, so the decision
  costs nothing and is available BEFORE the GRU runs.
    - skipped tokens: mixture = the gate's cheap weights renormalized
    - consulted tokens: full 6-expert gate mixture

Reported per threshold tau: test PP + GRU-usage fraction, against the
always-on gate and the selective oracle bound.

Usage:
  python benchmarks/selective_gru.py
  HEDGE_RESULTS=benchmarks/results/ptb python benchmarks/selective_gru.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gate_hybrid import load_split, train_gate, nll_pp

OUT = os.environ.get("HEDGE_RESULTS", "benchmarks/results")
H_SEL = {"ptb": 24}.get(os.path.basename(OUT.rstrip("/")), 128)
TAUS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)


def selective_pp(a6, L6, tau):
    """a6: (N,6) gate weights; consult GRU iff a6[:,5] >= tau."""
    consult = a6[:, 5] >= tau
    P = np.exp(L6.astype(np.float64))
    p_full = np.sum(a6 * P, axis=1)
    acheap = a6[:, :5] / np.maximum(a6[:, :5].sum(axis=1, keepdims=True), 1e-9)
    p_cheap = np.sum(acheap * P[:, :5], axis=1)
    p = np.where(consult, p_full, p_cheap)
    return math.exp(float(np.mean(-np.log(np.maximum(p, 1e-300))))), consult.mean()


def oracle_curve(L6):
    P = np.exp(L6.astype(np.float64))
    pc = P[:, :5].max(axis=1)
    adv = np.log(P[:, 5]) - np.log(pc)
    order = np.argsort(-adv)
    n = len(P)
    out = {}
    for q in (1.0, 0.5, 0.25, 0.10):
        sel = np.zeros(n, bool)
        sel[order[:int(n * q)]] = True
        p = np.where(sel, np.maximum(P[:, 5], pc), pc)
        out[q] = math.exp(float(np.mean(-np.log(p))))
    return out


def main():
    corpus = "PTB" if OUT.rstrip("/").endswith("ptb") else "WT-2"
    print(f"== selective GRU activation ({corpus}, realizable) ==")
    Y, L = {}, {}
    for k in ("train", "valid", "test"):
        Y[k], L[k] = load_split(k)

    pred5, kb5 = train_gate(Y["train"], L["train"][:, :5], nexp=5)
    print(f"  5-expert gate: test PP "
          f"{nll_pp(pred5(Y['test']), L['test'][:, :5]):.2f} ({kb5:.1f} KB)")

    pred6, kb6 = train_gate(Y["train"], L["train"], 6, hidden=H_SEL)
    va = nll_pp(pred6(Y["valid"]), L["valid"])
    pp6 = nll_pp(pred6(Y["test"]), L["test"])
    print(f"  6-expert gate (h={H_SEL}): valid {va:.2f}  test {pp6:.2f}")

    a6 = pred6(Y["test"])
    oc = oracle_curve(L["test"])
    lines = [f"selective GRU activation ({corpus} test) — realizable vs oracle",
             f"always-on 6-gate: {pp6:.2f} (GRU 100%)",
             f"oracle bound: 100% {oc[1.0]:.2f} | 50% {oc[0.5]:.2f} | "
             f"25% {oc[0.25]:.2f} | 10% {oc[0.10]:.2f}", ""]
    print("  tau | GRU usage | test PP")
    for tau in TAUS:
        pp, frac = selective_pp(a6, L["test"], tau)
        lines.append(f"tau={tau:.2f}: GRU on {frac*100:5.1f}% -> PP {pp:7.2f}")
        print(f"  {tau:.2f} | {frac*100:8.1f}% | {pp:7.2f}")
    with open(os.path.join(OUT, "selective_gru.log"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
