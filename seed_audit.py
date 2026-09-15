"""
seed_audit.py — 3-seed error bars + ensembles for the headline rows.

Audited (EXACT configs from gate_hybrid.py's money table):
  ROUTER, valid-tuned   h from gate_fair.json (2-fold CV on valid), 15 ep
                        — the headline row: the router gets the same data
                        budget as every other tuned baseline (validation only)
  static-alpha (6)      VALID-fit — the strongest fixed-weight baseline

Not audited: the train-trained router. It is a negative control, and its
instability is already the point — the 4-size sweep in the money table spans
250.8-263.6 PP on WT-2 (138.2-144.8 on PTB) at a single seed, and auditing
h=256 three times costs an hour to confirm a row we report as a failure mode.
fixed-lambda needs no audit (deterministic grid on valid).

Usage:
  python seed_audit.py              # 3 seeds
  python seed_audit.py --seeds 5    # more seeds
"""

import json
import os
import sys

import numpy as np

from gate_hybrid import (OUT, load_split, nll_pp, selected_h,
                         train_gate, train_static_alpha)


def audit(name, fit_fn, Lte, seeds):
    pps, As = [], []
    for s in range(seeds):
        A = fit_fn(s)
        As.append(A)
        pp = nll_pp(A, Lte)
        pps.append(pp)
        print(f"  [{name}] seed {s}: test PP {pp:.2f}", flush=True)
    m = float(np.mean(pps))
    sd = float(np.std(pps, ddof=1)) if seeds > 1 else 0.0
    print(f"  => {name}: {m:.2f} +/- {sd:.2f} ({seeds} seeds)", flush=True)
    if seeds > 1:
        ens = nll_pp(np.mean(As, axis=0), Lte)
        print(f"  => {name}: {seeds}-router ENSEMBLE PP {ens:.2f}", flush=True)
    return pps


def fair_h(default=64):
    """Hidden size the money table's CV picked (gate_fair.json)."""
    path = os.path.join(OUT, "gate_fair.json")
    if os.path.exists(path):
        with open(path) as f:
            return int(json.load(f)["h"])
    env = os.environ.get("HEDGE_GATE_H")
    return int(env) if env else default


if __name__ == "__main__":
    seeds = 3
    if "--seeds" in sys.argv:
        seeds = int(sys.argv[sys.argv.index("--seeds") + 1])

    data = {k: load_split(k) for k in ("train", "valid", "test")}
    (Ytr, Ltr), (Yva, Lva), (Yte, Lte) = (data["train"], data["valid"],
                                          data["test"])
    nexp = Ltr.shape[1]
    h = fair_h()

    print(f"== seed audit: ROUTER valid-tuned h={h} (CV-selected), 15 ep ==",
          flush=True)
    audit("router-valid", lambda s: train_gate(
        Yva, Lva, nexp, hidden=h, epochs=15, seed=s)[0](Yte), Lte, seeds)

    print("== seed audit: static-alpha (6 experts, VALID-fit) ==", flush=True)
    audit("static-alpha-valid", lambda s: np.repeat(
        train_static_alpha(Lva, nexp, seed=s), len(Yte), 0), Lte, seeds)

    print("== seed audit: static-alpha (6 experts, train-fit) ==", flush=True)
    audit("static-alpha-train", lambda s: np.repeat(
        train_static_alpha(Ltr, nexp, seed=s), len(Yte), 0), Lte, seeds)

    print(f"== context: the train-trained router (valid-selected "
          f"h={selected_h(128)}) is a single-seed negative control; see the "
          f"4-size sweep in money_table.log ==", flush=True)
