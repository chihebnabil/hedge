#!/usr/bin/env python3
"""Why per-token routing does not transfer: expert reliability is not
stationary between the training stream and the evaluation stream.

Prints, for each split, (a) each expert's own perplexity, (b) the router's
mean mixture weights, (c) the perplexity of the router, of its OWN MEAN
weights used statically, and of uniform weights, and (d) the statically tuned
mixture fit on train vs on valid.

The diagnostic that matters: if "PP of its mean alpha" is much better than the
router's own PP, the router's per-context modulation is what hurts — it has
learned train-specific trust in experts whose reliability collapses on held-out
text (the statistical experts memorize the training stream; the GRU does not).

    python benchmarks/router_diagnostic.py
    HEDGE_RESULTS=benchmarks/results/ptb python benchmarks/router_diagnostic.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from gate_hybrid import (OUT, load_split, nll_pp, selected_h,   # noqa: E402
                         train_static_alpha)

NAMES = ["kn3", "lag2", "cache", "lag3", "uni", "gru"]


def weights(X, W1, b1, W2, b2):
    h = np.maximum(X.astype(np.float64) @ W1 + b1, 0.0)
    z = h @ W2 + b2
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    corpus = "PTB" if OUT.rstrip("/").endswith("ptb") else "WT-2"
    h = selected_h(128)
    path = os.path.join(OUT, f"gate_mlp_h{h}.npz")
    data = {k: load_split(k) for k in ("train", "valid", "test")}
    nexp = data["train"][1].shape[1]
    names = NAMES[:nexp]

    if os.path.exists(path):
        z = np.load(path)
        W = (z["W1"], z["b1"], z["W2"], z["b2"])
        src = f"cached train-trained gate h={h}"
    else:
        W = None
        src = "no cached gate — static rows only"

    a_tr = train_static_alpha(data["train"][1], nexp)
    a_va = train_static_alpha(data["valid"][1], nexp)

    lines = [f"router diagnostic — {corpus} ({src})", ""]
    print(lines[-2], flush=True)
    for split in ("train", "valid", "test"):
        Y, L = data[split]
        one = np.eye(nexp)
        per = {n: nll_pp(np.repeat(one[i:i + 1], len(L), 0), L)
               for i, n in enumerate(names)}
        head = f"[{split}] {len(Y):,} positions"
        print("  " + head, flush=True)
        lines.append(head)
        row = "  expert PP      : " + "  ".join(
            f"{n}={per[n]:.1f}" for n in names)
        print(row, flush=True); lines.append(row)
        row = f"  uniform alpha  : {nll_pp(np.full((len(L), nexp), 1.0/nexp), L):.2f}"
        print(row, flush=True); lines.append(row)
        row = (f"  static train-fit: "
               f"{nll_pp(np.repeat(a_tr, len(L), 0), L):.2f}   alpha "
               + " ".join(f"{x:.2f}" for x in a_tr.ravel()))
        print(row, flush=True); lines.append(row)
        row = (f"  static valid-fit: "
               f"{nll_pp(np.repeat(a_va, len(L), 0), L):.2f}   alpha "
               + " ".join(f"{x:.2f}" for x in a_va.ravel()))
        print(row, flush=True); lines.append(row)
        if W is not None:
            A = weights(Y, *W)
            mean_a = A.mean(axis=0)
            r1 = f"  router PP      : {nll_pp(A, L):.2f}"
            r2 = ("  router's mean alpha used statically: "
                  f"{nll_pp(np.repeat(mean_a[None, :], len(L), 0), L):.2f}")
            r3 = ("  mean alpha     : "
                  + " ".join(f"{n}={v:.3f}" for n, v in zip(names, mean_a)))
            for r in (r1, r2, r3):
                print(r, flush=True); lines.append(r)
        lines.append("")

    lines.append("reading: if 'router PP' >> 'router's mean alpha used "
                 "statically', the per-context modulation is the problem, not "
                 "the average trust it learned; compare the expert PP rows "
                 "across splits to see the reliability shift it memorized.")
    print("  " + lines[-1])
    with open(os.path.join(OUT, "router_diagnostic.log"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {os.path.join(OUT, 'router_diagnostic.log')}")


if __name__ == "__main__":
    main()
