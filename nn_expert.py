"""
nn_expert.py — Stage 3: train a GRU language model on WikiText-2 (torch,
CPU) to serve as expert #7 in the mixer.

Outputs:
  benchmarks/results/nn_gru_best.pt         best checkpoint (by valid PP)
  benchmarks/results/nn_logp_valid.npy      per-position log p(w) aligned
  benchmarks/results/nn_logp_test.npy       with gate_*.npz row order

Protocol notes:
  - ids come from the trigram pickle's _w2i (must match the other experts);
    one extra id (V) is reserved for the eos separator.
  - the GRU streams across sentence boundaries (causal) — a legal expert
    that simply uses more context.
  - per-epoch valid PP is printed and checkpointed; best model is kept.
"""

import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from mixer_lm import GOLD, TRI_PKL
from ZipfNextWordPredictor import ZipfNextWordPredictor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HEDGE_RESULTS",
                     os.path.join(HERE, "benchmarks", "results"))
EPOCHS = 30
BS = 32
TSL = 64
EMB = 256
HID = 256
LR = 1e-3
CLIP = 1.0
WARMUP_STEPS = int(os.environ.get("HEDGE_WARMUP", "980"))
# WT-2 default: one epoch of linear warmup, then cosine decay


class GRULM(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, EMB)
        self.gru = nn.GRU(EMB, HID, num_layers=2,
                          dropout=0.25, batch_first=True)
        self.drop = nn.Dropout(0.35)
        self.dec = nn.Linear(HID, V)
        self.dec.weight = self.emb.weight                  # tied
        for p in self.parameters():
            nn.init.uniform_(p, -0.05, 0.05)
        for name in ("bias_ih_l0", "bias_hh_l0", "bias_ih_l1", "bias_hh_l1"):
            nn.init.zeros_(getattr(self.gru, name))

    def forward(self, x, h):
        e = self.emb(x)
        o, h = self.gru(e, h)
        return self.dec(self.drop(o)), h


def load_streams():
    with open(GOLD, "rb") as f:
        streams = pickle.load(f)
    tri = ZipfNextWordPredictor.load(TRI_PKL)
    w2i = tri._w2i
    unk = w2i["unk"]
    V = len(tri._i2w) + 1                               # +1 for eos
    def stream(split):
        out = []
        for sent in streams[split]:
            out.extend(w2i.get(w, unk) for w in sent)
            out.append(V - 1)                           # eos
        return np.asarray(out, dtype=np.int64)
    return (stream("train"), stream("valid"), stream("test"), V, streams)


def gate_row_count(streams, split):
    """Positions the gate scores: every word except each sentence's first."""
    return sum(max(len(s) - 1, 0) for s in streams[split])


def batchify(arr, bs):
    n = (len(arr) // bs) * bs
    return torch.from_numpy(arr[:n].reshape(bs, -1))


def evaluate(model, arr, crit, device):
    data = batchify(arr, BS)
    h = None
    tot, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, data.shape[1] - 1, TSL):
            k = min(TSL, data.shape[1] - 1 - s)
            if k < 1:
                break
            x = data[:, s:s + k].to(device)
            y = data[:, s + 1:s + 1 + k].to(device)
            logits, h = model(x, h)
            loss = crit(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            tot += float(loss.detach()) * y.numel()
            n += y.numel()
    return math.exp(tot / n)


def score_aligned(model, arr, device):
    """Stream bs=1 with carried hidden state; record log p(token) for every
    target that is non-eos AND not sentence-initial (predecessor is not eos).
    Gate collection (gate_lm.py) skips each sentence's first word and never
    scores eos, so kept rows == sum(len(sent)-1) == the gate npz row order."""
    model.eval()
    x_all = torch.from_numpy(arr)
    tgt_all = arr[1:]
    prev_all = arr[:-1]
    keep_all = (tgt_all != EOS) & (prev_all != EOS)
    n_keep = int(keep_all.sum())
    out = np.empty(n_keep, dtype=np.float32)
    h = None
    CH = 256
    w = 0
    with torch.no_grad():
        for s in range(0, len(arr) - 1, CH):
            k = min(CH, len(arr) - 1 - s)
            x = x_all[s:s + k].view(1, -1).to(device)
            logits, h = model(x, h)
            lp = torch.log_softmax(logits, dim=-1).squeeze(0).cpu()
            tgt = x_all[s + 1:s + 1 + k]
            val = lp.gather(1, tgt.unsqueeze(1)).squeeze(1).numpy()
            keep = keep_all[s:s + k]
            m = int(keep.sum())
            out[w:w + m] = val[keep]
            w += m
    assert w == n_keep
    return out


EOS = None


def main(score_only=False):
    global EOS
    t0 = time.time()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    tr, va, te, V, streams = load_streams()
    EOS = V - 1
    device = "cpu"
    torch.manual_seed(42)
    model = GRULM(V).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    data = batchify(tr, BS)
    steps_per_epoch = (data.shape[1] - 1) // TSL
    total_steps = EPOCHS * steps_per_epoch

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        p = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ckpt = os.path.join(OUT, "nn_gru_best.pt")
    last = os.path.join(OUT, "nn_gru_last.pt")
    start_ep = 0
    best_pp = 1e9
    if score_only and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        best_pp = evaluate(model, va, crit, device)
        print(f"score-only: loaded best checkpoint, valid PP={best_pp:.2f}",
              flush=True)
        start_ep = EPOCHS
    elif os.path.exists(last):
        snap = torch.load(last)
        model.load_state_dict(snap["model"])
        opt.load_state_dict(snap["opt"])
        sched.load_state_dict(snap["sched"])
        start_ep = snap["epoch"]
        best_pp = snap["best_pp"]
        print(f"resuming: epoch {start_ep}/{EPOCHS}, "
              f"best valid PP so far {best_pp:.2f}", flush=True)
    print(f"V={V}  train tok={len(tr):,}  steps/epoch={steps_per_epoch}  "
          f"threads={torch.get_num_threads()}  "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M",
          flush=True)

    for ep in range(start_ep, EPOCHS):
        model.train()
        h = None
        tot, n = 0.0, 0
        te0 = time.time()
        for s in range(0, data.shape[1] - 1, TSL):
            x = data[:, s:s + TSL]
            y = data[:, s + 1:s + TSL + 1]
            if x.shape[1] < TSL:
                break
            opt.zero_grad()
            logits, h = model(x, h)
            loss = crit(logits.reshape(-1, V), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            sched.step()
            with torch.no_grad():
                h = h.detach()
            tot += float(loss.detach()) * y.numel()
            n += y.numel()
        vpp = evaluate(model, va, crit, device)
        with torch.no_grad():
            probe, _ = model(data[:2, :TSL], None)
            lmax = float(probe.abs().max())
        print(f"epoch {ep+1}/{EPOCHS}: train NLL {tot/n:.4f} "
              f"(PP {math.exp(tot/n):.2f})  valid PP {vpp:.2f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  max|logit| {lmax:.1f}  "
              f"[{time.time()-te0:.0f}s, total {time.time()-t0:.0f}s]",
              flush=True)
        if vpp < best_pp:
            best_pp = vpp
            torch.save({k: v.clone() for k, v in model.state_dict().items()},
                       ckpt)
            print("  -> new best, checkpointed", flush=True)
        torch.save({"epoch": ep + 1, "model": model.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best_pp": best_pp}, last)

    model.load_state_dict(torch.load(ckpt))
    print("ALL EPOCHS DONE", flush=True)
    vpp = evaluate(model, va, crit, device)
    print(f"best model valid PP={vpp:.2f}", flush=True)
    for split, arr in (("train", tr), ("valid", va), ("test", te)):
        lp = score_aligned(model, arr, device)
        exp = gate_row_count(streams, split)
        assert len(lp) == exp, \
            f"{split}: {len(lp):,} logp rows != gate {exp:,}"
        np.save(os.path.join(OUT, f"nn_logp_{split}.npy"), lp)
        print(f"saved nn_logp_{split}.npy  n={len(lp):,} "
              f"(== gate rows)", flush=True)
    print("pure-GRU test PP (next step evaluates it against the gate npz):",
          flush=True)
    print(f"  {evaluate(model, te, crit, device):.2f}", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(score_only="--score-only" in sys.argv)
