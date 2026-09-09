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
import time

import numpy as np
import torch
import torch.nn as nn

from mixer_lm import GOLD, TRI_PKL
from ZipfNextWordPredictor import ZipfNextWordPredictor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "benchmarks", "results")
EPOCHS = 6
BS = 32
TSL = 64
EMB = 256
HID = 256
LR = 3e-3
CLIP = 0.25


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
    return stream("train"), stream("valid"), stream("test"), V


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
    non-eos position in stream order (== gate npz row order)."""
    model.eval()
    x_all = torch.from_numpy(arr)
    out = np.empty(len(arr) - 1, dtype=np.float32)
    h = None
    CH = 256
    with torch.no_grad():
        for s in range(0, len(arr) - 1, CH):
            k = min(CH, len(arr) - 1 - s)
            x = x_all[s:s + k].view(1, -1).to(device)
            logits, h = model(x, h)
            lp = torch.log_softmax(logits, dim=-1).squeeze(0).cpu()
            tgt = x_all[s + 1:s + 1 + k]
            val = lp.gather(1, tgt.unsqueeze(1)).squeeze(1).numpy()
            keep = tgt.numpy() != EOS
            out[s:s + len(tgt)][keep] = val[keep]
    return out


EOS = None


def main():
    global EOS
    t0 = time.time()
    tr, va, te, V = load_streams()
    EOS = V - 1
    device = "cpu"
    torch.manual_seed(42)
    model = GRULM(V).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    data = batchify(tr, BS)
    steps_per_epoch = (data.shape[1] - 1) // TSL
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS *
                                                       steps_per_epoch)
    ckpt = os.path.join(OUT, "nn_gru_best.pt")
    start_ep = 0
    best_pp = 1e9
    best_state = None
    if os.path.exists(ckpt):
        best_state = torch.load(ckpt)
        model.load_state_dict(best_state)
        best_pp = evaluate(model, va, crit, device)
        print(f"resuming: loaded checkpoint with valid PP={best_pp:.2f}",
              flush=True)
    print(f"V={V}  train tok={len(tr):,}  steps/epoch={steps_per_epoch}  "
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
        print(f"epoch {ep+1}/{EPOCHS}: train NLL {tot/n:.4f} "
              f"(PP {math.exp(tot/n):.2f})  valid PP {vpp:.2f}  "
              f"[{time.time()-te0:.0f}s, total {time.time()-t0:.0f}s]",
              flush=True)
        if vpp < best_pp:
            best_pp = vpp
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt)
            print("  -> new best, checkpointed", flush=True)

    model.load_state_dict(torch.load(ckpt))
    vpp = evaluate(model, va, crit, device)
    print(f"best model valid PP={vpp:.2f}", flush=True)
    for split, arr in (("valid", va), ("test", te)):
        lp = score_aligned(model, arr, device)
        np.save(os.path.join(OUT, f"nn_logp_{split}.npy"), lp)
        print(f"saved nn_logp_{split}.npy  n={len(lp):,}", flush=True)
    print("pure-GRU test PP (next step evaluates it against the gate npz):",
          flush=True)
    print(f"  {evaluate(model, te, crit, device):.2f}", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
