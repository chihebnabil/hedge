# Hedge 🦔
### One tamer, many beasts — a hybrid expert-mixture language model
*(built on the ZipfNextWordPredictor base — "BEAST" v5)*

A study of how far **tiny learned routing** can push frozen cheap experts:
five 1990s-style statistical experts plus one small GRU, blended per-token by
a 2.6–13.5 KB neural gate — **beating tuned classical interpolation on two
corpora at a footprint neural systems don't even report.**

**The one-sentence story:** a ~13 KB learned gate that reads six experts'
votes plus cheap context evidence cuts WikiText-2 perplexity from 199.5
(statistical team alone) to **90.3**, dominating tuned fixed interpolation
(135.8) — and the result replicates on Penn Treebank (**68.0 vs 92.7**).

---

## Findings (2 corpora, seed-audited, all reproducible)

### Money table — the paper's load-bearing comparison
| model | WT-2 test PP | PTB test PP |
|---|---|---|
| pure KN3 trigram | 296.2 | 165.2 |
| pure GRU (aligned) | 146.9 | 103.0 |
| 5-expert gate (no GRU) | 99.7 | 72.7 |
| fixed-λ KN3+GRU (tuned on valid) | 135.8 | 92.7 |
| static-α (6 experts) | 171.0 | 101.7 |
| **GATE(6) — learned per-token routing** | **90.3** | **68.0** |
| GATE(6), valid-trained | 77.5 | 63.9 |
| hindsight oracle (bound) | 57.4 | 44.4 |

- **The gate beats fixed interpolation on both corpora, at every tested
  seed** (WT-2 worst seed 125.6 < 135.8; PTB worst seed 86.1 < 92.7).
  3-seed error bars + ensembles in [PAPER_NOTES.md](PAPER_NOTES.md) §8–9.
- **Static mixing loses to everything learned** (171.0 / 101.7 ± 0.1): the
  win is *per-token* routing, not the extra parameters.
- Honest wrinkle, reported: on the smaller corpus the train-trained h=128
  gate overfits (78.8, worse than the 5-expert gate); the valid-selected
  2.6 KB gate (68.0) and the valid-trained gate (63.9) don't.

### The routing signal is the votes×evidence interaction
Ablations: votes alone 201.8 · evidence alone 367.9 (worse than equal
mixing!) · both → 90–100. The evidence tells the gate *whose* vote to trust.

### The "novel" prior is decorative (negative result, rigorously verified)
The base repo's headline (−9.8 PP from a Zipf-continuation prior) does not
survive controls (Phase 0 in [PROGRESS.md](PROGRESS.md)); the finding
replicates negatively across four corpora (`benchmarks/cross_corpus/`).

### The 5-gram "anomaly" was a tuning artifact
Fair-tuned with the trigram's grid, the 5-gram lands at 294.2 (was 302.5):
order-5 buys ≈nothing on WT-2 — classical n-gram scaling is exhausted, and
the hybrid beats the whole family ~3×. (`benchmarks/fix_5gram.py`)

### Cost: the router is (almost) free
Full hybrid: 3.6 ms/token (WT-2) / 1.0 ms/token (PTB) on 4 CPU cores
(~280 tok/s interactive); the gate's share ≤8% vs fixed-λ — identical cost,
−27% to −34% PP. Sizes: 69 MB (WT-2) / 29 MB (PTB) total.
(`benchmarks/energy_harness.py`; method + tables in PAPER_NOTES §10.)

---

## Quick tour

```bash
# the trained system, interactively:
python hybrid_cli.py suggest "the united states"     # what the model believes
python hybrid_cli.py cloze "<paste real text>"       # hit-rate test (the honest test)
python hybrid_cli.py why america the united states   # corpus counts behind a prediction
python hybrid_cli.py                                  # REPL: suggest/complete/cloze/why/observe

# benchmarks (first run trains+caches the gate):
python gate_hybrid.py          # the money table (WT-2)
python seed_audit.py           # 3-seed ±sd + ensembles
python benchmarks/gate_sweep.py                # gate size vs quality
python benchmarks/energy_harness.py            # ms/token + joule proxy
```

## Reproduce from scratch

Everything regenerates from scripts in a fresh Codespace (data and model
artifacts are gitignored). Full protocols and validated numbers:
[PAPER_NOTES.md](PAPER_NOTES.md); live experiment log: [PROGRESS.md](PROGRESS.md).

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# ---------- WikiText-2 ----------
python benchmarks/run_benchmark.py --stage data && \
python benchmarks/run_benchmark.py --stage beast_train   # gold streams + trigram
python -u nn_expert.py                    # GRU expert (~7 min/epoch, resumable)
python nn_expert.py --score-only          # aligned scores
python gate_lm.py                         # gate features + Phase-2 pipeline
python gate_hybrid.py && python seed_audit.py

# ---------- Penn Treebank (Mikolov split; auto-download below) ----------
for s in train valid test; do curl -sL -o data/ptb/ptb.$s.txt \
  https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.$s.txt; done
python benchmarks/ptb_setup.py gold && python benchmarks/ptb_setup.py beast
HEDGE_RESULTS=benchmarks/results/ptb HEDGE_WARMUP=442 python -u nn_expert.py
HEDGE_RESULTS=benchmarks/results/ptb python nn_expert.py --score-only
HEDGE_RESULTS=benchmarks/results/ptb python gate_lm.py
HEDGE_RESULTS=benchmarks/results/ptb python gate_hybrid.py
HEDGE_RESULTS=benchmarks/results/ptb HEDGE_GATE_H=24 python seed_audit.py
```

`HEDGE_RESULTS` redirects every script's read/write root (WT-2 artifacts
live in `benchmarks/results/`, PTB's in `benchmarks/results/ptb/`).
No per-corpus hyperparameters except gate hidden size, selected on valid
(h=128 WT-2, h=24 PTB) and the GRU warmup length (one epoch each).

Long runs are suspend-proof: `scripts/auto_resume.sh` (wired via
`postStartCommand`) relaunches an interrupted GRU run from its checkpoint
whenever the Codespace container restarts.

## Honest scope

- Frozen experts dominate the footprint; the *learned* part is the tiny
  gate. The claim is about the **tiny-footprint regime** (on-device,
  offline settings) and about *measuring* what learned routing extracts —
  not about competing with large transformers.
- Novelty positioning: mixture-of-experts routing is classical (Jacobs &
  Jordan 1991; Shazeer 2017); this work tests whether *learned per-token*
  routing changes the conclusion of the *fixed-weight* hybrid interpolation
  literature (Mikolov-era) in the small-footprint regime. See §5 of
  [PAPER_NOTES.md](PAPER_NOTES.md) for the locked positioning contract.
- CPU-only training is a feature of the story (green computing), not a
  secret. Negative results are first-class citizens here (Phase 0, GRU
  recipe v1/v2, the unbounded-discount knob warning).

## The base repo (v5, preserved)

A from-scratch modified Kneser-Ney trigram in pure standard-library Python
with a Zipf-continuation prior, personalization user layer, CLI, and a
fixed-protocol benchmark suite. Its published results reproduce exactly
(see `RESULTS.md`); its novel prior's contribution is analyzed honestly in
Phase 0 of `PROGRESS.md`. Engines: `"beast"` (default),
`"modified_kneser_ney"`, `"kneser_ney"`, `"zipf"` (legacy v4).

```python
from ZipfNextWordPredictor import ZipfNextWordPredictor

model = ZipfNextWordPredictor(corpus_texts=[...], n_gram_size=3)
model.predict_next_words("once upon a")
model.observe("user text")          # personalization
model.tune(heldout)                 # self-calibration
```

`python example.py` for the guided demo · `python cli.py chat -m model.pkl`
for the REPL · `python -m unittest tests.test_model` for the test suite.
