# Hedge 🦔
### One tamer, many beasts — a hybrid expert-mixture language model
*(built on the [ZipfNextWordPredictor](https://github.com/chihebnabil/ZipfNextWordPredictor)
base — "BEAST" v5; MIT-licensed, see [LICENSE](LICENSE))*

A budget-matched study of how far **tiny learned routing** can push frozen cheap
experts: five 1990s-style statistical experts plus one small GRU on
WikiText-2 and the Penn Treebank, under one aligned-scoring protocol.

**The short version.** Per-token routing helps, but only by **1–2%** over a
*tuned static mixture of the same six experts*, and by **5–13%** over classical
two-expert interpolation. An earlier version of this repo claimed **27–33%**.
That claim was an artifact: the router's "vote" features were the experts'
probabilities **of the word being scored**, so its mixture was not a
distribution (it summed to 2.4–2.7 over the vocabulary instead of 1) and every
router perplexity was deflated by that factor. The audit that catches this class
of bug is one line, it is in CI, and the invalidated artifacts are kept in
[`benchmarks/results/*/legacy_leaky_v1/`](benchmarks/results) with the code
revision that produced them.

---

## Findings (2 corpora, seed-audited, all reproducible)

### Main table — every tuned component gets validation data and nothing else

| model | WT-2 test PP | PTB test PP | WT-2 MB | PTB MB |
|---|---|---|---|---|
| pure KN3 trigram | 296.2 | 165.2 | 21.4 | 9.2 |
| KenLM 5-gram (toolkit defaults) | 250.9 | 138.4 | — | — |
| pure GRU (aligned) | 146.9 | 103.0 | 32.7 | 13.1 |
| uniform mixture (6 experts) | 182.6 | 138.4 | 68.7 | 28.7 |
| fixed-λ KN3+GRU (λ\* on valid) | 135.8 | 92.7 | 54.0 | 22.3 |
| static-α (6 experts), fit on train | 171.0 | 101.7 | 68.7 | 28.7 |
| **static-α (6 experts), fit on valid** | **119.1** | **89.7** | 68.7 | 28.7 |
| router, train-trained (negative control) | 250.8 | 138.2 | 68.7 | 28.7 |
| **router, valid-tuned (h=128 by CV, 15.5 KB)** | **117.9** | **87.8** | 68.7 | 28.7 |
| router over the 5 statistical experts, valid-tuned | 194.5 | 132.8 | 36.0 | 15.6 |
| best single expert in hindsight (normalized) | 146.9 | 103.0 | — | — |
| per-position best expert (*not* normalized) | (57.4) | (44.4) | — | — |
| ~~v1 leaky router (invalidated)~~ | ~~90.3~~ | ~~68.0~~ | — | — |

Aligned positions: 217,004 (WT-2) / 75,623 (PTB). Footprints measured by
`benchmarks/footprint_meta.py` (GRU = unique parameters, embeddings tied).
KenLM footprints are not quoted: only `lmplz` is installed here, so the
query-time binary size is not measurable in this environment (the ARPA size is
logged in `kenlm.log`).

- **The router's real margin is small.** 117.9 vs 119.1 (WT-2, −1.1%) and 87.8
  vs 89.7 (PTB, −2.2%, seed sd 0.06 over 3 seeds). Most of the gain over
  classical interpolation comes from *having six experts and tuning their
  global weights on held-out data*, not from reweighting per token.
- **Training the router on train is a trap.** The same architecture on the
  13× larger training stream scores 250.8 / 138.2 — worse than *uniform*
  mixing (182.6 / 138.4) and worse than every tuned static baseline.
  Mechanism (`benchmarks/router_diagnostic.py`): expert reliability is not
  stationary. On PTB the trigram goes 34.7 → 165.1 PP from its own training
  stream to test while the GRU goes 46.6 → 103.0, so a train-fit router learns
  trust that does not transfer. Freezing the router's *own mean weights* and
  applying them statically already recovers most of the loss (138.2 → 105.4):
  the per-context modulation is what mis-transfers.
- **Selective GRU activation does not survive the correction.** The router's
  GRU weight is context-only, so thresholding it is a free causal skip rule —
  but the corrected router keeps the GRU on for 99.0% (WT-2) / 89.7% (PTB) of
  tokens even at τ=0.5. The v1 "GRU on 47–55% of tokens for ≤0.5 PP" frontier
  was the same leak.

### The normalization audit (the reusable part)

A gated mixture is a language model only if its weights are a function of the
context. If the gate reads anything about the scored word, then
`Z(ctx) = Σ_w Σ_i a_i(x_{ctx,w}) p_i(w|ctx) ≠ 1` and the "perplexity" is
deflated by ≈`exp(mean log Z)`. Nothing in training signals this: the loss goes
down, validation looks healthy, and the model can even beat a "hindsight
oracle" that has the same defect.

| router | reported PP | vocabulary mass Z | normalized PP |
|---|---|---|---|
| WT-2 v1 (leaky) | 90.3 | 2.68 | ≈242 |
| PTB v1 (leaky) | 68.0 | 2.44–2.56 | ≈166–174 |
| KN3 / GRU / fixed-λ / static-α | — | 1.000 | — |
| WT-2 / PTB corrected causal router | 117.9 / 87.8 | 1.000 | same |

```bash
python benchmarks/normalization_audit.py                    # current router (Z must be 1)
python benchmarks/normalization_audit.py --legacy           # re-measure the v1 artifact
python -m unittest tests.test_gate -v                       # the guards
```

Guards, all committed:
- `gate_hybrid.load_split()` refuses to start if any feature column of a
  collected dataset is bit-identical to any label column;
- `tests/test_gate.py` asserts features are identical for two streams that
  share a context but diverge afterwards, that every expert sums to 1 over the
  vocabulary, and hence that any context-only mixture does too;
- `benchmarks/energy_harness.py` warns if a system measures faster than a
  component it contains (the v1 cost table did: GRU alone 5.09 ms/token vs the
  full hybrid 3.60 ms/token).

This is the same discipline that caught the earlier Witten-Bell baseline whose
distribution summed to >1 (see `RESULTS.md` and `tests/test_baselines.py`).

### Cost (1 torch thread, 200 test sentences, 5 interleaved rounds, median)

PTB: kn3 0.011 · mixer5 0.021 · **router-only 0.490** · gru 2.746 ·
fixed-λ 2.852 · **full hybrid 3.538 ms/token**. Routing is not free: +24% over
fixed interpolation, 13.8% of the full system, and the GRU dominates. Energy is
a wall-time × 15 W TDP *proxy*, not measured power. WT-2 numbers in
`benchmarks/results/energy.log`.

### The base repo's "novel" prior is decorative (negative result, unchanged)

The base repo's headline (−9.8 PP from a Zipf-continuation prior) does not
survive controls (Phase 0 in [PROGRESS.md](PROGRESS.md)); it replicates
negatively across four corpora (`benchmarks/cross_corpus/`). The 5-gram
"anomaly" was a tuning artifact too: fair-tuned it lands at 294.2 (was 302.5),
i.e. order-5 buys ≈nothing over order-3 on WT-2 — which the KenLM order sweep
independently confirms (257.3 → 251.9 → 250.9).

---

## Quick tour

```bash
# the trained system, interactively (trains + caches the router on first run):
python hybrid_cli.py suggest "the united states"     # what the model believes
python hybrid_cli.py cloze "<paste real text>"       # hit-rate test (the honest test)
python hybrid_cli.py why america the united states   # corpus counts behind a prediction
python hybrid_cli.py                                 # REPL: suggest/complete/cloze/why/observe

# benchmarks:
python gate_hybrid.py          # the main table + the router-size sweep
python seed_audit.py           # 3-seed ±sd + ensembles for the headline rows
python benchmarks/router_diagnostic.py     # why train-trained routing mis-transfers
python benchmarks/normalization_audit.py   # Z(ctx) over the full vocabulary
python benchmarks/energy_harness.py        # ms/token + joule proxy
python benchmarks/kenlm_baseline.py        # toolkit n-gram baseline (needs lmplz)
python benchmarks/selective_gru.py         # GRU-skip frontier
```

## Reproduce from scratch

One command per corpus regenerates every router-dependent number (data and
model artifacts are gitignored; logs are committed):

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# ---------- WikiText-2 ----------
python benchmarks/run_benchmark.py --stage data && \
python benchmarks/run_benchmark.py --stage beast_train   # gold streams + trigram
python -u nn_expert.py                    # GRU expert (~7 min/epoch, resumable)
python nn_expert.py --score-only          # aligned scores
bash scripts/run_paper_pipeline.sh wt2    # features -> table -> seeds -> audits

# ---------- Penn Treebank (Mikolov split; auto-download below) ----------
for s in train valid test; do curl -sL -o data/ptb/ptb.$s.txt \
  https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.$s.txt; done
python benchmarks/ptb_setup.py gold && python benchmarks/ptb_setup.py beast
HEDGE_RESULTS=benchmarks/results/ptb HEDGE_WARMUP=442 python -u nn_expert.py
HEDGE_RESULTS=benchmarks/results/ptb python nn_expert.py --score-only
bash scripts/run_paper_pipeline.sh ptb
```

`HEDGE_RESULTS` redirects every script's read/write root (WT-2 artifacts live in
`benchmarks/results/`, PTB's in `benchmarks/results/ptb/`). No per-corpus
hyperparameters: the KN discount scale (1.15) is shared, and the router's hidden
size is chosen by 2-fold CV on validation (h=128 on both corpora).

Long runs are suspend-proof: `scripts/auto_resume.sh` (wired via
`postStartCommand`) relaunches an interrupted GRU run from its checkpoint
whenever the Codespace container restarts.

## Honest scope

- Frozen experts dominate the footprint; the *learned* part is the tiny router.
  The claim is about the **tiny-footprint regime** (on-device, offline) and
  about *measuring* what learned routing actually extracts there — not about
  competing with large transformers.
- Our protocol excludes sentence-initial targets and `eos`, so our perplexities
  are systematically higher than batchified literature numbers for the same
  model (our GRU: 146.9 aligned vs 119.2 batchified on WT-2). We never compare
  against published numbers directly.
- Novelty positioning: mixture-of-experts routing is classical (Jacobs, Jordan,
  Nowlan & Hinton 1991; Shazeer et al. 2017). This work tests whether *learned
  per-token* routing changes the conclusion of the *fixed-weight* hybrid
  interpolation literature (Mikolov et al. 2011; Sundermeyer et al. 2012) at
  small footprint. Answer: marginally, and only with validation tuning.
- Negative results are first-class here: Phase 0 (the decorative prior), GRU
  recipes v1/v2, the unbounded-discount knob warning, the train-trained router,
  and the v1 leak itself (`benchmarks/results/*/legacy_leaky_v1/README.md`).

## The base repo (v5, preserved)

A from-scratch modified Kneser-Ney trigram in pure standard-library Python
with a Zipf-continuation prior, personalization user layer, CLI, and a
fixed-protocol benchmark suite, developed at
[chihebnabil/ZipfNextWordPredictor](https://github.com/chihebnabil/ZipfNextWordPredictor). Its published results reproduce exactly
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
for the REPL · `python -m unittest discover tests` for the test suite.
