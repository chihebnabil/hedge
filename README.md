# Hedge 🦔
### One tamer, many beasts — a hybrid expert-mixture language model
*(built on the ZipfNextWordPredictor base — "BEAST" v5)*

This project started as a from-scratch pure-stdlib Kneser-Ney trigram with a
"novel" Zipf-continuation prior, and grew into something bigger: **a study of
how far tiny learned routing can push frozen cheap experts — ending in a
hybrid statistical-neural LM.**

**The one-sentence story:** a 6.5 KB learned gate that watches five frozen
1990s-style experts and their context evidence cuts perplexity from 199.5 to
**99.7** — within 12% of the hindsight oracle (89.1) — using less learned
knowledge than one thumbnail image.

---

## Findings (all measured on WikiText-2, 217,004 test positions)

### 1. The "novel" prior is decorative (negative result, rigorously verified)
The base repo's headline (−9.8 PP from a Zipf-continuation prior) does not
survive controls: diverting the *same* backoff mass to the base distribution
instead of the Zipf curve wins at every setting and every corpus size.
Tuning the repo's own neglected baseline parameter (`discount_scale`) closes
~60% of the published headline gap on its own. Full reproduction details in
[RESULTS.md](RESULTS.md) and `PROGRESS.md` (Phase 0).

### 2. A team of cheap experts beats one fancy n-gram
Five frozen experts (modified Kneser-Ney trigram, lag-2/lag-3 bigrams, a
512-token recency cache, a continuation unigram) blended with online
context-bucket weighting: **292.4 → 199.5 PP** (`mixer_lm.py`).

### 3. The routing signal is the votes×evidence interaction
A tiny MLP gate that reads *both* the experts' actual predictions ("votes")
and 15 cheap context features ("evidence") reaches **99.7 PP**. Ablations:
votes alone 201.8 · evidence alone 367.9 (worse than equal mixing!) · both →
**99.7**. The evidence tells the gate *whose* vote to trust. All learned
knowledge: 6.5 KB.

### 4. A small neural expert joins the team (in progress)
A GRU (8.2M params, trained on CPU) becomes expert #7; the gate routes
between 1990s statistics and modern sequence memory. The open question —
the paper's spine — is the footprint-matched fight: **does old+new+gate beat
pure neural at equal memory, on two benchmarks?** Status and numbers:
[PROGRESS.md](PROGRESS.md).

### 5. Personalization is a routing side effect
Observed user text feeds a 6th expert; the gate re-weights toward it after a
short replay (`mixer_cli.py demo` shows the blend weights shifting live:
α_user 0.02 → 0.665 after ~55 user words).

---

## Quick tour

```bash
python mixer_lm.py        # benchmark: experts, mixer, gate-style protocol, top-k
python mixer_cli.py demo  # end-to-end tour with real prompts + live personalization
python gate_lm.py         # collect per-position expert data; train the MLP gate
python gate_lm2.py        # ablations: votes vs evidence vs both; seed audit
python gate_lm3.py        # cache-age distribution shift + the fix
python nn_expert.py       # train the GRU expert (CPU, checkpointed, resumable)
python -m unittest tests.test_model -v   # base repo tests (23, green)
```

The base repo's engines, CLI, personalization user layer and benchmarks are
unchanged and still work (see the original documentation below).

---

## Honest scope

- Statistical experts + tiny gate: ~21 MB of counts, 6.5 KB of learned routing,
  pure stdlib + numpy. The GRU expert is the only heavyweight part.
- Not competitive with large transformers on quality; the claim is about the
  **tiny-footprint regime** (on-device, offline, zero-dependency settings)
  and about *measuring* what learned routing extracts from frozen experts.
- CPU-only training is a feature of the story (green computing), not a secret.
- Every number in this README is reproducible from the scripts above with the
  fixed gold streams (`benchmarks/results/`).

## The base repo (v5, preserved)

A from-scratch modified Kneser-Ney trigram in pure standard-library Python
with a Zipf-continuation prior, personalization user layer, CLI, and a
fixed-protocol benchmark suite. Its published results reproduce exactly
(see `RESULTS.md`); its novel prior's contribution is analyzed honestly in
Phase 0 of `PROGRESS.md`. Engines: `"beast"` (default), `"modified_kneser_ney"`,
`"kneser_ney"`, `"zipf"` (legacy v4). API highlights:

```python
from ZipfNextWordPredictor import ZipfNextWordPredictor

model = ZipfNextWordPredictor(corpus_texts=[...], n_gram_size=3)
model.predict_next_words("once upon a")
model.observe("user text")          # personalization
model.tune(heldout)                 # self-calibration
```

`python example.py` for the guided demo · `python cli.py chat -m model.pkl`
for the REPL · `python -m unittest tests.test_model` for the test suite.

---

## Running in GitHub Codespaces

Everything retrains from scratch in a fresh Codespace (data artifacts are
gitignored — the scripts rebuild them):

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 1. data + base model (WikiText-2 download + gold streams + trigram pickle)
python benchmarks/run_benchmark.py --stage data
python benchmarks/run_benchmark.py --stage beast_train

# 2. GRU expert (CPU, ~30-40 min/epoch, checkpointed + resumable)
python -u nn_expert.py

# 3. gate experiments
python gate_lm.py        # collects per-position expert data (~2 min) + trains gate
python gate_lm2.py       # ablations + seed audit
python gate_lm3.py       # cache-age fix experiment
```

If the WikiText-2 download in `--stage data` gets blocked (Cloudflare),
fetch it manually first:

```bash
curl -L -A "Mozilla/5.0" -o data/wikitext-2-v1.zip \
  https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
```

Order of scripts, live status of every experiment, and all validated numbers:
[PROGRESS.md](PROGRESS.md).

