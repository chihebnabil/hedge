# PROGRESS.md — project checklist & status

**Project:** Hybrid expert-mixture language model (statistical experts + tiny
learned gate + small neural expert), built on top of the original
ZipfNextWordPredictor repo. Target: **conference-attempt paper**.

_Last updated: 2026-09-09 (session in progress)_

---

## Phase checklist

### Phase 0 — Verdict on the original repo ✅ DONE
- [x] Reproduce every published repo row exactly (MKN 306.15, BEAST+prior 296.35)
- [x] Tune the neglected baseline (discount_scale: 306.1 → 300.3) — ~60% of the headline gap was baseline neglect
- [x] Control experiment: divert the same backoff mass to the base distribution
      instead of the Zipf prior → wins at every setting and every corpus size
      (288.5 vs 289.6 tuned; 293.7 vs 296.3 fixed) → **the "novel" prior is decorative**
- [x] Scaling study: Heaps V=17.2·N^0.533; local PP(N) γ≈0.20; prior gain GROWS
      with N (contradicts the repo's sparse-corpus insurance claim)

### Phase 1 — The mixer (statistical system) ✅ DONE
- [x] `mixer_lm.py`: 5 frozen experts (MKN trigram, lag-2/lag-3 bigrams, cache
      window, unigram) + online context-bucket mixing
- [x] Benchmark: **MIXER 199.50** vs MKN 300.3 / repo BEAST 292.4
- [x] Top-k suggestions via candidate pools: 19.5%/32.6% vs kn3 19.3%/32.3%
- [x] `mixer_cli.py`: suggest/complete/observe/score/chat/demo — E2E tested
- [x] Personalization demo (user expert + replay): α_user 0.02→0.665 visible shift

### Phase 2 — The neural gate ✅ DONE (headline result)
- [x] `gate_lm.py`: per-position dataset (1.85M train / 193k valid / 217k test
      positions; experts' votes + 15 evidence features), cached as npz
- [x] MLP gate (6.5 KB, hidden 64) beats online bucket mixer: **test PP 104.8 vs 199.5**
- [x] Ablations (paper Table 1): votes-only 201.8 | evidence-only 367.9 | bucket-only
      255.5 | both → 103.2 — **the routing signal lives in the votes×evidence interaction**
- [x] Cache-age distribution shift found & fixed (age-invariant cache density
      feature): train-trained gate 103 → **99.7 (stable across h=64/128/256)**
- [x] Seed audit: valid-trained gate is high-variance (90–111 across seeds);
      90.0 was a lucky seed. **Stable honest number: 99.7 (train-trained).**
- [x] Verified: gate at ~99.7 vs hindsight oracle 89.1 (≈12% gap)

### Phase 3 — The neural expert 🔄 IN PROGRESS
- [x] torch 2.14.0+cpu installed
- [x] `nn_expert.py`: GRU 2×256, tied embeddings, 8.2M params, V=28,715
- [x] Bug fixed: batch orientation (batch_first=True) + chunk-edge crashes
      (first launch learned from scrambled text ~35 min, crashed, no checkpoint
      saved — clean restart)
- [ ] GRU training 6 epochs on WT-2 — **moved to GitHub Codespaces**
      (local run killed before epoch 1 finished; no checkpoint existed;
      devcontainer auto-installs numpy/torch-CPU; script is seed-deterministic,
      so Codespace training reproduces the identical model)
- [ ] Aligned per-position log-probs saved (nn_logp_valid.npy / nn_logp_test.npy)
      with length assertions vs gate npz rows (193,222 / 217,004)

### Phase 4 — The money table ⬜ PENDING (unblocks on Phase 3)
- [ ] `gate_hybrid.py`: GRU = expert #7 → retrain gate (minutes)
- [ ] **Fixed-λ interpolation baseline** (Sak 2013 / Levit 2023 style):
      best λ over grid for {KN3+GRU} AND best static-α over all 6 experts
- [ ] The table: pure GRU | 5 old experts | fixed-λ classic | our gate hybrid | oracle
- [ ] Footprint accounting per row (counts MB + gate KB + GRU MB, fp32/fp16)
- [ ] Verdict recorded honestly (either outcome is a paper path)

### Phase 5 — Second corpus (PTB) ⬜ PENDING
- [ ] Download PTB (standard Mikolov split), build gold streams + trigram pickle
- [ ] Rebuild 5 experts, collect gate data, train PTB GRU (vocab ~10k → faster)
- [ ] Repeat Phase 4 table on PTB

### Phase 6 — Ablations & robustness ⬜ PENDING
- [ ] Ablations on both corpora (votes/evidence/both; gate-size sweep)
- [ ] 3-seed ±sd for headline rows
- [ ] Cross-check: does the gate beat fixed-λ? (the paper's load-bearing claim)
- [ ] Energy/latency harness: ms/token + CPU-time×TDP joule proxy, stated method

### Phase 7 — Write-up ⬜ PENDING
- [ ] Intro / method / tables / honest limitations
- [ ] Related work: Sak 2013, Levit 2023, Mathur 2023 (PersonaLM), Qin 2023,
      Zhong 2025 — hybrid interpolation & on-device personalization exist;
      our lane = learned evidence-aware multi-expert routing + oracle-bound
      measurement + footprint-matched fight
- [ ] Repo release hygiene (scripts reproducible top-to-bottom)

---

## Results ledger (validated numbers, WT-2 test, 217,004 positions)

| System | Test PP ↓ | Status |
|---|---|---|
| MKN trigram (ds=1.0) | 300.15 (protocol 306.1 = untuned) | validated |
| Repo BEAST (fixed prior) | 296.35 | validated (= published) |
| Repo BEAST (tuned) | 292.4 | validated (= published) |
| Equal mix (5 experts) | 263.85 | validated |
| Static bucket gate | 256.5 | validated |
| Online bucket mixer (valid-warmed, frozen) | **199.50** | validated |
| MLP gate, train-trained (6.5 KB) | **99.7** | validated, stable |
| MLP gate, valid-trained | 90–111 (seed-dependent) | high variance |
| Hindsight oracle (5 experts) | **89.11** | bound |
| GRU (8.2M, 6 epochs) | — | training now |
| Hybrid (gate + GRU) | — | Phase 4 |

## Decisions log
- 2026-09-09: Project renamed **Hedge** (after the online expert-prediction
  algorithm family; "one tamer, many beasts"). Repo: chihebnabil/hedge
- 2026-09-09: GRU training continues in GitHub Codespaces; repo ships
  rebuild-everything scripts (artifacts gitignored)
- 2026-09-09: GRU trained for the **full 6 epochs** (quality over speed)
- 2026-09-09: Target = **conference attempt** (2 corpora + energy numbers);
  fallback = workshop routing-analysis paper
- 2026-09-09: New mandatory baseline: fixed-λ interpolation (literature: Sak/
  Levit line). Gate must beat it or the claim shrinks honestly.
- 2026-09-09: All training runs: detached process + per-epoch checkpoints +
  resumable (no more silent 20-minute grinds)

## Watchlist / risks
- Gate vs fixed-λ is the load-bearing comparison — if lost, reposition paper
  as routing analysis (workshop)
- PTB pipeline quirks (format, vocab, unk conventions) — buffer time reserved
- CPU-only training: GRU ≈ 30–40 min/epoch; keep checkpoints per epoch
- Seed variance on small-data gate training — always report ≥3 seeds

## Where things live
- Scripts: `mixer_lm.py`, `mixer_cli.py`, `gate_lm.py`, `gate_lm2.py`,
  `gate_lm3.py`, `gate_lm4.py`, `nn_expert.py`
- Data caches: `benchmarks/results/gate_{train,valid,test}.npz` (per-position
  expert votes + evidence), `gold_streams.pkl`, `beast_trigram.pkl`
- Training artifacts: `benchmarks/results/nn_gru_best.pt` (+ `nn_logp_*.npy`
  once Phase 3 completes)
- Original repo docs: `RESULTS.md`, `benchmarks/`, `tests/` (23 tests green)
