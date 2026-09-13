# PROGRESS.md — project checklist & status

**Project:** Hybrid expert-mixture language model (statistical experts + tiny
learned gate + small neural expert), built on top of the original
ZipfNextWordPredictor repo. Target: **conference-attempt paper**.

_Last updated: 2026-09-13_

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

### Phase 3 — The neural expert ✅ DONE (stopped by decision)
- [x] GRU recipe v3 (Adam 1e-3, warmup+cosine, clip 1.0): trained through
      ep24 across two idle-timeout interruptions (clean resumes both times);
      valid crept 162.3 → 171.8 through the anneal — revival failed, so
      training STOPPED by decision; final expert = ep11 checkpoint
      (valid 162.16 batchified / test 119.20 batchified / 146.85 aligned)
- [x] Aligned per-position log-probs saved (train/valid/test) with length
      assertions (1,846,068 / 193,222 / 217,004) — all green

### Phase 4 — The money table ✅ DONE (2026-09-12)
- [x] `gate_hybrid.py`: adds the GRU as 6th expert; staleness guard on
      aligned scores
- [x] Fixed-λ KN3+GRU baseline (λ*=0.21 on valid) + static-α over 6 experts
- [x] **The table: GATE(6) 90.28 BEATS fixed-λ 135.84 (−45.6 PP)**;
      hybrid < gate[5] 99.68 < pure GRU 146.85; oracle 57.35; footprint per
      row (counts 21.4 MB + gate 13.5 KB + GRU 32.9 MB). Full table +
      number-hygiene notes: PAPER_NOTES.md §8
- [x] Verdict recorded honestly — conference path live; remaining: 3-seed,
      PTB, energy numbers (Phase 5-6)

### Phase 5 — Second corpus (PTB) ✅ COMPLETE (2026-09-12)
- [x] Download PTB (standard Mikolov split), build gold streams + trigram pickle
- [x] Rebuild 5 experts, collect gate data, train PTB GRU (recipe v3 verbatim,
      30/30 epochs, best valid 123.6 @ep27; aligned scores exact-match rows)
- [x] Phase 4 table on PTB → **VERDICT REPLICATES: GATE(6) 78.81 beats
      fixed-λ 92.65 (−15%); worst seed 88.92 still beats fixed-λ;
      static-α 101.9 dominated; ensembles 80.92 / 65.37.** Honest wrinkle:
      train-trained GATE(6) < gate[5] on PTB (overfits smaller corpus) —
      report, doesn't touch the load-bearing claim. Full numbers: see
      PAPER_NOTES.md §9.

### Phase 6 — Ablations & robustness ✅ COMPLETE (2026-09-12)
- [x] Gate-size sweep on both corpora (h=24/64/128/256; optima: WT-2 h=128
      13.5 KB, PTB h=24 2.6 KB — valid selection picks right one; table in
      PAPER_NOTES §10; PTB money table updated to GATE 68.02 @2.6 KB)
- [x] 3-seed ±sd for headline rows (both corpora; fixed-λ dominated at
      every seed everywhere)
- [x] Cross-check: gate beats fixed-λ on BOTH corpora — load-bearing
      claim confirmed (§8, §9)
- [x] 5-gram baseline fixed: fair-tuned 294.16 (was 302.52 tuning artifact)
- [x] Energy/latency harness (benchmarks/energy_harness.py): full hybrid
      3.6 ms/tok WT-2 / 1.0 ms/tok PTB on 4 CPU cores; gate overhead ≈8%
      or less; sizes 69 MB / 29 MB. Tables in PAPER_NOTES §10

### Phase 7 — Write-up ⬜ PENDING
- [x] Draft written: `paper/main.tex` + `references.bib` (ACL style, 4 pp,
      incl. compiled PDF). Intro per §5 contract (MoE conceded first
      para), 2-corpus money table, ablations, sweep, energy, honest
      limitations incl. the PTB wrinkle, two-step plan in §7 of PAPER_NOTES.
      Remaining: venue pick + CFP conformance (author block filled).
- [ ] Related work — VERIFIED (see PAPER_NOTES §11): Mikolov et al. 2011
      (ASRU) + Sundermeyer et al. 2012 (fixed-weight hybrid interpolation),
      Chang et al. 2015 (discriminative interpolation weights), Mathur et
      al. 2023 (PersonaLM, on-device personalization); lineage: Jacobs et
      al. 1991 (4 authors!), Jordan & Jacobs 1994, Shazeer et al. 2017;
      oracle bound: Freund & Schapire 1997 (Hedge), Cesa-Bianchi & Lugosi
      2006; anchors: Chen & Goodman 1999, Jelinek 1980, Merity 2017,
      Zaremba 2014. DROPPED as unverifiable: Sak 2013, Levit 2023,
      Qin 2023, Zhong 2025.
- [x] Selective GRU activation (pre-submission add): realizable skip rule
      (threshold the gate's own alpha_gru — causal, free) runs the GRU on
      46.7%/54.7% of tokens for +0.41/+0.20 PP; oracle analysis shows value
      concentration (full oracle at 50% usage). In paper future work +
      extension pillar #3 (benchmarks/selective_gru.py, PAPER_NOTES §10)
- [x] KenLM toolkit baseline (pre-submission add): 5-gram 250.9 WT-2 /
      138.4 PTB under the aligned protocol — beats our trigram expert,
      ~2-3x worse than the hybrid; in the paper money table (PAPER_NOTES §10)
- [x] Repo release hygiene: MIT LICENSE + base-repo attribution; binaries
      gitignored w/ regen commands in README; verified bibliography (§11);
      23 base tests green; auto-resume watchdog wired.

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
| GRU expert (8.2M, epoch-11) | 146.85 aligned / 119.20 batchified | final (§3) |
| Hybrid (gate + GRU, 6 experts) | **90.28** | §8 |

## Decisions log
- 2026-09-10: **Recipe v2 (clip 1.0, Adam 3e-3, bs64/tsl32, 30ep) killed at
  epoch 1.** Train PP 1417 (clip fix works — context learning resumed) but
  valid PP exploded to ~1e9: no warmup + hot LR blew up the tied-embedding
  decoder (max|logit| ≈ 10 after 980 steps; 92% of valid tokens at
  logp < -15). bs64/tsl32 was also ~40% SLOWER per epoch than bs32/tsl64
  (thread-sync overhead on small GRU cell ops). Snapshot discarded.
- 2026-09-10: **Recipe v3:** Adam 1e-3, 1-epoch linear warmup then cosine to
  1% LR, clip 1.0, bs32/tsl64 (known pace), 30 epochs, per-epoch canary
  print (lr, max|logit|). Fresh start (seed 42, deterministic).
- 2026-09-10: **GRU recipe v1 declared a negative result.** The 6-epoch run
  (clip_grad_norm 0.25, cosine→0 over 5,880 steps) produced a context-blind
  model: identical output distribution for any context, valid PP 917 / test
  734 (≈ unigram floor 825). Root cause isolated by controlled 250-step probe:
  global bias gradients eat the 0.25 norm budget and clip the recurrent
  (context) pathway to zero — clip=1.0 shows context-KL 0.348 vs 0.001 at
  equal steps. Checkpoint archived as `nn_gru_contextblind_v1.pt`.
- 2026-09-10: **Recipe v2 launched:** EPOCHS 30, CLIP 1.0, BS 64/TSL 32,
  threads=4, cosine eta_min=1% LR, per-epoch `nn_gru_last.pt` snapshot
  (epoch+opt+sched+best_pp) for clean resume. Launch detached so the run
  survives closed terminals:
  `python -u nn_expert.py 2>&1 | tee benchmarks/results/nn_train.log`
- 2026-09-09: Added `progress.py` (stdlib-only watcher: process health,
  checkpoint age, ETA from epoch-1 calibration; `watch -n 60 python
  progress.py`). Future long runs (incl. PTB GRU) launch with
  `python -u nn_expert.py 2>&1 | tee benchmarks/results/nn_train.log`
  so progress survives closed terminals.
- 2026-09-09: Fixed `score_aligned` in `nn_expert.py` (pre-training bug):
  the old keep-mask dropped only eos targets, yielding W-1 rows instead of
  the gate's W-S rows. Now drops sentence-initial targets too; mask verified
  on real gold streams to produce exactly 193,222 / 217,004 rows. Added
  `--score-only` flag (re-score a checkpoint without retraining — the
  in-flight training run still carries the old code, so its end-of-run
  .npy outputs must be regenerated via `python nn_expert.py --score-only`)
  and length assertions before each np.save.
- 2026-09-09: Project renamed **Hedge** (after the online expert-prediction
  algorithm family; "one tamer, many beasts"). Repo: chihebnabil/hedge
- 2026-09-09: GRU training continues in GitHub Codespaces; repo ships
  rebuild-everything scripts (artifacts gitignored)
- 2026-09-09: GRU trained for the **full 6 epochs** (quality over speed)
- 2026-09-09: Target = **conference attempt** (2 corpora + energy numbers);
  fallback = workshop routing-analysis paper
- 2026-09-09: New mandatory baseline: fixed-λ interpolation (literature:
  Mikolov 2011 / Sundermeyer 2012 fixed-weight hybrid line). Gate must beat
  it or the claim shrinks honestly.
- 2026-09-09: All training runs: detached process + per-epoch checkpoints +
  resumable (no more silent 20-minute grinds)

## Watchlist / risks
- Gate vs fixed-λ is the load-bearing comparison — if lost, reposition paper
  as routing analysis (workshop)
- PTB pipeline quirks (format, vocab, unk conventions) — buffer time reserved
- CPU-only training: GRU ≈ 30–40 min/epoch; keep checkpoints per epoch
- Seed variance on small-data gate training — always report ≥3 seeds

## Where things live
- **Paper dossier (stable findings, negative results, protocol, decision
  tree, paper TODO): [PAPER_NOTES.md](PAPER_NOTES.md) — keep it current**
- Scripts: `mixer_lm.py`, `mixer_cli.py`, `gate_lm.py`, `gate_lm2.py`,
  `gate_lm3.py`, `gate_lm4.py`, `gate_hybrid.py`, `nn_expert.py`,
  `progress.py`
- Data caches: `benchmarks/results/gate_{train,valid,test}.npz` (per-position
  expert votes + evidence), `gold_streams.pkl`, `beast_trigram.pkl`
- Training artifacts: `benchmarks/results/nn_gru_best.pt` (+ `nn_logp_*.npy`
  once Phase 3 completes)
- Original repo docs: `RESULTS.md`, `benchmarks/`, `tests/` (23 tests green)
