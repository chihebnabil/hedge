# PAPER_NOTES.md — the paper dossier (living document)

**Purpose:** every fact, number, protocol rule and decision the paper needs,
in one place, updated as results land. The live checklist remains
[PROGRESS.md](PROGRESS.md); this file holds the *stable findings*.
_Last updated: 2026-09-15 — **§0 is the only authoritative ledger.** §1, §8,
§9 and §10 are kept for history and are SUPERSEDED: their router numbers came
from a leaky protocol (§12)._

---

## 0. READ THIS FIRST — the corrected ledger (2026-09-15)

**The v1 headline was an artifact.** The router's features `[0:5]` were
`log p_expert(w_gold)` — bit-identical to the label columns — and feature `[12]`
was the gold word's cache count. The router's weights therefore depended on the
word being scored, so `P(w|ctx)` was not a distribution: its mass over the
vocabulary measured **2.66 (WT-2)** and **2.56 (PTB)** instead of 1.000, and
every router perplexity was deflated by that factor. Normalized, the same
trained router scores ≈240 (WT-2) / ≈174 (PTB) instead of the 90.3 / 68.0 it
appeared to. Full forensics: §12. Reproduce:
`python benchmarks/normalization_audit.py --legacy`.

**Corrected main table** (aligned protocol; 217,004 WT-2 / 75,623 PTB
positions; every tuned component gets validation data only;
`benchmarks/results/{,ptb/}money_table.log`):

| system | WT-2 PP | PTB PP | WT-2 MB | PTB MB |
|---|---|---|---|---|
| pure KN3 trigram | 296.23 | 165.15 | 21.4 | 9.2 |
| KenLM 5-gram (defaults) | 250.89 | 138.42 | — | — |
| pure GRU (aligned) | 146.85 | 103.01 | 32.7 | 13.1 |
| uniform mixture (6) | 182.56 | 138.44 | 68.7 | 28.7 |
| fixed-λ KN3+GRU (λ* on valid) | 135.84 (λ*=0.21 **on KN3**) | 92.65 (λ*=0.28) | 54.0 | 22.3 |
| static-α (6), train-fit | 170.97 | 101.74 | 68.7 | 28.7 |
| **static-α (6), valid-fit** | **119.14** | **89.71** | 68.7 | 28.7 |
| router, train-trained (control) | 250.79 (h=256) | 138.19 (h=64) | 68.7 | 28.7 |
| **router, valid-tuned (h=128 by CV)** | **117.86** | **87.77** | 68.7 | 28.7 |
| router 3-seed mean ± sd | 118.06 ± 0.29 | 87.84 ± 0.06 | | |
| router 3-seed ensemble | 117.60 | 87.74 | | |
| router over 5 stat. experts, valid-tuned | 194.47 | 132.82 | 36.0 | 15.6 |
| router over 5 stat. experts, train-trained | 481.00 | 226.71 | 36.0 | 15.6 |
| best single expert (normalized bound) | 146.85 (gru) | 103.01 (gru) | — | — |
| per-position best expert (**not** normalized) | (57.35) | (44.37) | — | — |

KenLM footprint cells are deliberately empty: this environment has `lmplz` but
not `build_binary`, so the 53.8/23.1 MB "trie binary" figures recorded in §10
are not reproducible from the committed code and must not be quoted.
`kenlm.log` now records the ARPA size instead.

**Verdict (corrected).** The router beats tuned two-expert interpolation by
13.2% (WT-2) / 5.3% (PTB) and a tuned static 6-expert mixture by 1.1% / 2.2%
(≈3.7 and ≈32 router-sd). Trained on the training stream instead, it is worse
than *uniform* mixing (250.8 vs 182.6; 138.2 vs 138.4). The load-bearing
positive result is the **hybrid**, not the routing: a valid-tuned static
mixture of five cheap experts + a small GRU beats the GRU alone, tuned
interpolation and a KenLM 5-gram at 68.7/28.7 MB.

**Protocol rules added (must not drift):**
1. Router features must be candidate-independent (`gate_lm.FeatMaker`); the
   mixture must sum to 1 over the vocabulary. Guards:
   `gate_hybrid.assert_no_label_leak`, `tests/test_gate.py`,
   `benchmarks/normalization_audit.py`.
2. Budget matching: λ*, static-α and the router (weights **and** hidden size,
   by 2-fold CV inside valid) are all tuned on validation only. Train-fit
   variants are negative controls, never headline rows.
3. Cost rows quote the **minimum** of ≥5 interleaved rounds, single torch
   thread, and the harness asserts that no system measures faster than a
   component it contains.
4. Corpus stats quoted from the artifacts, not the literature: WT-2 gold
   streams are 1,927,034 / 201,797 / 226,731 tokens, vocab 28,714 (headers
   dropped, lowercased, OOV→unk — NOT the commonly quoted 2.0M/214k/245k/33k).
   PTB: 907,138 / 71,825 / 80,594, vocab 9,644.

**Cost (min of 7 interleaved rounds, 1 torch thread, 200 test sentences).**
PTB: kn3 0.006 · mixer5 0.018 · router-only 0.479 · gru 2.231 · fixed-λ 2.435 ·
static6 — · full hybrid 3.427 ms/tok. WT-2: see `benchmarks/results/energy.log`.
Routing's marginal cost over the *same* six experts mixed statically is the
router-only row: ~0.44–0.48 ms/token (8–14% of the hybrid). The hybrid vs
2-expert fixed interpolation gap (+15% WT-2, +41% PTB) is mostly the four extra
expert evaluations, not the router.

**Selective GRU activation is dead** (v1 claimed 47–55% usage for ≤0.5 PP): the
corrected router keeps the GRU on for 99.0% (WT-2) / 89.7% (PTB) of tokens even
at τ=0.5 — the honest router trusts the GRU nearly everywhere.

**Bibliography note:** `paper/submission/` is now GENERATED from
`paper/main.tex` by `scripts/make_submission.py` (ACL `review` option =
anonymous + line numbers). The hand-written copy it replaces misattributed
`mathur2023` ("On-device language modeling", A. Mathur — actually PersonaLM,
P. Mathur, Findings of EMNLP 2023), `chang2015` ("D. Chang" — actually S. Chang
et al., ASRU 2015), `mikolov2011` (the RNNLM toolkit instead of the ASRU
paper), `jelinek1980` (added Mercer) and `chen1999` (wrong journal). Never
hand-edit the submission copy again.

---

## 1. Verified numbers ledger (WikiText-2) — ⚠️ SUPERSEDED by §0/§12 (router rows invalid)

All rows: word-level perplexity, natural log, one shared gold token stream,
217,004 aligned test positions, single core, pure-stdlib statistical stack.

| System | Test PP ↓ | Learned knowledge | Status |
|---|---|---|---|
| MKN trigram (ds=1.0) | 300.15 (protocol 306.1 untuned) | counts only | validated |
| Repo BEAST (fixed prior) | 296.35 | counts only | validated (= published) |
| Repo BEAST (tuned) | 292.4 | counts only | validated |
| Equal mix (5 experts) | 263.85 | none | validated |
| Static bucket gate | 256.5 | tiny | validated |
| Online bucket mixer | **199.50** | tiny | validated |
| MLP gate, 5 experts | **99.7** | **6.5 KB** | validated, stable h∈{64,128,256} |
| MLP gate, valid-trained | 90–111 (seed-dependent) | 6.5 KB | 90.0 was a lucky seed (disclosed) |
| Hindsight oracle, 5 experts | **89.11** | — | bound; gate within 12% |
| GRU recipe v3 (in flight) | ep3 valid 202.04 | 8.2M params | epoch 3/30 |

**Ablations (paper Table 1 material):** votes-only 201.8 · evidence-only
367.9 (worse than equal mixing!) · bucket-only 255.5 · both → 99.7.
**Finding: the routing signal lives in the votes × evidence interaction** —
evidence tells the gate *whose* vote to trust.

**Cache-age distribution shift (found & fixed):** train-collected features
leaked cache age (window size, raw count) because train streams start with a
full cache but valid/test start empty. Fix: age-invariant density c/(t+1).
Train-trained gate 103 → 99.7, stable across widths.

**Baseline corrections (honesty material):** Witten-Bell re-measured with the
correct textbook formula (246.4 → 483.6); corrected ordering matches Chen &
Goodman (1998). All baselines guarded by sum-to-one tests
(tests/test_baselines.py). Known open weakness: our 5-gram (302.5) loses to
our trigram (292.3) — must be fixed or explicitly justified before submission.

## 2. Negative results (appendix-grade)

### 2.1 Phase 0 — the base repo's "novel" Zipf prior is decorative
- Reproduced every published row exactly (MKN 306.15, BEAST+prior 296.35).
- Tuning the neglected `discount_scale` closes ~60% of the headline gap alone
  (306.1 → 300.3).
- Control: diverting the *same* backoff mass to the base distribution beats
  the Zipf prior at every setting and corpus size (288.5 vs 289.6 tuned;
  293.7 vs 296.3 fixed).
- Scaling study: Heaps V = 17.2·N^0.533; local PP(N) slope γ ≈ 0.20; the
  prior's gain GROWS with corpus size — contradicting the repo's
  sparse-corpus-insurance justification.

### 2.2 GRU recipe v1 — gradient clipping can silently produce a unigram
- Recipe: Adam 3e-3, clip_grad_norm **0.25**, cosine→0, 6 epochs (5,880 steps).
- Outcome: valid 917 / test 734 (unigram floor ≈ 825) and **context-blind**:
  identical output distribution for any context; GRU hidden out std ≈ 0.11.
- Mechanism (controlled 250-step probe, identical seed/data): with clip 0.25
  the context-KL between two different contexts is **0.001** vs **0.348** at
  clip 1.0 — global bias (unigram) gradients eat the norm budget and clip the
  recurrent (context) pathway toward zero.
- Checkpoint archived: `benchmarks/results/nn_gru_contextblind_v1.pt`.

### 2.3 GRU recipe v2 — hot LR without warmup explodes tied embeddings
- Recipe: clip 1.0 (fix), Adam 3e-3, **no warmup**, bs64/tsl32, 30 epochs.
- Killed at epoch 1: valid PP ≈ 1.17e9; 92% of valid tokens at logp < −15;
  max|logit| ≈ 10 after 980 steps. Tied embedding/decoder weights blow up;
  the model becomes extremely confident on a few frequent tokens.
- Also measured: bs64/tsl32 is ~40% SLOWER per epoch than bs32/tsl64 on CPU
  (thread-sync overhead on small GRU cell ops) despite 4 threads.

## 3. Recipe v3 (final expert) — spec & trajectory

Adam 1e-3 · 1-epoch linear warmup (980 steps) then cosine to 1% LR ·
clip 1.0 · bs32/tsl64 (2,048 tok/step) · 30 epochs · seed 42, deterministic ·
per-epoch snapshot `nn_gru_last.pt` (epoch+opt+sched+best_pp) for clean
resume · per-epoch canary print (lr, max|logit|) · ~30 min/epoch on the
4-core Codespace · arch: GRU 2×256, tied embeddings, dropout .25/.35,
V = 28,715 (trigram w2i + eos), 8.2M params.

| epoch | train PP | valid PP | lr | max\|logit\| |
|---|---|---|---|---|
| 1 | 1009.9 | 406.25 | 1.00e-3 | 12.4 |
| 2 | 318.4 | 246.75 | 9.97e-4 | 17.0 |
| 3 | 210.3 | 202.04 | 9.88e-4 | 17.4 |
| 4 | 163.1 | 184.76 | 9.74e-4 | 18.2 |
| 5 | 137.2 | 174.60 | 9.54e-4 | 17.7 (first decline) |
| 6 | 120.4 | 169.44 | 9.29e-4 | 17.1 (second decline — explosion risk retired) |
| 7 | 108.2 | 168.34 | 8.99e-4 | 18.2 (plateau phase begins; train drops, valid stalls — normal mid-run) |
| 8 | 98.8 | 164.08 | 8.64e-4 | 18.2 |
| 9 | 91.4 | 162.58 | 8.25e-4 | 19.3 |
| 10 | 85.5 | 162.62 | 7.83e-4 | 18.1 (first flat epoch — plateau noise; best remains ep9) |
| 11 | 80.5 | 162.34 | 7.37e-4 | 18.2 (marginal new best) |
| 12 | 76.3 | 163.11 | 6.88e-4 | 20.1 (valid wiggle up — still plateau noise; best remains ep11) |
| 13 | 72.8 | 163.08 | 6.37e-4 | 19.4 |
| 14 | 68.5 | 164.78 | 5.85e-4 | 19.8 (post-resume; valid creeping — hot-LR overfit, anneal not yet started) |
| 15 | 65.0 | 165.11 | 5.32e-4 | 19.9 |
| 16 | 62.2 | 165.30 | 4.78e-4 | 20.7 (creep decelerating: +0.19) |
| 17 | 60.0 | 165.40 | 4.25e-4 | 20.8 (+0.10 — flattening; LR crosses the 3.5e-4 endgame gate next epoch) |
| 18 | 58.0 | 167.03 | 3.73e-4 | 20.1 (creep blip +1.6) |
| 19 | 56.4 | 165.77 | 3.22e-4 | 20.4 (pulled back −1.3) |
| 20 | 54.9 | 166.67 | 2.73e-4 | 22.2 (canary spike, settled after) |
| 21 | 53.7 | 167.76 | 2.27e-4 | 19.8 |
| 22 | 51.6 | 169.91 | 1.85e-4 | 20.1 |
| 23 | 50.4 | 170.91 | 1.46e-4 | 21.2 |
| 24 | 49.5 | 171.81 | 1.11e-4 | 21.4 |

**STOP DECISION (2026-09-12, after ep24): training halted by decision;
epoch-11 checkpoint (162.34) is the final expert.** Rationale: the
deep-anneal revival failed — valid RISING through the endgame (162.3 → 171.8
across ep11-24, +1/epoch even at LR 1.1e-4) while train keeps falling
(53.7 → 49.5); reclaiming 9.5 PP in the last 6 epochs against that trend is
implausible, and the best-by-valid artifact is immutable. The mid-run-best-
beats-the-anneal outcome is itself a reportable recipe finding (hot-phase
best not recovered by cosine tail, CPU-scale GRU, WT-2).
Recipe text for the paper: "Adam 1e-3, 1-epoch warmup, cosine to 1e-5,
clip 1.0, bs32/tsl64; up to 30 epochs with best-by-validation checkpointing;
validation-optimal checkpoint occurred at epoch 11."

**Final GRU expert numbers (epoch-11 weights):**
- valid PP 162.16 (batchified eval; 161.86 on re-eval — small eval noise)
- **test PP 119.20** (batchified) — WT-2 test is substantially easier than
  valid for this model; all table rows use the same test stream, so the
  comparison is internally consistent
- aligned scores regenerated 2026-09-12: nn_logp_{train,valid,test}.npy,
  length assertions passed (1,846,068 / 193,222 / 217,004). NOTE: train
  scoring was added to --score-only (the 6-expert gate needs GRU votes on
  train positions).

**Epoch-20 review (policy B formally fired): conditions met (no gain in
4 epochs, LR < 3.5e-4). DECISION: continue to 30 anyway.** Rationale: (a)
best-by-valid is banked — continuation is free-option; (b) remaining cost is
bounded (~3.9 h); (c) the deep-anneal epochs (25-30, LR < 1.5e-4) are the
classic revival window and the scientific record benefits from completing
the 30-epoch curve the recipe specifies; (d) if revival fails, the paper
reports it honestly — "mid-run best survived the anneal" is itself a
findings-grade observation about this recipe at this scale.

**Stop decision (2026-09-11, after ep15):** CONTINUE to 30. Policy B
not fired (LR 5.3e-4 > 3.5e-4 threshold — anneal endgame has not begun).
Valid creep during the hot phase is the textbook cosine pattern; the deep
anneal (last ~5 epochs, LR < 1.5e-4) typically reclaims 10-25 PP. The
best-by-valid checkpoint (162.34 @ep11) is immutable once recorded, so
continuation is free-option value: it can only improve the final artifact,
never worsen it. Re-assess at epoch ~20-22 if valid still > ~160.

**Incident log (root cause both times: the Codespace idle timeout stops the
container ~30 min after the editor disconnects):**
- 2026-09-11 (daytime): stopped mid-epoch-14; resumed from ep13.
- 2026-09-11 → 09-12 (overnight): stopped mid-epoch-22; relaunched next
  morning — `resuming: epoch 21/30, best valid PP so far 162.34`.
- Loss each time: one partial epoch. Mitigations: raise the idle timeout in
  GitHub settings (Settings → Codespaces), and `scripts/auto_resume.sh`
  (wired via the devcontainer `postStartCommand`) now relaunches an
  interrupted run automatically whenever the container restarts.

## 4. Protocol & alignment rules (must not drift) — see §0 for the added invariants

- **Aligned positions:** every word except each sentence-initial word and
  every eos. WT-2 rows: train 1,846,068 · valid 193,222 · test 217,004
  (gate npz rebuilt in Codespace 2026-09-11 with exactly these counts).
- `score_aligned` (nn_expert.py) produces GRU log-probs on exactly those rows
  (keep-mask: target ≠ eos AND predecessor ≠ eos); length assertions run
  before each save. Regenerate scores with `python nn_expert.py --score-only`.
- Experts are FROZEN; only the gate learns. Gate features: 5 votes + 15
  evidence (20 dims, all causal), age-invariant density feature per §1.
- Model selection on valid only; fixed-λ grid 0..1 (201 pts), λ* picked on
  valid, test reported once. `gate_hybrid.py` refuses stale GRU scores
  (nn_logp mtime must be ≥ nn_gru_best.pt mtime).
- Footprint accounting per table row: counts 21.4 MB (pickle) + gate KB +
  GRU 32.9 MB fp32 / 16.4 fp16.

## 5. Decision tree, risks, positioning

- **Load-bearing comparison:** GATE(6 experts incl. GRU) vs fixed-λ KN3+GRU.
  - Gate wins on both corpora → conference-short path (efficient-ML /
    TinyNLP-style venue; *SEM/ACL-short stretch).
  - Gate loses → reposition as routing-analysis workshop paper
    (methodology + oracle bounds carry it).
- Conference tier REQUIRES: PTB replication (Phase 5) + 3-seed ±sd on
  headline rows + energy/latency harness (Phase 6) + 5-gram baseline fix.
- Related work to cite/position against (VERIFIED — see §11): Mikolov et al.
  2011 (ASRU) + Sundermeyer et al. 2012 (fixed-weight n-gram/neural
  interpolation), Chang et al. 2015 (ASRU, discriminatively trained LM
  interpolation weights), Mathur et al. 2023 (PersonaLM, on-device
  personalization), plus the verified classic anchors in §11. Our lane:
  **learned evidence-aware multi-expert routing + oracle-bound measurement +
  footprint-matched fight**, toy-scale but fully reproducible.
- Watch items: max|logit| racing >~30 while valid stalls → stop early, keep
  best (per-epoch snapshots make this safe); Codespace idle suspension
  stretches wall-clock (harmless); gate seed variance → always ≥3 seeds.
- **Novelty positioning (locked 2026-09-12, tightened after external
  critique):** MoE-with-learned-router itself is NOT the claim — the paper
  must CONCEDE it specifically in the intro's FIRST paragraph, naming
  Jacobs, Jordan, Nowlan & Hinton 1991 (NOTE: four authors — cite as
  "Jacobs et al. 1991", NOT "Jacobs & Jordan") and Shazeer et al. 2017
  explicitly, before any reviewer
  can accuse us of hiding it. "We know this isn't new; here's exactly why
  we did it anyway." Three concrete claims:
  1. **The regime is the hook — attach numbers immediately:** a ~13 KB
     gate beats a tuned fixed-λ blend by ~34% PP, at a memory cost
     sparse-MoE papers wouldn't even report (beneath their scale). The
     justifying sentence: "the systems doing this at scale don't operate
     anywhere near this footprint, and nobody has checked whether the
     routing benefit survives when experts are frozen classical
     statistics + one small net."
  2. **Routing-signal finding:** votes × context-evidence interaction
     (ablations 202/368/~90–100).
  3. **Study rigor:** 2 corpora, seed ±sd, oracle bounds, negative
     results.
  - **Preempt the next-level objection:** "forgotten regime" overclaims —
    Mikolov-era small hybrid n-gram/neural interpolation EXISTS. Gesture
    at it in related work and pivot: "prior small-scale hybrid work
    interpolates with fixed weights; we test whether learned per-token
    routing changes that conclusion." That is claim (1) defended one
    level down.
  - Claiming "new architecture" = desk reject; this is an empirical-study
    paper: concede lineage → concrete regime numbers → the one new
    question (learned vs fixed routing at this scale).

## 6. Artifact map

- Scripts (post-audit, 2026-09-15): `mixer_lm.py` (experts + `top1()`),
  `mixer_cli.py`, **`gate_lm.py` (`FeatMaker` = the causal feature contract;
  `--collect` rebuilds the npz caches)**, `gate_lm2.py` (ablations),
  `gate_lm3.py` (`age_invariant` is now a pass-through), `gate_lm4.py`,
  `nn_expert.py`, **`gate_hybrid.py` (main table + `fair_router` +
  `assert_no_label_leak`; writes `gate_fair.json`, `gate_fair_h<H>.npz`,
  `gate_mlp_h<H>.npz`, `money_table.log`)**, **`seed_audit.py`**,
  **`hybrid_cli.py` (interactive tester; trains/caches the valid-tuned
  router on first use — the v1 note here about an h=128 gate with valid PP
  94.41 is stale and belongs to the leaky protocol)**, `progress.py`
- Benchmarks/audits: `benchmarks/{normalization_audit,router_diagnostic,
  energy_harness,selective_gru,footprint_meta,kenlm_baseline,fix_5gram,
  gate_sweep,run_benchmark,ptb_setup,cross_corpus}.py`,
  `scripts/run_paper_pipeline.sh` (one command per corpus),
  `scripts/make_submission.py` (generates `paper/submission/`),
  `scripts/auto_resume.sh`
- Tests: `tests/test_gate.py` (the normalization/leak guards),
  `tests/test_baselines.py` (sum-to-one for the n-gram baselines),
  `tests/test_model.py`. Run all: `python -m unittest tests.test_gate
  tests.test_baselines tests.test_model` (33 tests; `discover` does not work —
  `tests/` has no `__init__.py`)
- Caches: `benchmarks/results/gate_{train,valid,test}.npz`,
  `gold_streams.pkl`, `beast_trigram.pkl`
- Checkpoints: `nn_gru_last.pt` (resume), `nn_gru_best.pt` (best-by-valid),
  `nn_gru_contextblind_v1.pt` (archived v1, kept for the paper's honesty
  section), `nn_logp_{train,valid,test}.npy` (regenerated from the ep11
  expert; lengths asserted)
- Logs: `benchmarks/results/nn_train.log`; detached launch protocol
  (survives closed terminals): `setsid nohup python -u nn_expert.py >
  benchmarks/results/nn_train.log 2>&1 < /dev/null &`
- **Ops playbook (checked 2026-09-11):** the trainer is terminal-detached
  (`setsid`, TT = `?`) — closing VS Code does NOT kill it. The only threat is
  the Codespace idle timeout (default 30 min after editor disconnect; raise to
  240 min in GitHub settings): container stop kills the process. Recovery:
  relaunch with the command above — it prints `resuming: epoch N/30` and
  continues from `nn_gru_last.pt` (max loss = one partial epoch).
  Liveness check: `ps -o pid,etime,time,%cpu -p <pid>` — the TIME column must
  keep growing while disconnected.

## 8. THE MONEY TABLE (Phase 4 verdict, 2026-09-12) — ⚠️ INVALIDATED, see §0/§12

WT-2 test, 217,004 aligned positions, gate trained on train (model selection
on valid); GRU = epoch-11 expert (valid 162.16 / aligned-row table below):

| system | test PP ↓ | footprint |
|---|---|---|
| pure KN3 | 296.23 | 21.4 MB |
| pure GRU | 146.85 | 32.9 MB fp32 |
| gate[5 experts] | **99.68** | 21.4 MB + 6.5 KB |
| fixed-λ KN3+GRU | 135.84 (λ*=0.21 on valid) | 54.3 MB |
| static-α (6 experts) | 170.97 | — |
| **GATE hybrid (6 experts)** | **90.28** | 54.3 MB total, gate 13.5 KB |
| GATE hybrid, valid-trained | 77.49 | fair-data control (single seed — needs 3-seed) |
| oracle (6 experts) | 57.35 | bound |

**VERDICT: GATE BEATS fixed-λ 90.28 < 135.84 (−45.6 PP, −33%). The
load-bearing claim survives on WT-2. Conference path is live.**

Findings:
1. Hybrid dominates every component: 90.28 < gate[5] 99.68 < GRU 146.85.
2. Pareto story: gate[5] (21.4 MB) already DOMINATES pure GRU (32.9 MB,
   worse PP); adding the GRU to the gate improves further at +32.9 MB.
3. Static mixing fails (170.97) vs learned evidence routing (90.28) —
   6-expert confirmation of the §1 ablation finding.
4. Gate-vs-oracle gap widened to +32.9 PP (57.4% rel; was 12% at 5 experts):
   the GRU adds diversity the gate only partially exploits.

**Number-hygiene note:** pure-GRU PP 119.20 (batchified eval, printed by
nn_expert) vs 146.85 (aligned table row) is NOT a bug — the batchified eval
scores ALL targets incl. eos + sentence-initial words (easy wins for a GRU
with cross-sentence memory), the table scores exactly the gate's aligned
rows. The table is the apples-to-apples number; cite 146.85, footnote 119.20.

**3-seed audit (2026-09-12, seed_audit.py):**
| row | single seeds (test PP) | mean ± sd | 3-gate ENSEMBLE |
|---|---|---|---|
| GATE(6) train-trained h=128 | 90.28 / 125.57 / 87.47 | 101.11 ± 21.23 | **90.24** |
| GATE(6) valid-trained h=64 | 77.49 / 81.36 / 85.77 | 81.54 ± 4.14 | **79.18** |
| static-α (6) | 170.97 / 170.68 / 171.93 | 171.19 ± 0.66 | 171.19 |

Reading: train-trained h=128 has high seed variance (one bad basin, seed 1);
the seed-ensemble restores stable headline performance (90.24 ≈ the good
seeds). Valid-trained is tight. **fixed-λ (135.84) is dominated by every
gate variant at every seed and by both ensembles** — the load-bearing claim
is seed-robust. Recommended reporting: headline = GATE(6) ensemble 90.2
(footnote single-seed spread), control = valid-trained ensemble 79.2.

**Next for the paper (order):** PTB replication (Phase 5); 5-gram baseline
fix; energy/latency harness. Revisit: the ep25-30 anneal bet is moot for the
verdict — only worth finishing for camera-ready if a stronger expert is
wanted (table re-run costs ~40 min via --score-only + gate_hybrid).

## 7. Paper TODO (order matters) — ⚠️ re-scoped 2026-09-15 by §0/§12: the
paper is now an audit + budget-matched negative/positive result, not a "tiny
router wins by 30%" claim. Venue plan below still holds (ARR Oct 2026 cycle);
the selling point is the methodological finding plus the corrected two-corpus
study.



**Publication plan (re-scoped 2026-09-15 after the Phase-8 audit — see §0/§12):**

- **This is a WORKSHOP paper.** Content: the normalization audit
  (candidate-conditioned gating is not an LM; the 27–33% "win" was deflation) +
  the corrected budget-matched two-corpus study (routing buys 1.1–2.2% over a
  tuned static mixture and 5.3–13.2% over two-expert interpolation, at 13–66%
  extra latency) + the reliability-shift mechanism. Fit: negative-results /
  methodology / efficient-or-tiny-NLP workshops. **Venue TBD — check live CFPs
  before committing**; the 2026-09-13 check found EMNLP 2026 workshop deadlines
  (e.g. Insights, June 8) already closed, so the realistic windows are the next
  workshop cycle or an ARR short with a workshop commitment.
- **The conference paper is a DIFFERENT study, not an extension of this table**:
  compute saving at LLM scale (routing/skipping expensive components, possibly
  mixing a transformer layer with cheap experts). It must cite this workshop
  paper and disclose the overlap. Three carry-over obligations from §12: (i) the
  sum-to-one audit on any gated mixture, (ii) budget-matched baselines — a tuned
  static mixture, not a single tuned λ, (iii) stationarity of expert reliability
  between the tuning stream and deployment. Note the honest starting point: at
  this scale a skip rule on the router's own expensive-expert weight found
  almost nothing to skip (99.0%/89.7% usage at τ=0.5).
- Do NOT reuse any v1 number, figure or claim: §1, §8, §9, §10 router rows are
  invalidated. `paper/main.tex` is the source of truth;
  `python scripts/make_submission.py` regenerates the anonymized, line-numbered
  review copy (never hand-edit `paper/submission/`).
- Remaining before submission: venue + CFP conformance (page limit, anonymity),
  responsible-NLP checklist, ARR OpenReview profile, final proofread, and one
  last `bash scripts/run_paper_pipeline.sh {wt2|ptb}` to confirm every quoted
  number still regenerates.

1. [x] GRU v3 trained/stopped by decision (ep24); expert = ep11 checkpoint
2. [x] `python nn_expert.py --score-only` — aligned scores, assertions passed
3. [x] `python gate_hybrid.py` → verdict recorded (§8): GATE 90.28 BEATS
       fixed-λ 135.84; sanity anchor gate[5] 99.68 ≈ 99.7 (matches Phase 2)
4. [x] 5-gram baseline anomaly FIXED — fair-tuned 294.16 (was 302.52
       artifact); strengthens the story (§10)
5. [x] 3-seed ±sd on GATE(6) rows — WT-2: train 101.1±21.2 (ens 90.24),
       valid 81.5±4.1 (ens 79.18); PTB: 74.6±10.1 (ens 69.86) / 70.7±12.1
       (ens 65.37); fixed-λ dominated at every seed, both corpora (§8, §9)
6. [x] Phase 5: PTB replication (§9) — **VERDICT REPLICATES: GATE 68.02
       beats fixed-λ 92.65; every-seed dominance; ensembles 69.86/65.37**
7. [x] Phase 6: 5-gram fix + gate-size sweep + energy/latency harness (§10)
8. [x] Venue decision: ARR October 2026 cycle (deadline Oct 12) → commit to
       NAACL 2027 or COLING 2027 (Dec 23); extension → ACL 2027 Jan cycle;
       TMLR backup. Anonymized copy ready in `paper/submission/`

## 9. Phase 5 — PTB replication (Mikolov split) — ⚠️ router rows INVALIDATED, see §0/§12

**Purpose:** kill the "WT-2 quirk" objection — does the money-table verdict
(gate > fixed-λ, hybrid dominates components) replicate on a second corpus?

**Protocol notes (all WT-2 rules preserved):**
- Data: tomsercu/lstm Mikolov split → `data/ptb/`; OUR tokenizer (identity
  check OK), vocab from train (9,644), OOV→unk. Gold streams: train
  907,138 / valid 71,825 / test 80,594 tokens — EXACTLY the base repo's
  cross-corpus counts (same protocol).
- Path isolation: `HEDGE_RESULTS=.../benchmarks/results/ptb` env override
  added to mixer_lm/nn_expert/gate_lm*/gate_hybrid/hybrid_cli/progress —
  WT-2 artifacts untouched. `HEDGE_WARMUP` env for GRU warmup steps.
- **No per-corpus hyperparameters:** KN discount_scale kept at 1.15 (WT-2's
  value; PTB valid grid {1.0–1.5} flat within 0.4%).
- ⚠️ **Knob-bounding warning (appendix-grade):** the repo's
  `_recompute_discounts` scales discounts linearly with `discount_scale`
  with NO bound; at ds≥1.7 discounts exceed counts (D2 7.34 at ds=6),
  probabilities stop normalizing, and apparent PP "gains" (190→59!) are
  fake. Anyone tuning this knob must bound it to ≲1.5. Explains why
  cross-corpus study grids stayed low.
- Gate feature collection: rows 851,095 / 67,396 / 75,623 = exactly
  Σ(len(s)−1) per split (alignment rule verified on PTB).
- GRU: recipe v3 verbatim (seed 42, Adam 1e-3, clip 1.0, bs32/tsl64,
  4 threads, cosine to 1%), warmup 442 steps (≈0.94 PTB epoch), V=9,645,
  3.3M params, 470 steps/epoch ≈ 6.6 min.
- h-selection: same valid-based protocol (gate_hybrid grid 64/128/256);
  gate_lm preview says h=64 wins on PTB (valid PP 80.0 vs 124.9 @h=128 —
  PTB overfits faster: 851k rows vs 1.85M). Seed audit uses HEDGE_GATE_H=64.

**Phase-2 replication preview (gate_lm.py on PTB):** static mixing fails
again (equal 196.2, learned static-α 152.8 vs 5-expert oracle 64.8);
neural gate h=64 → test PP 75.9, beats static bucket on 96.2% of
positions. Routing stats replicate: kn3 58% / cache 18% / lag2 13% /
lag3 10% / uni 0%.

**Money table (PTB test, 75,623 aligned positions) — VERDICT REPLICATES:**
| model | PTB test PP | (WT-2 for comparison) |
|---|---|---|
| pure KN3 | 165.15 | 296.23 |
| pure GRU (aligned / batchified) | 103.01 / 86.32 | 146.85 / 119.20 |
| gate[5 experts] | 72.73 | 99.68 |
| fixed-λ KN3+GRU | 92.65 (λ*=0.28) | 135.84 (λ*=0.21) |
| static-α (6 experts) | 101.74 | 170.97 |
| **GATE(6) train-trained** | **68.02** (h=24) | 90.28 (h=128) |
| GATE(6) valid-trained | 63.93 | 77.49 |
| oracle (6 experts) | 44.37 | 57.35 |

**VERDICT: GATE 68.02 BEATS fixed-λ 92.65 (−27%) — load-bearing claim holds
on TWO corpora.** After widening the h-selection grid to {24,64,128,256}
(Phase 6 sweep), PTB valid selects h=24 → 2.6 KB gate, test 68.02 (was
78.81 @h=128 with the 3-point grid). WT-2 unchanged by the wider grid
(h=128, 90.28). Gate-vs-oracle gap: PTB 53.3%, WT-2 57.4%.

**3-seed audit (PTB, h=24 headline):** gate6-train 74.56 ± 10.05
(68.02/86.14/69.53, ensemble **69.86**); gate6-valid 70.73 ± 12.06
(63.93/63.60/84.65, ensemble **65.37**); static-α 101.88 ± 0.14.
**Worst seed (86.14) still beats fixed-λ (92.65)** — every-seed dominance
on both corpora (WT-2: worst 125.57 < 135.84). h=24 has wide seed variance
on small data — report ±sd and ensembles (ensemble ≈ seed-0 here).

**Honest wrinkle (report in paper):** on PTB (half the data) train-trained
GATE(6)@h=128 78.81 LOSES to gate[5] 72.73 (WT-2: 90.28 BEAT 99.68) —
overfits the smaller corpus; the valid-selected h=24 (68.02) and the
valid-trained gate (63.93) both beat gate[5]. The load-bearing comparison
(gate vs fixed-λ) is unaffected. fixed-λ picked a GRU-heavier λ*=0.28
(WT-2: 0.21) yet still lost — fixed interpolation can't reweight per
position. Non-monotonic h=64 PTB point (87.47 test) = single-seed basin;
sd/ensemble context provided by the audit.

**Phase 5 COMPLETE (2026-09-12).** Two-corpus claim secured →
conference-short path (§5). Remaining: Phase 6 (5-gram fix + energy
harness), Phase 7 writing per §5 positioning.

## 10. Phase 6 — robustness & cost — ⚠️ PARTLY INVALIDATED (gate rows, cost table, selective activation): see §0/§12

**5-gram baseline anomaly RESOLVED:** the original 302.52 row tuned the
5-gram with discount_scale frozen at 1.0 (3 combos) vs the trigram's
30-combo grid. Same-grid fair re-tune (`benchmarks/fix_5gram.py`):
**294.16 test PP** (scale=1.15, z=0.1, adaptive=False — near the trigram's
optimum). Story strengthens: order-5 buys ≈nothing over order-3 on WT-2
(294.2 vs 292.3) — classical n-gram scaling is exhausted at this corpus
size, while the hybrid (90.28) beats the whole family ~3×.

**Gate-size sweep (`benchmarks/gate_sweep.py`, GATE(6) train-trained, seed 0):**
| gate | KB | WT-2 valid | WT-2 test | PTB valid | PTB test |
|---|---|---|---|---|---|
| h=24 | 2.6 | 148.49 | 144.06 | **71.27** | **68.02** |
| h=64 | 6.8 | 96.51 | 92.17 | 93.41 | 87.47 |
| h=128 | 13.5 | **94.41** | **90.28** | 84.33 | 78.81 |
| h=256 | 27.0 | 123.95 | 120.46 | 90.44 | 84.43 |

Optimum is corpus-dependent (13.5 KB on WT-2, 2.6 KB on PTB) and
non-monotonic; valid selection picks the right one on both. Both optima
are far below any neural-component footprint — the paper's "tiny router"
claim now has a curve, not a point.

**Energy/latency harness (`benchmarks/energy_harness.py`)** — method: 200
test sentences, causal token-by-token stream, best of 3 repeats; energy =
WALL time × 15 W TDP proxy (stated as proxy, not measured power);
interactive batch-1 GRU (worst case).
| model | WT-2 ms/tok | WT-2 mJ/tok | PTB ms/tok | PTB mJ/tok |
|---|---|---|---|---|
| kn3 | 0.004 | 0.05 | 0.003 | 0.05 |
| mixer5 | 0.011 | 0.16 | 0.009 | 0.14 |
| gru | 5.089 | 76.3 | 1.100 | 16.5 |
| fixed-λ | 3.340 | 50.1 | 1.208 | 18.1 |
| **gate6 (full)** | **3.597** | **54.0** | **1.022** | **15.3** |

Reading: the GRU dominates cost; the gate's share is ~8% (WT-2) /
negligible (PTB, where gate6 ≈ fixed-λ within noise). Full system:
~280 tok/s interactive on 4 CPU cores. Sizes: WT-2 total ≈ 68.7 MB
(tri 21.4 + lag 14.3 + cache/uni 0.3 + GRU 32.7 + gate 0.013); PTB ≈
28.7 MB (GRU 13.1). Quality-per-joule: gate6 ≈ fixed-λ cost, −34% PP
(WT-2) / −27% PP (PTB).

### Toolkit baseline — KenLM (2026-09-13, pre-submission add)

`benchmarks/kenlm_baseline.py`: KenLM (Heafield 2011) trained with lmplz
defaults on OUR dumped gold streams, scored under the exact aligned
protocol (per-sentence full_scores, drop sentence-initial targets; row
counts asserted = 217,004 / 75,623; lmplz unigram count 1,927,034 = our
train stream Σlen ✓).

| order | WT-2 test | PTB test | footprint (trie binary) |
|---|---|---|---|
| 3-gram | 257.32 | 146.34 | 57.2 MB arpa / 26.2 MB |
| 4-gram | 251.96 | 139.79 | 126.3 / 56.7 MB arpa |
| 5-gram | **250.89** | **138.42** | ~~53.8 MB / 23.1 MB~~ (see note) |

> ⚠️ Footprint note (2026-09-15): the perplexities in this table stand and
> re-verify (`kenlm.log`), but the MB column does not: it was a `build_binary`
> trie measured by hand, and `build_binary` is not installed in this
> environment, so the number is not reproducible from committed code. Do not
> quote it. `kenlm_baseline.py` now logs the ARPA size instead.

Reading: KenLM's SGI-modified KN beats our trigram expert (296.2 → 250.9
WT-2; 165.2 → 138.4 PTB) — expected of a battle-tested toolkit — but sits
~2.8× / 2× WORSE than the hybrid (90.3 / 68.0) and far above even the
5-expert gate (99.7 / 72.7). Order scaling flat (257→251, 146→138)
independently confirms the "classical n-gram scaling exhausted" finding.
Added to the paper money table + limitations rewritten. Logs:
`benchmarks/results/kenlm.log`, `benchmarks/results/ptb/kenlm.log`.

### Selective expert activation — cost-quality frontier (2026-09-13, pre-submission add)

**Motivation (oracle concentration):** the GRU is the best expert on only
~43% of tokens (WT-2) / ~41% (PTB), and the selective-GRU oracle reaches
FULL oracle PP while running on just 50% of positions (WT-2: 57.35 at 50%
usage; PTB: 44.37 at 50%). The expensive expert's value is concentrated.

**Realizable scheme (`benchmarks/selective_gru.py`):** the 6-expert gate's
GRU weight alpha_gru(x) is computed from the 5 cheap votes + evidence
only — it never reads the GRU's output — so it is a FREE, causal skip
signal. Rule: consult the GRU iff alpha_gru >= tau; skipped tokens use the
gate's renormalized cheap weights. Anchors reproduce exactly (6-gate
90.28 / 68.02; 5-gate 99.68 / 72.73).

| tau | WT-2 usage | WT-2 PP | PTB usage | PTB PP |
|---|---|---|---|---|
| 0 (always on) | 100% | 90.28 | 100% | 68.02 |
| 0.02 | 46.7% | 90.69 | 54.7% | 68.22 |
| 0.10 | 33.3% | 91.29 | 40.7% | 68.71 |
| 0.50 | 16.9% | 93.78 | 21.8% | 70.72 |

Reading: **~half the GRU compute for <=0.5 PP** on both corpora; at
tau=0.50 the system runs the GRU on ~1/5 of tokens and still beats the
5-expert gate (93.8 < 99.7; 70.7 < 72.7). Since the GRU dominates system
cost (5.1 of 3.6 ms/tok class), tau~0.02 cuts total energy roughly in
half at +0.45% PP. This is the bridge from the paper to LLM inference
economics (skip-the-big-model tokens) and pillar #3 of the conference
extension. Logs: `selective_gru.log` in both results dirs.

**Phase 6 COMPLETE. Remaining: Phase 7 writing only.**

## 11. Related-work verification (2026-09-12)

Every planned citation was checked against live metadata (Crossref / arXiv /
dblp APIs). Status per claim:

### VERIFIED — cite as-is
- **Jacobs, Jordan, Nowlan & Hinton 1991** — "Adaptive Mixtures of Local
  Experts", Neural Computation 3(1), doi:10.1162/neco.1991.3.1.79.
  Abstract confirms the MoE content. CAUTION: FOUR authors — cite as
  "Jacobs et al. 1991" (the dossier's earlier "Jacobs & Jordan 1991" label
  conflates this with Jordan & Jacobs 1994).
- **Jordan & Jacobs 1994** — "Hierarchical Mixtures of Experts and the EM
  Algorithm", Neural Computation 6(2), doi:10.1162/neco.1994.6.2.181.
  (Optional second lineage cite; our per-position supervised gate is closer
  to this than to online exp-weights.)
- **Shazeer et al. 2017** — "Outrageously Large Neural Networks: The
  Sparsely-Gated Mixture-of-Experts Layer", arXiv:1701.06538 (ICLR 2017).
  Title confirmed on arXiv.
- **Mikolov, Deoras, Povey, Burget & Černocký 2011** — "Strategies for
  training large scale neural network language models", IEEE ASRU 2011,
  doi:10.1109/asru.2011.6163930. THE pointer for "Mikolov-era small hybrid
  interpolation" (RNN + KN interpolated with a fixed heldout-tuned weight);
  also validates our PTB split lineage. Supports the §5 pivot sentence.
- **Sundermeyer, Schlüter & Ney 2012** — "LSTM neural networks for language
  modeling", Interspeech 2012, doi:10.21437/interspeech.2012-65. Second
  fixed-weight hybrid interpolation pointer.
- **Chang, Lahiri, Alphonso, Oguz & Levit 2015** — "Discriminative training
  of context-dependent language model scaling factors and interpolation",
  IEEE ASRU 2015, doi:10.1109/asru.2015.7404772. Fixed/discriminatively
  trained interpolation — REPLACES the dossier's "Levit 2023" (misremembered;
  Levit is the last author here).
- **Mathur et al. 2023** — "PersonaLM: Language Model Personalization via
  Domain-distributed Span Aggregated K-Nearest...", Findings of EMNLP 2023,
  doi:10.18653/v1/2023.findings-emnlp.757. On-device personalization pointer.
- **Freund & Schapire 1997** — "A Decision-Theoretic Generalization of
  On-Line Learning and an Application to Boosting", J. Comput. Syst. Sci.,
  doi:10.1006/jcss.1997.1504. The Hedge algorithm — project namesake;
  cite for prediction-with-expert-advice lineage of the oracle bound.
- **Kneser & Ney 1995** — "Improved backing-off for M-gram language
  modeling", ICASSP 1995, doi:10.1109/icassp.1995.479394.
- **Merity et al. 2017** — "Regularizing and Optimizing LSTM Language
  Models", arXiv:1708.02182 (AWD-LSTM; WT-2/PTB anchor numbers).
- **Zaremba, Sutskever & Vinyals 2014** — "Recurrent Neural Network
  Regularization", arXiv:1409.2329 (PTB LSTM anchor numbers).

### CANONICAL — cite from memory, re-confirm metadata at camera-ready
(page checks blocked in this sandbox; papers are unambiguous classics)
- Bengio, Ducharme, Vincent & Jauvin 2003 — "A Neural Probabilistic Language
  Model", JMLR 3:1137–1155 (jmlr.org/papers/v3/bengio03a.html exists).
- Chen & Goodman 1999 — "An Empirical Study of Smoothing Methods for
  Language Models", Computational Linguistics 25(1):73–100.
- Jelinek 1980 — "Interpolated estimation of Markov source parameters from
  sparse data", Pattern Recognition in Practice, North-Holland, 381–397
  (predates the neural hybrids; the origin of interpolation).
- Mikolov et al. 2010 — "Recurrent neural network based language model",
  Interspeech 2010 (RNN-LM origin).
- Cesa-Bianchi & Lugosi 2006 — "Prediction, Learning, and Games", Cambridge
  UP (oracle/competitor bound formalism; §"the hindsight oracle" name-drop).

### DROPPED — could not verify; do NOT cite
- "Sak 2013" — no such LM paper; nearest real match is Sak, Senior &
  Beaufays 2014 (Interspeech, doi:10.21437/interspeech.2014-80) which is
  ASR, off-topic. Fixed-interpolation pointers are Mikolov 2011 + Sundermeyer
  2012 instead.
- "Levit 2023" — replaced by Chang et al. 2015 (above).
- "Qin 2023" — could not identify any relevant paper after repeated
  searches. Drop.
- "Zhong 2025" — only junk/unrelated matches. Drop.

### Bibliography discipline for Phase 7
1. Cite only from the VERIFIED + CANONICAL lists above.
2. Every citation gets a DOI or arXiv ID in the .bib.
3. Before submission: batch re-check all DOIs resolve (one curl loop).

## 12. The v1 invalidation — forensic record (2026-09-15)

**How it was found:** a pre-submission accuracy audit of `paper/main.tex`
against the artifacts. Three checks, in order of damage:

1. `np.array_equal(X[:, :5], L[:, :5])` on every collected split → **True**.
   `gate_lm.collect()` computed `x[0:5] = log p_i(w_gold)` and then assigned
   `L[n, 0:5] = x[0:5]`. The router was fed the answer. (`gate_lm.py:6` even
   asserted "all causal, no leakage".) Feature `[12]` was the gold word's cache
   count — the same bug one column over.
2. **Is the mixture a distribution?** For a candidate-conditioned gate,
   `Z(ctx) = Σ_w Σ_i a_i(x_{ctx,w}) p_i(w|ctx)`. Measured over the full
   vocabulary with the cache streamed causally: **Z = 2.66 (WT-2)**,
   **2.56 (PTB)** for the v1 router; **1.000** for KN3, the GRU, fixed-λ and
   the corrected router. `exp(mean log Z)` is exactly the factor by which the
   reported perplexity was deflated: 90.28 → ≈240, 68.02 → ≈174.
3. **Are the baselines tuned the same way?** No: `train_static_alpha` was fit on
   *train* while λ* and the router's hidden size were selected on *valid*. Fit
   on valid, the same static mixture goes 170.97 → **119.14** (WT-2) and
   101.74 → **89.71** (PTB), i.e. the "static mixing fails badly" row was a
   straw man. (It is not an optimizer bug: EM from four inits converges to the
   same train-fit α, kn3 .68 / lag3 .09 / gru .22. Train is simply the corpus
   the GRU memorized — train PP 46.6 vs test 103.0 on PTB.)

**Why nobody caught it earlier:** the loss decreased, validation tracked test,
three seeds agreed, two corpora replicated, an ensemble helped, and the router
landed plausibly between the baselines and the "oracle" (which had the same
defect: its mass is ≈3.1). Every consistency check was internal to the broken
protocol. Only an external invariant — Σ_w P(w|ctx) = 1 — catches it.

**Secondary errors found in the same audit (all fixed):**
- `paper/main.tex` claimed fixed interpolation "prefers a more GRU-heavy
  mixture on PTB (λ*=0.28 vs 0.21)". λ is the weight on **KN3**
  (`fixed_lambda(i=0, j=5)`: `p = λ·p_kn3 + (1−λ)·p_gru`), so PTB is *less*
  GRU-heavy. The same inversion was in §9 of this dossier.
- The paper called the comparison "fixed interpolation **of the same experts**"
  / "identical experts"; the baseline interpolates 2 of the 6.
- Table 1's WT-2 column quoted the literature (≈2.0M / 214k / 245k / 33k) rather
  than the gold streams actually scored (1,927,034 / 201,797 / 226,731 / 28,714;
  `common.load_wikitext2` drops `= header =` lines and lowercases). PTB was right.
- The footprint column of the main table used WT-2 sizes for both corpora
  (PTB: KN3 9.2 MB, GRU 13.1 MB, system 28.7 MB) and omitted the lag/cache/
  unigram experts (14.6/…MB) that the 6-expert rows need — contradicting the
  "total footprint 68.7/28.7 MB" sentence in the same paper. Now measured by
  `benchmarks/footprint_meta.py` into `nn_meta.json`.
- §3 described the router input as "each expert's vote (top prediction, its
  probability, and rank statistics)". No rank statistics existed and the GRU's
  vote was never an input (which the Conclusion relied on). The description now
  matches `FeatMaker`: 5 confidences + 5 agreement flags + 14 context stats.
- The intro called the router "one of the largest learnable surface areas"; it
  is 3.9k parameters against the GRU's 3.3–8.2M.
- The v1 cost table was physically impossible: GRU alone 5.09 ms/token vs the
  full hybrid that contains it 3.60 ms/token (CPU time showed the same
  inversion: 17.1 vs 13.6). Re-timed interleaved, GRU-alone and GRU+KN3+blend
  are the same within noise. The harness now quotes the minimum of ≥5
  interleaved rounds on one torch thread and asserts the containment ordering.
- The "hindsight oracle" was described as "the classic competitor bound
  (Freund & Schapire; Cesa-Bianchi & Lugosi)" and "no realizable system can
  beat it". It is the per-position max over experts, which is unnormalized; the
  Hedge competitor bound is a different (weaker) object. Now reported as a
  diagnostic and labelled as not achievable.
- `paper/submission/main.tex` had drifted from `paper/main.tex`: it dropped the
  ACL template for fontspec/geometry (so: no line numbers, wrong venue format,
  and it could not even compile with the pdflatex in this image) and replaced
  `references.bib` with a hand-typed bibliography containing five
  misattributions (see §0). It is now generated by
  `scripts/make_submission.py`.

**Code changes (2026-09-15):**
- `gate_lm.py`: `FeatMaker` — 24 causal, candidate-independent features
  (5 expert confidences in their own favourites, 5 plurality-agreement flags,
  14 context statistics incl. two age-invariant cache densities). Dropped the
  sentence-length and position-fraction features: both need the sentence END,
  which is future information at scoring time. `--collect` rebuilds the caches.
- `mixer_lm.py`: `top1()` on every expert (KN3 memoized per bigram context;
  lag-2/3 from the existing top-16 lists; cache via a lazily-deleted max-heap;
  unigram leaders cached) + `CacheExpert.clear()`.
- `gate_hybrid.py`: `assert_no_label_leak()` on load; `fair_router()` (h by
  2-fold CV inside valid, weights fit on valid); static-α reported train-fit
  **and** valid-fit; λ printed with its semantics; footprint from
  `nn_meta.json`; "best single expert" (normalized) added and the per-position
  max relabelled as unnormalized; verdict compares against the strongest tuned
  baseline; writes `gate_fair.json` + `gate_fair_h<H>.npz`.
- `seed_audit.py`: audits the headline (valid-tuned) router and both static
  rows; the train-trained control is single-seed by design (documented).
- `benchmarks/energy_harness.py`: `router-only` and `static6` rows, single
  torch thread, ≥5 interleaved rounds, min-of-rounds, containment assertion.
- `benchmarks/normalization_audit.py` (new): measures Z over the full
  vocabulary for the current router and, with `--legacy`, reconstructs the v1
  leaky features and re-measures the artifact from the archived npz files.
- `benchmarks/router_diagnostic.py` (new): per-split expert PPs, router PP,
  the router's own mean weights applied statically, uniform, and both static
  fits — the mechanism behind the train-trained failure.
- `benchmarks/footprint_meta.py` (new): one measured source for every footprint.
- `tests/test_gate.py` (new): features identical for diverging futures; every
  expert sums to 1; any context-only mixture sums to 1; cache `top1()` matches
  a rescan; cached npz has no feature/label column identity or near-1
  correlation.
- `scripts/run_paper_pipeline.sh` (new): one command per corpus for every
  router-dependent number. `scripts/make_submission.py` (new): generates the
  anonymized ACL review copy.
- Invalidated artifacts + the code revision that produced them (`9d2eac4`) are
  archived under `benchmarks/results/*/legacy_leaky_v1/` with a README.

**What the paper now claims** (`paper/main.tex`): (1) the audit — a
candidate-conditioned gate is not an LM, here is the one-line check, and here is
a 27–33% "win" that was pure deflation; (2) budget-matched, causal per-token
routing buys 13.2%/5.3% over tuned two-expert interpolation but only 1.1%/2.2%
over a tuned static mixture of the same experts; (3) train-trained routing is
worse than uniform mixing, because expert reliability is not stationary
(`router_diagnostic.py`); (4) the tuned static hybrid is the real small-footprint
win (119.1/89.7 vs KenLM-5 250.9/138.4 and GRU-alone 146.9/103.0).
