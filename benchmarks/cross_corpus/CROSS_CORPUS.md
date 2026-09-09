# Cross-corpus validation of the Zipf-continuation prior

**Question.** Does the prior's measured gain over plain Modified Kneser-Ney
(-9.8 PP, 3.3% on WikiText-2) hold outside WikiText-2, and does it beat
another named sparse-region regularizer — Witten-Bell's confidence weighting
— when both are installed in the same junction of the same backbone?

**Answer.** Yes and yes, so far: the prior improves test perplexity on all
four corpora tested (-1.9% to -2.7%), and Witten-Bell weighting degrades it
on all four.

## Design

* **Corpora** (different sizes and domains):
  - `ptb` — Penn Treebank standard LM split (WSJ news, 907k train tokens;
    Mikolov split via tomsercu/lstm)
  - `brown` — Brown corpus, 15 genres, detagged; seeded (42) shuffled file
    split 80/10/10 (928k train tokens)
  - `wikitext2` — the original benchmark corpus (1.93M train tokens)
  - `wt103prefix` — WikiText-103 official splits, first 8M train tokens,
    first 60k valid/test tokens (compute-budget prefix)
* **Protocol** (identical to the main benchmark where applicable):
  - one shared gold token stream per corpus (the predictor's own tokenizer),
    OOV → `unk` in valid/test
  - one trained trigram backbone per corpus; three variants tuned and scored
    on it with **explicit, non-overlapping grids**:
    1. `mkn` — plain Modified Kneser-Ney: discount scale tuned, prior
       structurally off
    2. `witten_bell` — textbook Witten-Bell in the same junction
       (P = c/(T+D) + D/(T+D)·P_bo): **parameter-free**, no tuning
    3. `zipf_prior` — the novel prior: discount scale × prior ceiling ×
       adaptive switch tuned
  - tuning: capped valid text (30k tokens; wt103: 10k — its 107k vocabulary
    makes trials ~3x slower), same grids for every corpus except as noted
    per-corpus in the result JSONs
  - scoring: the model's exact `_p_kn` scorer + exact top-k (same as
    `evaluate()`)

## Results (test perplexity)

| Corpus | Domain | Train | MKN | + Witten-Bell | + Zipf prior | Prior Δ vs MKN |
|---|---|---|---|---|---|---|
| Penn Treebank | news | 0.91M | 177.5 | 192.1 | **172.9** | **-2.6%** |
| Brown | 15 genres | 0.93M | 882.4 | 1011.3 | **865.6** | **-1.9%** |
| WikiText-2 | wiki | 1.93M | 300.2 | 321.1 | **292.1** | **-2.7%** |
| WikiText-103 prefix | wiki | 8.0M | 313.1 | 329.0 | **304.5** | **-2.7%** |

Tuned prior ceiling: **0.25 on all four corpora**. Adaptive trust curve: lost
to a constant prior on all four corpora (consistent with the main report's
ablation).

## Findings

1. The WikiText-2 gain replicates: -1.9% to -2.7% across three new corpora,
   two new domains, and a 4x range of corpus sizes. It does not decay with
   more data (-2.7% at 8M tokens).
2. The prior beats the named prior-art comparator in the identical junction
   on every corpus. Witten-Bell weighting *hurts* relative to plain MKN,
   reproducing the classical Chen & Goodman (1998) ordering.
3. The study's WikiText-2 MKN baseline (300.2) is slightly stronger than the
   main report's fixed-scale 306.1 because the study tunes the discount
   scale; the prior still improves on it by 8.1 PP.

## Scope limits (stated plainly)

* One n-gram order (trigram).
* One prior-art comparator. A hierarchical Pitman-Yor LM is the other obvious
  resident of this space and remains future work.
* `wt103prefix` uses the first 8M tokens, not the full 101M-token corpus;
  its tuning cap and grids are recorded in its result JSON.

## Reproduce

```bash
python benchmarks/cross_corpus.py --corpus ptb
python benchmarks/cross_corpus.py --corpus brown
python benchmarks/cross_corpus.py --corpus wikitext2
python benchmarks/cross_corpus.py --corpus wt103prefix
```

Raw outputs: `benchmarks/cross_corpus/results/<corpus>.json`.
