# Benchmark results (WikiText-2)

Test tokens: 217,004 · one shared gold token stream · one harness loop for every model · trigram order unless noted · pure-stdlib models, single core

Context: all rows are classical n-gram models. The best widely reported
neural result on the same benchmark (AWD-LSTM ≈ 65.8, Merity et al., 2018 —
different tokenizer, literature value) is far below this class. No
transformer number is cited because transformer LM papers report on
WikiText-103/PTB/enwik8, not WikiText-2; see README.md "How to read that
table honestly" for the correctness-check framing of the textbook baselines.

| Model | Test PP ↓ | Top-1 ↑ | Top-3 ↑ | Train+tune (s) | Size (MB) | ms/sugg |
|---|---|---|---|---|---|---|
| Trigram Witten-Bell | 246.4 | 15.8% | 27.3% | 4.09 | 64.7 | 0.2 |
| BEAST (AFTER, tuned) | 292.3 | 19.2% | 32.8% | 200.99 | 21.4 | 0.7 |
| MKN + fixed Zipf prior | 296.3 | 19.2% | 32.6% | 0.0 | 21.4 | 0.7 |
| BEAST 5-gram | 302.5 | 19.2% | 32.6% | 70.87 | 102.7 | 0.7 |
| Modified Kneser-Ney (no prior) | 306.1 | 19.2% | 32.6% | 0.0 | 21.4 | 0.7 |
| Bigram interp (Markov-style) | 332.6 | 17.5% | 31.3% | 1.49 | 11.1 | 0.1 |
| Zipf v4 (BEFORE, tuned) | 335.9 | 18.5% | 31.6% | 97.59 | 325.2 | 43.7 |
| Simple Kneser-Ney | 378.4 | 19.2% | 32.4% | 0.0 | 21.4 | 0.7 |
| Interpolated trigram MLE | 416.5 | 18.8% | 31.8% | 8.12 | 149.5 | 43.3 |
| Unigram MLE (floor) | 825.2 | 6.5% | 16.4% | 0.42 | 0.6 | 0.0 |
| Bigram add-1 (Laplace) | 2053.2 | 17.4% | 30.9% | 1.64 | 7.4 | 0.3 |

PP = word-level perplexity, natural log (lower is better).
Top-k = next-word suggestion accuracy on the test stream (higher is better).
Size = raw pickle of the fitted model (what save() writes, before compression).