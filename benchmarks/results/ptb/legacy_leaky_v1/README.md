# INVALIDATED artifacts — leaky router features (v1, 2026-09-09 .. 09-13)

Every gate/router number in these logs was produced with a router whose input
features `[0:5]` (and `[12]`) were functions of the **word being scored**:
`gate_lm.collect()` stored `x[0:5] = log p_expert(w_gold)` and then set
`L[:, 0:5] = x[:, 0:5]`, i.e. the features WERE the labels.

Consequences (measured, see `tests/test_gate.py`):

* the router's weights depended on the candidate word, so
  `P(w|ctx) = sum_i a_i(x_w) p_i(w|ctx)` was **not a distribution**: its total
  mass over the vocabulary was 2.68 (WikiText-2) / 2.44 (PTB) instead of 1.000
  (KN3, GRU and the fixed-interpolation baseline all measure 1.000);
* every gate perplexity was therefore deflated by roughly that factor.
  Re-normalizing the same trained router gives ~242 (WT-2) / ~166 (PTB)
  instead of the 90.28 / 68.02 reported here — i.e. worse than the tuned
  fixed interpolation it claimed to beat (135.84 / 92.65);
* the "hindsight oracle" rows (57.35 / 44.37) have mass ~3.1 and are not a
  bound on any realizable model.

Also invalidated here: the `static-alpha` rows, which were fit on TRAIN while
every other tuned row (fixed lambda, gate hidden size) was selected on VALID.
Fit on validation the same static mixture reaches 118.9 (WT-2) / 88.9 (PTB).

Unaffected by the leak (kept in place): `kenlm.log`, `fix_5gram.log`,
`nn_*.log`, the GRU checkpoints and the aligned `nn_logp_*.npy` scores, and
every pure-expert / fixed-interpolation row.

The code that produced these numbers is in git history at commit `9d2eac4`.
The fix: `gate_lm.FeatMaker` (candidate-independent features) + the guards in
`gate_hybrid.assert_no_label_leak()` and `tests/test_gate.py`.
