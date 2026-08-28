# Solution log

Running record of what has been tried on top of the iteration-0 baseline. The
Research agent reads this as "current model state"; the Coding agent gets the
Current baseline section pasted into its prompt as the code it is editing.

---

## Iteration 0 -- FM baseline

Seeded by hand, but it is a *real* iteration 0: `Orchestrator.bootstrap_baseline()`
runs it through the executor before the research loop starts, so it appears in
`logs/runs.jsonl` as iteration 0 and becomes the registry incumbent.
Research-driven iterations are numbered from 1.

`solution/train.py` is a direct port of `harness/baseline.py`'s `run_fm`,
wrapped in the executor's `--config/--seed/--out` contract. Same model, same
Adam settings, same early stopping, so it reproduces the published numbers
rather than being a near-miss re-implementation.

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| published FM (test) | 0.6610 | 0.5282 | **0.5946** |
| ours, seed 0 (test) | 0.6621 | 0.5286 | **0.5953** |
| ours, seed 1 (test) | 0.6609 | 0.5287 | **0.5948** |
| ours, seed 0 (valid) | 0.6671 | 0.5358 | **0.6015** |

Published seed std is 0.0008, so both seeds land inside noise. Validation
matches the published 0.6016 to four decimals.

Config: `k=16, lr=0.001, l2=1e-6, batch=8192, max_epochs=40, patience=4`,
fields `[user_id, video_id, author_id, tab, dur_bucket]`, pointwise logloss.

**This is the bar to beat.** Anything the agent loop produces is compared
against 0.6015 on validation (never against the test number -- that lives in
`logs/quarantine/` and no agent can see it).

---

## Known dead ends (measured by the organisers -- do not re-spend iterations)

| tried | result | why |
|---|---|---|
| More static features (all 13 CWM feature domains) | 0.5940 vs 0.5950 | `user_id x video_id` already absorbs the signal; coarse user buckets are redundant given `user_id` |
| More capacity (k = 8 / 16 / 32) | 0.5895 / 0.5902 / 0.5887 | 1.14M rows can't support a bigger model |

Also structural, and worth knowing before proposing a feature: **a pure
user-side first-order term contributes exactly zero.** Ranking happens within
a user, so any term constant across that user's impressions cannot reorder
them. User features can only pay off through crosses with item-side features.

---

## Unexplored directions (organisers' ranking, highest expected value first)

1. **Ranking loss instead of pointwise logloss.** The metric is GAUC/nDCG --
   ranking -- but the objective is logloss. BPR (pairwise) or a within-user
   softmax (listwise) aligns the two. Judged most likely to pay off, and it is
   contained: same features, same model, different gradient.
2. **User behaviour sequences.** Nothing currently uses them; each user has
   hundreds to thousands of train interactions. DIN/SIM-style interest
   modelling is completely untouched here.
3. **Multi-task.** `is_click`, `is_like`, `is_follow`, `is_comment`,
   `is_forward`, `play_time_ms` as auxiliary heads on the `long_view` task.
4. **Watch-time modelling.** Censored regression (CWM's contribution):
   watch time is truncated when a video completes, so a one-sided loss beats
   squared error.
5. **Different architecture** (DeepFM / DCN / xDeepFM). Deprioritised --
   capacity is measurably not the bottleneck.
6. **Time features and drift.** `hourmin`, `date`, train-vs-test drift.
7. **Unbiased validation.** `log_random_4_22_to_5_08_pure.csv` is a random
   exposure log (1.18M rows) usable as an unbiased check for overfitting to
   biased traffic.

## Headroom, for calibration

Oracle (true labels as scores) scores 0.8645 primary on test, not 1.0 -- 27.1%
of test users are all-negative, so their nDCG is 0 for any model. The FM
baseline has already taken ~31% of the reachable range. Remaining headroom is
0.27, not 0.41.

---

## Iterations

_(appended by the agent loop; see `logs/runs.jsonl` for the machine-readable
record and `docs/results.md` for the run report)_
