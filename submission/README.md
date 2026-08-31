# Final result: multi-task FM

## What this is

The validation-best result across three hand-implemented experiments, each
testing one idea in isolation against the official FM baseline
(`solution/train.py`, valid primary 0.60161):

| Model | valid primary | vs. baseline |
|---|---|---|
| **Multi-task FM (this one)** | **0.60420** | **+0.00258** |
| Sequential attention (self-attention over watch history) | 0.60137 | -0.00024 |
| Combined (sequential attention + multi-task) | 0.60252 | +0.00091 |

Multi-task wins outright -- combining it with sequential attention diluted
the result rather than improving it, so multi-task alone is the answer, not
an ensemble.

## What the model does

Same 5 fields and same FM interaction structure as the baseline
(`user_id`, `video_id`, `author_id`, `tab`, `dur_bucket`), so the improvement
is attributable to the added supervision, not a different architecture.
Trained with two auxiliary heads reading the same shared embeddings as the
main `long_view` head:

- `is_click` (46.3% positive rate in train)
- `is_like` (1.9% positive rate in train)

`is_follow` / `is_comment` / `is_forward` were deliberately excluded: their
train-split positive rates are 0.10% / 0.26% / 0.10%, sparse enough that a
naive BCE auxiliary head on them was judged more likely to inject noisy
gradients than useful signal. See `train.py`'s module docstring.

## Validation across seeds

| seed | valid primary |
|---|---|
| 0 | 0.604194 |
| 1 | 0.604197 (checkpoint.npz in this directory) |

Extremely tight across seeds (delta ~0.000003) -- this is checkpoint.npz for
the higher-scoring seed (1), following the same "best individual seed, not
an average" convention `agent/registry.py` uses for orchestrator-driven runs.

## Verification

Every number here was independently re-derived from `val_predictions.npz`
through the vendored, unmodified `harness/evaluate.py` (the sole scoring
authority) and matched the claimed `result.json` values to 4+ decimal places
-- not just trusted from the training run's own self-report. `result.json`
contains validation-split metrics only; no test-split number appears in any
file in this directory (test metrics were quarantined separately during
each run, per the same discipline the rest of this harness follows).

## Files

- `train.py` -- the exact model definition and training code that produced this checkpoint (copied from `experiments/multitask/train.py`)
- `config.json` -- the exact hyperparameters used
- `checkpoint.npz` -- trained weights (seed 1, best epoch)
- `result.json` -- claimed validation metrics
- `val_predictions.npz` -- the raw (user_id, label, score) arrays the metrics were computed from, for independent re-verification

## Reproduce

```
KUAIRAND_PATH=/path/to/KuaiRand-Pure/data python train.py --config config.json --seed 1 --out /tmp/result.json
```

## Provenance note

This result did NOT come from the autonomous Research/Coding/Evaluator loop
(`scripts/run_loop.py`) -- it was hand-implemented directly, after the
autonomous loop's own attempts (BPR, DeepFM, hybrid loss, weighted sampling,
short-history features) all landed under the competition's 0.002 convergence
threshold. `logs/registry.json` reflects only autonomous-loop runs and is
regenerated fresh each time `run_loop.py` runs; this directory is the durable
record of the actual best result.
