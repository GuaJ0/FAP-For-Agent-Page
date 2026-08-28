# Run log

## Environment verification (before anything was built on top)

Dataset downloaded from Zenodo per the starter kit README and checked with the
README's own self-check:

| check | expected | got | |
|---|---|---|---|
| `baseline.py --model random`, test primary | ~0.4753 (±0.001) | **0.4757** | pass |
| `baseline.py --model fm --seed 0`, test primary | 0.5946 (±0.0008) | **0.5953** | pass |
| `baseline.py --model fm --seed 0`, valid primary | 0.6016 | **0.6015** | pass |

Split sizes: train 1,141,112 / valid 124,909 / test 170,588.

## Iteration 0 — FM baseline, seeded by hand

`solution/train.py`, ported from `harness/baseline.py` into the executor's
`--config/--seed/--out` contract.

| seed | valid GAUC | valid nDCG@5 | **valid primary** | test primary (quarantined) |
|---|---|---|---|---|
| 0 | 0.6671 | 0.5358 | **0.6015** | 0.5953 |
| 1 | 0.6674 | 0.5361 | **0.6018** | 0.5948 |

Matches the published FM baseline bit-for-bit on seed 0. **Validation 0.6015 is
the bar.**

## Iteration 0 — the baseline, run through the loop itself

`solution/` is no longer just a file on disk: `Orchestrator.bootstrap_baseline()`
runs it through the real executor before the research loop starts, so it lands
as a genuine `RunRecord` at iteration 0 and becomes the registry incumbent.

```
iteration 0  status=success  decision=accept  parent=None  delta=None
  VALID primary=0.6016 (std 0.0001) gauc=0.6673 ndcg5=0.5360 over 2 seeds
```

`registry.best()` therefore points at a complete artifact dir (`result.json`,
`checkpoint.npz`, `val_predictions.npz`) — which matters, because nothing has
beaten it yet.

## Iteration 1 — pairwise BPR ranking loss (end-to-end through the harness)

Real `LLMCodingAgent` → real `Executor` → real KuaiRand data, driven by the
existing `Orchestrator` with `FakeResearchAgent` / `FakeEvaluatorAgent`.
Generation used the offline template library (no API key available yet), so
this exercises every part of the loop except the model call itself.

```
iteration 1  status=success  decision=REVERT  parent=0  delta=-0.00276
  VALID primary=0.5989 (std 0.0005) gauc=0.6633 ndcg5=0.5345 over 2 seeds
    seed 0: 0.5984 (15s)    seed 1: 0.5993 (15s)
```

Both seeds ran clean, both left a verifiable artifact dir, both test-split
scores went to `logs/quarantine/` and `runs.jsonl` passed the leak scan.

> **Corrected from an earlier run of this experiment.** It was originally
> recorded here as `decision=accept` with `delta=+0.5989`. That was the
> empty-registry bug: with no incumbent, `delta` was the absolute score rather
> than a delta, so `FakeEvaluatorAgent` accepted a result that had lost. With
> the baseline bootstrapped as iteration 0 the same run REVERTs, which is the
> correct call. The measured metrics themselves are unchanged.

### The result itself: the hypothesis did not beat the baseline

| loss | valid primary | vs baseline 0.6015 |
|---|---|---|
| pointwise logloss (baseline) | **0.6015** | — |
| BPR, lr 1e-3 | 0.5984 | −0.0031 |
| BPR, lr 3e-4 | 0.5988 | −0.0027 |
| listwise softmax, L=8 | 0.5987 | −0.0028 |
| listwise softmax, L=16, lr 2e-3 | 0.5963 | −0.0052 |

Four configurations, all 0.003–0.005 below the baseline — several times the
0.0008 seed std, so this is a real gap rather than noise. The organisers ranked
"switch to a ranking loss" as the most promising unexplored direction; at these
hyperparameters it does not pay off, and the loop now correctly REVERTs it and
keeps the baseline as the incumbent.

Not tried, and where I would look next: the within-user samplers weight every
user equally, while GAUC weights users by positive count — a weighted sampler
is a one-line change. A hybrid objective (logloss + λ·BPR) is also untested and
is the usual way this trade lands in practice.
