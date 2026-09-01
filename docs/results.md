# Results and resource summary (Deliverable 4)

Everything below is re-derived from committed artifacts, not asserted: the
results table is independently re-checked against the vendored
`harness/evaluate.py` (see `submission/README.md`'s Verification section),
and the resource totals are summed from the real usage/run logs in `logs/`
and `logs/archive/`, not estimated.

## Environment verification

Dataset downloaded from Zenodo per the starter kit README and checked
against the README's own self-check, before anything was built on top of it:

| check | expected | got | |
|---|---|---|---|
| `baseline.py --model random`, test primary | ~0.4753 (±0.001) | **0.4757** | pass |
| `baseline.py --model fm --seed 0`, test primary | 0.5946 (±0.0008) | **0.5953** | pass |
| `baseline.py --model fm --seed 0`, valid primary | 0.6016 | **0.6015** | pass |

Split sizes: train 1,141,112 / valid 124,909 / test 170,588. This is the
source of the "official baseline" numbers cited throughout this document.

## Results table

**Valid and test are never mixed below.** The official baseline's test score
is a published reference number the starter kit provides; no test-split
number exists anywhere for the submitted model -- see "Why no test number for
the submission" below.

| | valid primary | valid GAUC | valid nDCG@5 | test primary |
|---|---|---|---|---|
| Official FM baseline (`solution/train.py`) | 0.60161 | 0.6673 | 0.5360 | 0.5946 (published) |
| **Submitted model (multi-task FM)** | **0.60420** | **0.6713** | **0.5371** | not measured (by design) |
| **Absolute delta vs. baseline** | **+0.00258** | **+0.00400** | **+0.00116** | — |

Baseline valid GAUC/nDCG@5 are `solution/train.py`'s own iteration-0
bootstrap run (`logs/runs.jsonl` iteration 0, replicated identically as the
starting incumbent in all three loop runs: 0.6016151905059814 / 0.6672640740871429
/ 0.5359663367271423, mean of 2 seeds). Submitted-model numbers are
`submission/checkpoint/result.json` (seed 1).

### Why no test number for the submission

Every training script (agent-generated and hand-written alike) computes a
test-split score and can print it, but the harness's own discipline is that
test metrics are never surfaced to anything that makes a decision --
`agent/executor.py` quarantines them straight to `logs/quarantine/`, never
into a `RunRecord`, and the hand-implemented `experiments/multitask/train.py`
that produced the submission followed the identical discipline (see
`submission/README.md`'s Verification section: *"no test-split number appears
in any file in this directory"*). That is a deliberate integrity choice, not
a gap -- reporting a test number here would mean it was looked at.

## Submitted checkpoint provenance

**Not produced by the autonomous loop.** Evidence:

- `submission/checkpoint/{train.py,config.json}` are byte-identical to
  `experiments/multitask/{train.py,config.json}`, committed directly to git
  (`52f7b19`, `f91d181`, both 2026-08-31) -- **before Run 1 even started**
  (2026-09-01). Code the CodingAgent writes lands in `solutions/attempt_NNN/`
  instead; nothing in `experiments/` or `submission/` ever went through that
  path.
- No `RunRecord` in any of the three runs' `runs.jsonl` (current or archived)
  contains this model's numbers, its `is_click`/`is_like` auxiliary-head
  design, or a matching hypothesis. Checked by exact-value and substring
  search across all three.
- No LLM usage record (`coding_agent_usage.jsonl` / `evaluator_usage.jsonl`,
  any run) references it; the code itself makes no API calls. Cost was real
  CPU-time only, hand-run, not captured in any `logs/` file because it ran
  outside the orchestrator entirely.
- Checkpoint: seed 1 (`checkpoint.npz`, 16 epochs), the higher-scoring of the
  two validated seeds (0.604197 vs 0.604194 -- delta ~0.000003), per the same
  "best individual seed, not the average" convention `agent/registry.py` uses
  for orchestrator-driven runs.

One real nuance: the offline research backlog's own `MULTITASK` family
(`agent/research/findings.jsonl`) includes a variant that used the *same*
`is_click`+`is_like` label pair, at a lower auxiliary weight (`lambda_aux`
0.05-0.1) and a different implementation -- and it **regressed**
(delta -0.0073). So the loop tried a version of this direction and it failed;
the submission is a separately-tuned, better-implemented, entirely human
build of a related idea, not a repeat of the loop's own attempt.

## Resource summary

### LLM tokens and cost (all 3 runs, summed)

| | tokens in | tokens out | cost (USD) |
|---|---|---|---|
| Coding agent | 119,885 | 152,669 | $1.6766 |
| Evaluator agent | 12,549 | 5,314 | $0.0688 |
| **Total** | **132,434** | **158,983** | **$1.7454** |

Research agent: $0 across all three runs -- all three used the default
deterministic `OfflineResearchAgent` (template backlog, no API calls;
`summary.json`'s `"research": null` in every run confirms this).

Per-run breakdown:

| run | coding calls | coding in/out | evaluator calls | evaluator in/out | cost |
|---|---|---|---|---|---|
| Run 1 (archived, 5 records) | 8 | 51,896 / 65,010 | 3 | 6,026 / 2,597 | $0.7485 |
| Run 2 (archived, 4 records) | 7 | 46,126 / 59,651 | 3 | 5,045 / 2,121 | $0.6817 |
| Run 3 (complete, 2 records) | 3 | 21,863 / 28,008 | 1 | 1,478 / 596 | $0.3152 |

The hand-implemented `experiments/`/`submission/` work spent $0 in LLM cost
(no API calls; see provenance section above) -- its cost is the CPU time
below, not counted separately since it isn't logged anywhere as a run.

### Compute

**0 GPU-hours, 0.349 CPU-hours** (across all three loop runs; measured, not
estimated -- see below).

This is CPU-only **by design**, a feasibility result rather than a missing
number: the entire model family (FM, multi-task FM, the DIN-style and
self-attention variants tried) is a small embedding-table model trained with
plain NumPy, deliberately kept reproducible on a consumer laptop with no GPU
dependency, no CUDA setup, and no accelerator cost. The published baseline
itself trains this way; every experiment in this project (loop-generated and
hand-built alike) preserved that property.

0.349 CPU-hours is `sum(RunRecord.resources.cpu_hours)` across all three
runs' `runs.jsonl` (`runlog/report.py::total_compute`, which itself sums each
subprocess's real `getrusage()` delta from `agent/executor.py` -- not derived
from wall-clock, and not an estimate): Run 1 0.1607, Run 2 0.1452, Run 3
0.0431. The hand-implemented `experiments/multitask/` training itself adds
roughly another 0.03 CPU-hours (2 seeds × ~52s wall, single-threaded NumPy) on
top of this, per its own `result.json.wall_s` -- not included in the 0.349
total since, per the resource-usage docstring in `agent/orchestrator.py`, that
number is specifically the loop's own accounting and this run never touched
the loop.

## What was tried and didn't work (or didn't make the cut)

Every entry below is from `agent/research/findings.jsonl` (the project's
persistent, deterministically-written Do/Don't ledger -- see `agent/research/findings.py`)
or `submission/README.md`'s independently re-verified comparison table, not
a summary written after the fact. Full per-iteration detail (hypothesis,
metrics, evaluator commentary) is in `docs/run_report.md`.

### The three the brief asked about

**BPR pairwise loss -- mixed, not simply negative.** The family split into
two materially different variants:

| variant | measured deltas (3 independent runs) | verdict |
|---|---|---|
| Plain hybrid BPR (`OFFLINE-HYBRID-BPR`) | −0.00083 (original campaign), −0.00026 (Run 2) | reverts |
| Positive-count-weighted BPR (`OFFLINE-GAUC-WEIGHTED-BPR`) | **+0.0016** (original), **+0.0010** (Run 1), **+0.0006** (Run 2) | **accepted, 3/3** |

The GAUC-weighted variant is the single most consistently-replicated positive
result in this project outside the submission itself -- accepted three
independent times, delta shrinking each time but never negative. It is not in
`submission/`; the human-authored multi-task path scored higher
(+0.00258 vs. an estimated +0.0006-0.0016) and was chosen instead, not because
this direction failed.

One caution, reported honestly rather than hidden: Run 1's own attempt to
re-implement plain hybrid BPR crashed to primary 0.4161, driven by GAUC
0.4023 -- below the 0.5 random-ranking baseline, i.e. an inverted ranking,
not merely a weak one. That is a generated-code bug, not evidence about the
technique. `findings.py`'s `_is_sub_random` guard (GAUC < 0.5) correctly
excluded it from the ledger for exactly this reason, with a docstring citing
a prior real incident of the same shape.

**Candidate-conditioned history (DIN-style short-history features) --
marginal, inconsistent.** Accepted in the original exploration campaign
across 3 variants (window 20-200, delta averaging +0.00044, "marginal"
effect) but reverted in both Run 1 (delta −0.0014) and Run 2 (delta −0.0010)
at the specific window (20) tried each time. Evaluator commentary across
both graded runs converged on the same diagnosis: the signal is too noisy at
that window length, and a strict no-leak audit plus a shorter/decayed window
would be the next thing to try, not a bigger one.

**Sequential attention (self-attention over watch history) -- slightly
negative.** Hand-built (SASRec-style, target-attention pooled against the
candidate item), independently re-verified: **0.60137 vs. 0.60161 baseline,
−0.00024**. Combined with multi-task it reached 0.60252 (+0.00091) -- better
than alone, but still below multi-task by itself (0.60420), so it was not
carried into the submission either standalone or combined.

### Other directions explored (for completeness -- negative results are findings)

| direction | best delta | verdict | note |
|---|---|---|---|
| Watch-time auxiliary target | +0.0001 | accept (within noise) | 3 variants, gain too small to act on |
| DeepFM-style nonlinear tower | −0.0033 | revert | likely overfitting/interference from naive logit addition |
| Broad multi-task (all 5 engagement labels, low weight) | +0.00003 | revert (within noise) | one attempt regressed −0.0073; see submission provenance note above for the related, successful, differently-weighted variant |
| Recency-weighted training + time×duration cross | −0.0007 | revert | recency weighting over-discounted front-loaded data |

## See also

- `docs/run_report.md` -- Deliverable 3, full per-iteration detail for all
  three runs.
- `submission/README.md` -- the submission's own verification note and
  comparison table.
- `logs/interventions.md` -- the manual-intervention log referenced above.
