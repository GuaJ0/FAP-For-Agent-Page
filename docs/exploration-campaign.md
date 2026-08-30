# Pre-final-run exploration campaign

A one-time pass that measures the seven "Unexplored directions" from
`solution/ideas.md` for real, so the graded run starts with an honest
`agent/research/findings.jsonl` instead of an empty one.

Nothing here fabricates an outcome. Each direction goes through the real
Coding → Executor → Evaluator loop, and the entries are built deterministically
from the resulting `RunRecord`s by `agent/research/findings.py`.

## Running it

```bash
# plan only -- prints the 14 rounds and the exact run_loop invocation, spends nothing
python scripts/seed_findings.py --dry-run

# the campaign (needs OPENAI_API_KEY in .env; costs real money)
python scripts/seed_findings.py --model gpt-5

# after reviewing runs/exploration/findings.jsonl
python scripts/seed_findings.py --promote
```

This uses the **real** Coding and Evaluator agents. It is deliberately not
`--offline`: that mode swaps in the hand-written template library, which can
only express ranking-loss variants, so it could not implement DIN sequences,
multi-task heads, DeepFM, watch-time regression or time features. An
`--offline` campaign would produce real-looking records for ideas that were
never actually built. Research stays deterministic (`OfflineResearchAgent`
walks `DEFAULT_BACKLOG` in rank order) because the point is covering a known
list rather than letting a model choose.

## Isolation from the graded run

| | campaign | graded run |
|---|---|---|
| `logs/`, `solutions/`, registry, state, artifacts | `runs/exploration/` | repo-root `logs/` |
| Do/Don't ledger | `runs/exploration/findings.jsonl` | `agent/research/findings.jsonl` |
| repair attempts | 5 | 3 (`RetryConfig` default) |
| per-idea backstop | 90 min | 45 min |
| per-run timeout | 60 min | 15 min |

The campaign's iterations, wall-clock and convergence window are per-run state
inside that scratch `logs/`, so it cannot consume the graded run's budget.
`--promote` is the separate, explicit step that merges reviewed findings into
the committed ledger; until you run it the graded run's memory is untouched.

## Confidence, and why a single attempt is not a "Don't"

Ledger entries roll up by **family**, not by individual proposal id, so the
three DIN variants accumulate into one `DIN-SEQUENCE` entry recording
`attempts`, `variants`, `deltas`, `coverage` and a `confidence` tier:

- `well_tested` — 3+ real attempts across the stated range all failed; closed.
- `tested` — 2 attempts.
- `inconclusive` — a **single** attempt lost. That rules out one
  implementation, not the mechanism. The Research prompt says so explicitly:
  such a direction may be reproposed if the proposal states what it is doing
  differently from the variant named in `variants`.

This is why the structurally complex directions carry 2–3 entries each. One
Coding Agent generation at one setting cannot distinguish "the mechanism is
wrong" from "the one implementation we happened to get was weak", and a false
"Don't" is worse than no entry — it steers Research away from something nobody
actually tested.

## Testing one implementation at several settings

Different from the above: `scripts/config_sweep.py` reruns a **fixed**
already-generated implementation with only the config swapped, so a difference
between points is attributable to the setting rather than to the Coding Agent
having written different code.

```bash
python scripts/config_sweep.py \
    --solution-dir runs/exploration/solutions/attempt_004 \
    --base-config  runs/exploration/solutions/attempt_004/config.yaml \
    --sweep lambda_bpr=0.05,0.1,0.2 \
    --incumbent-primary 0.6015
```

It never registers a checkpoint, so it cannot move the incumbent, and it writes
generated configs outside the solution dir so the swept implementation stays
byte-identical. See `agent/sweep.py` for why it is a separate mode rather than
a flag on `Orchestrator._step()`.

---

# OPEN ITEMS — these need your review before the campaign runs

## 1. `LOG-RANDOM-DIAGNOSTIC` will produce a misleading "Don't"

Per the diagnostic-only framing, this entry changes no model: it scores the
unchanged incumbent on the randomized-exposure log and reports the number.
Its delta on validation primary is therefore **zero by construction**, and the
Evaluator will read that as REVERT — which lands in the ledger as a `dont` for
the `UNBIASED-VALIDATION` family.

That verdict would be about the absence of a modelling change, not about the
direction being a dead end. Three ways to handle it, your call:

- **(a)** Run it, then delete that one entry from the scratch ledger before
  `--promote`. Simplest, and the scratch/promote split exists for exactly this.
- **(b)** Run it outside the campaign as a plain diagnostic script, so it never
  produces a ledger entry at all.
- **(c)** Pair it with a modelling change so it has something to measure —
  which would contradict the diagnostic-only decision.

Recommendation: **(a)**.

## 2. Data boundary for the randomized-exposure log

`agent/research/profiles/kuairand_pure.json` declares
`allowed_data_boundary: ["public_metadata", "train", "validation"]`. It does
**not** list `log_random_4_22_to_5_08_pure.csv`. Read-only use for diagnosis is
the narrowest possible framing, but it is still a file the declared boundary
does not name. Confirm this is in bounds before running round 14.

## 3. Two citations were added to the catalog

`agent/research/references.json` had no claim supporting either new direction,
and a proposal cannot validate without one:

- `covington2016youtube` gained a second claim, `example-age-freshness`, for the
  temporal-drift direction. The example-age contribution is genuinely in that
  paper.
- `schnabel2016propensity` (Schnabel et al., ICML 2016, *Recommendations as
  Treatments*) was added for the unbiased-validation direction.

Both are real papers and the claim texts are accurate to my knowledge, but I
paraphrased them rather than quoting the sources — worth a check against the
originals since the catalog is the pipeline's evidence base.

## 4. Hyperparameter values to confirm

Proposed, not settled. Each is the variable being tested, so these are the
numbers that decide what "we tested this direction" actually means.

| entry | proposed setting | rationale |
|---|---|---|
| `DIN-SHORT-HISTORY` | `history_length [20,50]`, `dim 16` | existing entry, unchanged |
| `DIN-LONG-HISTORY` | `history_length [100,200]`, `dim 32` | users have hundreds–thousands of interactions; short window may simply be too small |
| `DIN-MEAN-POOL` | `history_length [50]`, mean pooling | the control: separates "history helps" from "candidate conditioning helps" |
| `MULTITASK-ENGAGEMENT` | `[is_click, is_like]`, `λ 0.05–0.1` | existing entry, unchanged |
| `MULTITASK-ALL-ENGAGEMENT` | all 5 labels, `λ 0.02` | breadth at low weight; the rare labels are very sparse |
| `MULTITASK-CLICK-HEAVY` | `[is_click]`, `λ 0.3–0.5` | densest, most related label at high weight — the strongest-effect regime |
| `WATCHTIME-AUXILIARY` | `log1p_capped`, `λ 0.05–0.1` | existing entry, unchanged |
| `WATCHTIME-CENSORED` | `censoring_threshold_ratio 0.95`, `λ 0.05–0.1` | ideas.md item 4 names censored regression specifically |
| `WATCHTIME-RATIO` | `completion_ratio_clipped`, `λ 0.05–0.2` | removes the duration confound; no censoring machinery needed |
| `DEEPFM` | `[64,32]`, `dropout 0.1` | existing entry, unchanged; 1 variant only — ideas.md measures capacity as not the bottleneck |
| `TIME-DRIFT` | `half_life [7, 21, None] days`, `8 hour buckets`, `hour × dur_bucket` | `None` is the control isolating weighting from the feature |
| `LOG-RANDOM-DIAGNOSTIC` | `report_only` | see open item 1 |

Two judgment calls worth singling out:

- **Recency half-lives (7 / 21 days).** Chosen to bracket the training window
  without knowing its exact span. If the window is much shorter or longer than
  I assumed, these are wrong and should move.
- **Censoring threshold (0.95 of duration).** "Watched to completion" needs a
  threshold and the data does not supply one. 0.95 tolerates timing noise; a
  stricter or looser value changes which rows are treated as censored.

One structural constraint the entries already respect, from `ideas.md`: a pure
user-side first-order term contributes exactly zero, because ranking happens
within a user. That is why `TIME-DRIFT` crosses the time bucket against an
item-side field rather than adding a bare time feature.

## 5. Backlog ordering

The eight new entries were appended at ranks 7–14; ranks 1–6 are untouched, so
the graded run's first choices are byte-identical to today. The campaign runs
all 14 regardless of rank. If you would rather the campaign ran the two
genuinely-untried directions (`TIME-DRIFT`, `LOG-RANDOM-DIAGNOSTIC`) earlier,
that is a rank change — say so and I will reorder.
