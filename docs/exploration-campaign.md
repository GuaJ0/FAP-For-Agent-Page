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

## Re-running: the scratch root resumes, it does not restart

`runs/exploration/` holds `runs.jsonl`, `orchestrator_state.json` and the
registry, and the orchestrator is deliberately crash-resume safe — so pointing a
second run at a root that already finished is a **no-op**, not a fresh run.
`should_stop` fires before the first `_step()` and `run()` returns immediately.

This is easy to mistake for success. The summary still prints the previous
run's records, the previous run's LLM totals and a `finished in 0s` line that is
the only real tell:

```
=== finished in 0s, 3 record(s) ===
...
coding totals:    {'calls': 4, ...}      <- unchanged from the previous run
```

To genuinely re-run, move the root aside first:

```bash
mv runs/exploration "runs/exploration-archived-$(date +%Y%m%d-%H%M%S)"
python scripts/seed_findings.py --model gpt-5
```

Archive rather than delete: a scratch run's `solutions/` are the evidence for
whatever its findings claim, and the whole point of the campaign is that those
claims are checkable.

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

# REVIEW ITEMS — all resolved

All five items below were reviewed and decided before the campaign was cleared
to run. The reasoning is kept rather than deleted, so the audit trail shows a
decision that was made rather than a question that was dropped.

## 1. `LOG-RANDOM-DIAGNOSTIC` would produce a false "Don't" — RESOLVED

**The problem.** That entry changes no model: it scores the unchanged
incumbent on the randomized-exposure log and reports the number. Its delta on
validation primary is therefore **zero by construction**, and the Evaluator
reads that as REVERT — landing a `dont` in the ledger for the
`UNBIASED-VALIDATION` family. The verdict would be about the entry having
tested nothing, not about the direction failing.

**Decision: option (a) — run it, but never promote its verdict.** Made
automatic rather than manual. `_promote()` in `scripts/seed_findings.py` now
skips any finding whose family resolves to one where *every*
`DEFAULT_BACKLOG` entry declares `hyperparameters["report_only"] == [True]`,
and prints why:

```
  SKIPPED  UNBIASED-VALIDATION    (diagnostic-only entry: it changes no model,
                                   so its verdict measures nothing and is not a real Don't)
```

Three properties worth noting:

- **Detected from the backlog, not hardcoded.** A future report-only entry is
  excluded the same way with no list to remember to update.
- **`all`, not `any`.** A family mixing a diagnostic with real modelling
  variants has genuinely measured something, so its verdict still promotes.
- **Excluded, not destroyed.** The finding stays in the scratch ledger, so what
  the diagnostic actually reported remains auditable.

Relying on a human deleting a line before `--promote` was the original plan and
was rejected as too fragile — forgetting once puts a false "Don't" into the
graded run's memory permanently. Covered by four tests in
`tests/test_seed_findings_campaign.py`.

## 2. Data boundary for the randomized-exposure log — RESOLVED, with a scope limit

**Decision: in-bounds for read-only diagnostic use.** The use is strictly
read-only, scores an already-trained model, never becomes training data or a
selection metric, and is the dataset's own documented purpose for the file
(`solution/ideas.md`, unexplored direction 7).

`agent/research/retrieval/models.py` validates `allowed_data_boundary` against
exactly three literal values, so no fourth category was added to
`agent/research/profiles/kuairand_pure.json`. The decision is recorded in that
profile's `assumptions` list instead, where it travels with the profile.

**A scope limit found while implementing this, and applied.** The file spans
`20220422-20220508`, but `harness/data.py` puts `20220429-20220508` in the
held-out split. So three quarters of it is out of bounds:

| portion | dates | rows | share |
|---|---|---|---|
| validation window | 0422–0428 | 288,338 | 24.5% |
| held-out window | 0429–0508 | 897,721 | **75.5%** |

This matters because the diagnostic's output becomes Evaluator commentary,
which flows into `RunRecord` events → the ledger's `why` field → the Research
prompt. Scoring on held-out-period rows would put that period's information on
a live path into agent-visible context.

The entry is therefore restricted to `20220422-20220428` — the official
validation window, 288,338 randomized-exposure rows, still ample for the
check. The filter is pinned in `hyperparameters["date_range"]`, stated in the
implementation steps, and listed in `must_hold_constant`, so it is not left to
whatever a Coding Agent infers from the filename. Guarded by
`test_the_randomized_exposure_diagnostic_never_reads_a_held_out_row`.

## 3. The two new citations — RESOLVED, both verified against primary sources

Both were checked against the papers themselves, not from memory.

**`covington2016youtube` / `example-age-freshness` — accurate.** From §3.3 of
the paper, verbatim:

> "Machine learning systems often exhibit an implicit bias towards the past
> because they are trained to predict future behavior from historical
> examples. The distribution of video popularity is highly non-stationary but
> the multinomial distribution over the corpus produced by our recommender
> will reflect the average watch likelihood in the training window of several
> weeks. To correct for this, we feed the age of the training example as a
> feature during training."

Each clause of the claim text maps onto that passage: the bias toward the past,
representing time-dependent popularity, and the contrast with averaging over
the training window. The paper's own Figure 4 caption confirms the model
"is able to accurately represent the upload time and time-dependant
popularity".

One honest gap, already stated in the entry rather than hidden: Covington uses
example age as a **feature**, not as sample weighting. `TIME-DRIFT` does both,
and its `evidence_application` says so explicitly ("motivates … here it is
applied as sample weighting plus an explicit time-of-day interaction"), so the
citation is not being stretched to cover the extension.

**`schnabel2016propensity` / `biased-logging-biases-evaluation` — accurate.**
All three parts of the claim check out. The abstract establishes that data is
"subject to selection biases" and that their approach yields "unbiased
performance estimators despite biased data" via "estimation techniques from
causal inference". The randomized-exposure part is confirmed in the evaluation
setup: the Yahoo! R3 test set is "ratings by a subset of 5400 users who were
asked to rate 10 randomly chosen songs", and for the coat dataset "the
self-selected ratings are the training set and the uniformly selected ratings
are the test set" — i.e. randomly-exposed data used as unbiased ground truth,
which is exactly the role `LOG-RANDOM-DIAGNOSTIC` puts it in.

**No changes made to `agent/research/references.json`.**

## 4. Hyperparameter values — RESOLVED; TIME-DRIFT updated against the real data

The dataset was located at `$KUAIRAND_PATH` and the actual date coverage
computed from the `date` field of both standard logs.

**Measured splits** (`harness/data.py` declares the boundaries; row counts are
from the data):

| split | declared | actual data | days | rows |
|---|---|---|---|---|
| train | 20220408–20220421 | 20220409–20220421 | **13** | 1,141,112 |
| valid | 20220422–20220428 | 20220422–20220428 | 7 | 124,909 |

The split declares 04-08 but the data holds no rows for that date, so the real
training window is 13 days. (1,141,112 matches the "1.14M rows" in `ideas.md`,
confirming the split reconstruction.)

**The old grid `[7, 21, None]` bracketed that badly** — it was chosen before
the window was known. Weight retained across the whole training set:

| half-life | oldest row's weight | total weight retained | |
|---|---|---|---|
| 3 days | 0.062 | 18.6% | genuinely aggressive |
| 7 days | 0.305 | 45.2% | *old low value* — over half the window |
| 14 days | 0.552 | 66.5% | ≈ one window length |
| 21 days | 0.673 | 76.0% | *old high value* — nearly the control |
| `None` | 1.000 | 100.0% | control: uniform |

At a 21-day half-life the oldest row still keeps 67% of its weight, so that
cell was measuring almost the same thing as the `None` control and wasting a
run.

**Updated to `[3, 14, None]`**: 3 days is roughly a quarter of the window and
retains ~19% of sample weight (a real aggressive end), 14 days is one window
length at ~67%, and `None` remains the control.

**A formula/name mismatch fixed at the same time.** The entry's step said
`exp(-age_days / half_life)`, which makes the parameter a *time constant* —
weight halves at `0.693 × h`, not at `h`. The field is named
`recency_half_life_days`, so the step now specifies
`0.5 ** (age_days / recency_half_life_days)`, a true half-life. Without this
the chosen numbers would not have meant what this table says they mean.

**One property of the data worth knowing when reading the result**, now in the
entry's `risks`: the training window is heavily front-loaded — 63.6% of its
rows fall in its first four days, which are also its *oldest*. Recency
weighting therefore down-weights the majority of the data, which is why the
aggressive cell retains only ~19% of total weight and may lose on variance
alone rather than on drift.

Everything else in the table below is unchanged, including the
`WATCHTIME-CENSORED` censoring threshold of 0.95, which is kept as proposed.

| entry | setting | rationale |
|---|---|---|
| `DIN-SHORT-HISTORY` | `history_length [20,50]`, `dim 16` | existing entry, unchanged |
| `DIN-LONG-HISTORY` | `history_length [100,200]`, `dim 32` | users have hundreds–thousands of interactions; short window may simply be too small |
| `DIN-MEAN-POOL` | `history_length [50]`, mean pooling | the control: separates "history helps" from "candidate conditioning helps" |
| `MULTITASK-ENGAGEMENT` | `[is_click, is_like]`, `λ 0.05–0.1` | existing entry, unchanged |
| `MULTITASK-ALL-ENGAGEMENT` | all 5 labels, `λ 0.02` | breadth at low weight; the rare labels are very sparse |
| `MULTITASK-CLICK-HEAVY` | `[is_click]`, `λ 0.3–0.5` | densest, most related label at high weight — the strongest-effect regime |
| `WATCHTIME-AUXILIARY` | `log1p_capped`, `λ 0.05–0.1` | existing entry, unchanged |
| `WATCHTIME-CENSORED` | `censoring_threshold_ratio 0.95`, `λ 0.05–0.1` | ideas.md item 4 names censored regression specifically; kept as proposed |
| `WATCHTIME-RATIO` | `completion_ratio_clipped`, `λ 0.05–0.2` | removes the duration confound; no censoring machinery needed |
| `DEEPFM` | `[64,32]`, `dropout 0.1` | existing entry, unchanged; 1 variant only — ideas.md measures capacity as not the bottleneck |
| `TIME-DRIFT` | **`half_life [3, 14, None] days`**, `8 hour buckets`, `hour × dur_bucket` | **updated**: brackets the measured 13-day window; `None` isolates weighting from the feature |
| `LOG-RANDOM-DIAGNOSTIC` | `report_only`, **`date_range [[20220422, 20220428]]`** | **updated**: validation window only — see item 2 |

One structural constraint the entries already respect, from `ideas.md`: a pure
user-side first-order term contributes exactly zero, because ranking happens
within a user. That is why `TIME-DRIFT` crosses the time bucket against an
item-side field rather than adding a bare time feature.

## 5. Backlog ordering — RESOLVED, no reordering

**Decision: keep ranks 1–14 exactly as they are.** Two reasons:

- **Ranks 1–6 stay byte-identical to today's graded-run behaviour.** The eight
  new entries were appended at 7–14, so the graded run's first choices are
  unchanged and this work cannot have altered them.
- **There is no budget pressure to reorder for.** Summed `expected_wall_s`
  across all 14 entries is ~6.5h of training wall-clock, against the campaign's
  `EXPLORATION_MAX_WALL_S` ceiling of 24h — comfortable margin, so there is no
  risk of the later ranks being cut off before they run.

Reordering was only worth considering if the campaign might not reach ranks 13
and 14 (the two genuinely-untried directions). It will.
