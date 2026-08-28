# The Coding agent (`agent/coding/`)

Real implementation of the `CodingAgent` Protocol in `agent/agents.py`:

```python
implement(self, idea: Idea, feedback: Optional[str]) -> Diff
```

It turns one hypothesis into a solution dir the existing `executor.py` runs
as-is — a generated `train.py`, a config, and copies of the vendored
`evaluate.py` / `data.py` so the dir is self-contained under the executor's
`cwd=<solution_dir>` subprocess.

| file | role |
|---|---|
| `agent/coding/agent.py` | `LLMCodingAgent`: the generate → check → smoke → repair loop |
| `agent/coding/llm.py` | `LLMClient` protocol, `OpenAIClient`, test/offline clients, token & cost accounting |
| `agent/coding/prompts.py` | System prompt (the contract) + generate/repair prompt builders |
| `agent/coding/templates/train_ranking.py` | Hand-written BPR / listwise implementation used by the offline client |

## The inner loop, and why it isn't a duplicate of the orchestrator's

The orchestrator already retries: up to `max_fix_attempts` (3) per idea, with
the previous failure passed back as `feedback`. This agent does **not**
re-implement that cap. It runs a cheaper loop underneath it:

```
generate → static checks → smoke run (~20k rows, 1 epoch) → repair → ship
```

An orchestrator-level attempt costs a full multi-seed training run and one of
only three attempts before the idea is abandoned. Spending one on a `NameError`
is waste. The static checks (unavailable imports, missing CLI flags, a
reimplemented `evaluate()`, a test-split key in `result.json`) cost nothing,
and the smoke run catches contract violations in seconds. What reaches the
executor has already been shown to start, train, score, and produce a
**verifiable** `result.json`.

`feedback` is used rather than ignored: when it is not `None`, the last shipped
source *and* the failure both go into a repair prompt, so the model fixes what
it wrote instead of starting over and reintroducing solved bugs.

`implement()` always returns a `Diff`, even when every repair failed — the
orchestrator has no `try/except` around the call, so raising would kill the
whole run instead of being recorded as one honest failed iteration.

## Clients

| client | real model? | use |
|---|---|---|
| `OpenAIClient` | yes | production. Model from `CODING_AGENT_MODEL` (default `gpt-5`) |
| `ScriptedClient` | no | unit tests. Deterministic, free |
| `TemplateLibraryClient` | **no** | serves hand-written implementations keyed off the hypothesis, so the whole pipeline can be exercised end-to-end with no API key and no spend |

`TemplateLibraryClient` is not a language model and never claims to be: every
response it returns is flagged `is_real_model_call=False` in the usage log, and
it has no repair capability. When it can't match a hypothesis it says
`NO_TEMPLATE` rather than guessing, and the agent records that as a failed
attempt.

---

# Known gaps — for whoever picks this up

## 1. `RunRecord.resources` is still not populated  ← the one I was asked to flag

`ResourceUsage(wall_s, gpu_s, tokens_in, tokens_out)` exists on every
`RunRecord`, and `orchestrator.py` constructs it with `wall_s` only. Nothing
ever sets `tokens_in` / `tokens_out`.

Token and dollar figures are therefore written to their own file,
`logs/coding_agent_usage.jsonl`, one line per LLM call:

```json
{"timestamp": …, "agent": "coding", "purpose": "generate", "attempt": 0,
 "model": "gpt-5", "is_real_model_call": true,
 "tokens_in": 8412, "tokens_out": 2170, "cost_usd": 0.032218, "idea": "…"}
```

Wiring this into `RunRecord.resources` means editing `orchestrator.py`, which
was out of scope this round. When someone is authorised:

- `LLMCodingAgent.last_usage` already holds the per-`implement()` totals, which
  is the right granularity — one `RunRecord` is one iteration.
- `UsageLog.as_resource_usage(wall_s)` is a ready-made shim, but it aggregates
  the *whole run*, not one iteration. Prefer `last_usage`.
- Both `_handle_successful_run` and `_handle_failed_run` build
  `ResourceUsage(wall_s=...)` and both need the change, or failed attempts
  (which still cost tokens — often the most) will report zero.

**Caveat on the dollar column:** `PRICING_USD_PER_MTOK` in `agent/coding/llm.py`
is a hardcoded list-price table and prices change. Token counts come straight
from the API response and are always exact; only `cost_usd` depends on that
table. Override without a code change via `OPENAI_PRICE_IN_PER_MTOK` /
`OPENAI_PRICE_OUT_PER_MTOK`.

## 2. `Diff.diff_path` is really the config path

`agents.py` documents `diff_path` as "where the change is recorded (patch file,
commit ref, ...)", but `orchestrator.py` passes it straight to
`Executor.run_seeds()` as `config_path`. The executable meaning wins, so this
agent puts the config file there. A real unified diff against the baseline is
still written as `changes.patch` in the solution dir and referenced from
`attempt.json` — it just can't live in the field named after it.

The clean fix is to give `Diff` a separate `config_path` field, or rename
`diff_path`. Both need an `orchestrator.py` change. There is a test
(`test_diff_path_is_the_config_file_the_orchestrator_will_pass_to_the_executor`)
pinning the current behaviour so the swap is a deliberate act.

## 3. ~~First-iteration `delta_vs_current_best` is meaningless~~ — FIXED

**Was:** the registry started empty, so `orchestrator.py` computed
`delta = agg.primary_mean - best.val_primary if best else agg.primary_mean`.
On iteration 1 that made the "delta" the *absolute score* (0.5989), and
`FakeEvaluatorAgent` accepts anything with `delta > margin` — so the first
iteration was always accepted, even when it lost to the baseline.

**Now:** `Orchestrator.bootstrap_baseline()` runs `solution/` as a genuine
iteration 0 before the research loop starts — a real `RunRecord` in the shared
history, registered through the same `_register_checkpoint()` an accepted
iteration uses. `scripts/run_loop.py` calls it automatically (pass
`--skip-baseline` to opt out).

The root problem was never really "the registry starts empty" — it was that
`solution/`'s known result was not represented anywhere in the loop's data
model. Making it a real iteration 0 fixed three symptoms at once, with no
change to delta computation, `registry.py` or `convergence.py`:

| symptom | now |
|---|---|
| delta computed against nothing → everything accepted | iteration 1 measures against 0.6016 and correctly REVERTs |
| `convergence.should_stop` needs `n_window+1` *scored* records; the baseline wasn't one | the baseline is the first scored iteration, so stalling is noticed a full iteration sooner |
| `Idea.parent_iteration` for the first hypothesis resolved to `None` | resolves to `0` |

**Iteration numbering now starts at 0.** Iteration 0 is reserved for the seeded
baseline; research-driven iterations are numbered from 1, as before. A run that
never bootstraps simply has no iteration 0 and behaves exactly as it always
did. Worth knowing when reading `runs.jsonl`.

Two consequences worth knowing:

- The baseline **counts toward `max_iterations`**, since
  `convergence.should_stop` counts every non-FAILED record. `run_loop.py`
  therefore treats `--max-iterations` as a count of *research* iterations and
  adds one slot for the baseline.
- If nothing ever beats the baseline, `registry.best()` **is** the baseline —
  and because bootstrapping runs it for real rather than trusting a recorded
  score, a complete artifact (`result.json`, `checkpoint.npz`,
  `val_predictions.npz`) actually exists at that path for a submission step to
  use.

`bootstrap_baseline()` is idempotent and crash-resume safe, keyed off the run
log on disk. It is the second change outside the Coding agent's lane (after
`executor.py`) and is isolated in its own commit for review.

## 4. What metric verification does *not* catch

`agent/verification.py` proves the reported metrics are arithmetically
consistent with the predictions the run persisted. It does not prove those
predictions came from the validation split. A `train.py` that trained on
validation, or that persisted test-split arrays, would verify clean. That is
data hygiene rather than arithmetic honesty and would need the executor to know
the split definitions.

---

# Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # then set KUAIRAND_PATH (and OPENAI_API_KEY)
```

```bash
# offline: no API key, no spend. Uses the hand-written template library.
python scripts/run_loop.py --offline

# live: real OpenAI generation.
python scripts/run_loop.py --model gpt-5 \
    --hypothesis "Replace pointwise logloss with a pairwise BPR ranking loss ..."
```

Outputs land in `logs/`: `runs.jsonl` (the agent-facing record),
`coding_agent_usage.jsonl` (tokens + cost), `artifacts/` (per-seed
`result.json`, `val_predictions.npz`, `checkpoint.npz`), and `quarantine/`
(hidden-test metrics, which no agent ever reads).

## Tests

```bash
pytest                                  # default: 100 tests, offline, ~2s
RUN_SLOW_TESTS=1 KUAIRAND_PATH=... pytest -m slow    # real data, ~100s
RUN_LLM_TESTS=1  OPENAI_API_KEY=...  pytest -m llm   # the only billable test
```
