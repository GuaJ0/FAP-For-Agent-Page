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

## Permitted libraries and frameworks

**External open-source libraries and frameworks (PyTorch, JAX, scikit-learn, pandas, etc.) are
permitted for the Coding agent to use.** This is not an official challenge restriction and never
was; the NumPy-only sandbox reflected what the *baseline* (`solution/train.py`) happened to be
written with. This matches `docs/research-agent.md`'s "Feasibility and dependencies" section,
which already tells the Research agent that open-source ML libraries may be proposed when
justified.

**The code now matches this.** The standing contradiction — Research free to propose PyTorch while
the Coding sandbox was guaranteed to reject it — is resolved.

### Availability is measured, not declared

`agent/coding/agent.py` used to carry two hand-maintained lists: an `ALLOWED_IMPORTS` allowlist
(numpy + stdlib) and a `KNOWN_UNAVAILABLE` dict of rejection reasons. Both had drifted from
reality — the dict asserted that pandas, scikit-learn and scipy were "not installed" when all
three were installed the whole time. A hand-maintained claim about the environment is a claim that
goes stale silently.

They are replaced by two rules:

| | what it is | how it is decided |
|---|---|---|
| `FORBIDDEN_IMPORTS` | sandbox rules — `subprocess`, `socket`, `requests`, `urllib` | refused **however the environment is provisioned**. Every one is in the stdlib and would pass an availability test, so this cannot be inferred |
| everything else | any library | allowed **iff `importlib.util.find_spec` can resolve it** |

This implements the provisioning rule directly: a dependency is legal exactly when it is installed.
Provisioning one needs no code change — `pip install torch` makes torch legal on the next run, and
removing it makes it illegal again. `find_spec` resolves without executing, so probing a heavy
framework costs nothing.

Measuring in the agent's own process is valid because `executor.py` launches solutions with
`sys.executable` — the same interpreter — so what imports here imports there.

### What the model is told

`SYSTEM_PROMPT` is now `SYSTEM_PROMPT_TEMPLATE` plus `prompts.system_prompt(available_libraries)`,
rendered per call against the measured environment, so the prompt names what is actually installed
rather than a constant that can drift from it. Rendering is by substitution, not `str.format` —
the template contains literal JSON braces that `format()` would try to read as fields.

A permitted-but-absent import is still caught by the **static** check rather than at runtime. That
distinction is worth keeping: the static check costs nothing and says what to use instead, whereas
letting it reach the executor costs a full multi-seed run and surfaces as a `CRASH` traceback that
reads like a modelling bug rather than a missing package.

### Constraints that still apply, whatever the library

No network access, no downloads, no subprocess spawning, a single CPU core budget, and the
executor's per-run timeout — a heavier framework's import and setup cost must still fit inside it.
`evaluate.py` remains the sole scoring authority: `evaluate()` must be imported, never
reimplemented.

### Current state of this machine

Available to a generated solution: `numpy`, `scipy`, `pandas`, `scikit-learn`, `torch` (2.13).

`torch` was provisioned specifically so solutions needing gradients — DIN-style sequence models,
multi-task heads, DeepFM — do not have to hand-derive backprop, which would confound "the mechanism
failed" with "the gradient was hard to write". It is pinned in `requirements.txt`, so it must be
installed in the graded run's environment too or findings measured with it will not transfer.

Provisioning it required **no change to the import policy** — `available_third_party()` and
`static_check()` picked it up on the next call, and the rendered system prompt began advertising it
automatically. That is the payoff of measuring availability rather than declaring it.

**Threading caveat.** `torch.get_num_threads()` defaults to 4 here, against the documented
single-CPU-core budget, so the system prompt now instructs solutions to call
`torch.set_num_threads(1)`. Note this is guidance in the prompt, not an enforced limit — nothing in
`static_check` or the executor currently rejects a solution that thread-pools anyway, and NumPy's
BLAS can do the same thing. Worth hardening (e.g. exporting `OMP_NUM_THREADS=1` from the executor)
if CPU-time comparability starts to matter.

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

## Accumulation: each idea builds on the current best

The agent used to start every idea from the static `solution/train.py`, so
improvements never compounded — iteration 5 was written against the baseline,
not against whatever iterations 1–4 had established.

It now resolves the current best by chaining three things the harness already
maintains:

```
registry.best()   → the accepted iteration with the best validation primary
runs.jsonl        → that iteration's RunRecord
record.diff_path  → the config the executor ran, whose sibling train.py is
                    the source that actually produced the score
```

Pass `registry_path` and `run_log_path` to enable it (`run_loop.py` does).
Both default to `None`, which keeps the old static-baseline behaviour exactly
— so existing callers and tests are unaffected.

A useful property falls out of `bootstrap_baseline()`: immediately after
bootstrapping, the best is iteration 0 whose config is `solution/config.yaml`,
so the sibling `train.py` **is** `solution/train.py`. "Current best" and
"static baseline" resolve to the same file until something actually beats the
baseline, which is what makes this safe to turn on by default. That identity
is pinned by a test.

Resolution fails soft in every direction — missing registry, empty registry,
no matching record, a cleaned-up solution dir, a corrupt run log — falling
back to the static baseline rather than raising. A provenance problem must not
take down a run. Each attempt's `attempt.json` records what it was `built_from`.

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

## 1. ~~`RunRecord.resources` is not populated~~ — FIXED

**Was:** `ResourceUsage(wall_s, gpu_s, tokens_in, tokens_out)` sat on every
`RunRecord` and only `wall_s` was ever set, so every record reported zero
tokens regardless of what the run actually spent.

**Now:** `Diff` carries an optional `AgentUsage(tokens_in, tokens_out,
cost_usd)`, `LLMCodingAgent` populates it with that `implement()` call's
totals (repair cycles included), and `orchestrator.py` folds it into
`ResourceUsage`. Both the success **and** the failure path — a failed attempt
still costs tokens, and it's usually the most expensive one since it's the one
that burned repairs.

`usage=None` (the default, and what `FakeCodingAgent` produces) yields exactly
the old `wall_s`-only record, which is what keeps `tests/test_orchestrator.py`
passing unmodified.

**`cost_usd` deliberately has no `ResourceUsage` field.** Token counts are
ground truth from the API response; a dollar figure is derived from a mutable
list-price table, so persisting one into an append-only log freezes a number
that goes quietly stale as prices change. Tokens are what's stored; cost stays
derivable, is logged next to the model name in `logs/coding_agent_usage.jsonl`,
and the orchestrator also writes it into a per-iteration `coding_usage` event
so it's visible in `runs.jsonl` without a `records.py` schema change.

## 2. ~~`Diff.diff_path` is really the config path~~ — FIXED

**Was:** `agents.py` documented `Diff.diff_path` as "where the change is
recorded (patch file, commit ref, ...)", but `orchestrator.py` passed it
straight to `Executor.run_seeds()` as the config path. Documented meaning and
executable meaning had drifted, and the executable one was load-bearing.

**Now:** `Diff` carries `config_path` (what the executor runs) and
`patch_path` (a real unified diff, `None` when the agent doesn't produce one).

**The name `diff_path` is deliberately gone from `Diff` rather than reused for
the patch.** `RunRecord.diff_path` in `records.py` still means "the config the
executor ran", so keeping the name on `Diff` with a *different* meaning one
layer up would have replaced the old ambiguity with a worse one — two fields,
same name, adjacent layers, opposite meanings, crossing in `orchestrator.py`.
With the name gone there is exactly one `diff_path` in the codebase and it has
one meaning.

`RunRecord.diff_path` keeps taking the **config** path, for three reasons
(spelled out in `Orchestrator._record_diff_path`):

1. `runs.jsonl`'s meaning is unchanged — repointing it would silently make old
   and new lines mean different things in an append-only log with no version
   marker.
2. It's the only path always present: `FakeCodingAgent` and the bootstrapped
   baseline produce no patch.
3. Accumulation resolves the current best by taking this path's **sibling
   `train.py`**, which works precisely because a config lives inside its
   solution dir.

### The permanent record carries both paths

`RunRecord` now has `patch_path` alongside `diff_path`, so `runs.jsonl` alone
answers *what code did this iteration actually run* — previously it got you
only to the settings file, and finding the diff meant hunting for the solution
directory by hand. That matters because the hypothesis log is a graded artifact
and an LLM wrote the code: verifying "did this iteration implement what its
hypothesis claimed" should be a matter of following the run log.

| field | holds | `None` when |
|---|---|---|
| `diff_path` | the **config** the executor ran | never (always recorded) |
| `patch_path` | the **unified diff** of the code change | the producer makes no patch — `FakeCodingAgent`, and the bootstrapped baseline, which is a pre-existing solution rather than an edit to one |

`diff_path`'s meaning and how it's populated are untouched — this is purely
additive.

**Backward compatibility.** `runs.jsonl` is append-only and permanent, so
`from_json()` reads the new field with `d.get("patch_path")`, following the
precedent `manual_intervention` already set. Lines written before the field
existed load with `patch_path=None`, which reads correctly: absent is
indistinguishable from "the producer never made one". Verified against 12
records in 6 run logs written before the field existed, plus tests in
`tests/test_records_serialization.py`.

**Residual, knowingly left:** a field named `diff_path` holding a config path.
Renaming it means deciding what already-written JSONL lines mean — a bigger
call, and a separate one. Now at least the record is no longer *missing*
the thing the name suggests; it's in `patch_path` next door.

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

- The baseline does **not** consume a `max_iterations` slot.
  `convergence.should_stop` excludes `BOOTSTRAP_ITERATION` from that count, so
  `max_iterations=50` gets you 50 real research attempts and callers need no
  arithmetic. (Briefly after bootstrapping first landed, the baseline *did*
  count and `run_loop.py` compensated with a `+1`; both are gone.)
- If nothing ever beats the baseline, `registry.best()` **is** the baseline —
  and because bootstrapping runs it for real rather than trusting a recorded
  score, a complete artifact (`result.json`, `checkpoint.npz`,
  `val_predictions.npz`) actually exists at that path for a submission step to
  use.

`bootstrap_baseline()` is idempotent and crash-resume safe, keyed off the run
log on disk. It is the second change outside the Coding agent's lane (after
`executor.py`) and is isolated in its own commit for review.

### How the baseline interacts with each stopping rule

`should_stop()` has three checks, the baseline record is visible to all three,
and each wants something different from it. The asymmetry is deliberate —
"making it consistent" in either direction is a plausible and wrong edit, so
it is pinned by tests in `tests/test_convergence.py`.

| check | baseline counted? | why |
|---|---|---|
| `max_iterations` | **no** | counts *research attempts*; the baseline is the incumbent they're measured against, not an attempt |
| stalled-progress window | **yes** | counts *scored results*; the baseline is the score everything must beat, so it's the window's first data point |
| `max_wall_s` | **yes** | records are timestamped at *completion*, so the iteration-0 record marks when research **began** — its own runtime is already outside the window |

That last one is the easy one to get backwards. Excluding the baseline would
restart the clock at iteration 1's *completion* and silently drop iteration 1's
duration from the 6h budget — less accurate and more permissive than counting
it. `BOOTSTRAP_ITERATION` lives in `agent/config.py` (not `orchestrator.py`)
because `convergence.py` needs it too and `orchestrator.py` already imports
`should_stop` from `convergence.py`; `config.py` is imported by both and
imports neither.

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
