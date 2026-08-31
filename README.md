# FAP-For-Agent-Page

Autonomous ML research agent for TikTok TechJam Track 2: improve a Factorization
Machine (FM) baseline on the KuaiRand-Pure ranking task, measured by
`primary = mean(GAUC, nDCG@5)` on the validation split.

## The answer

**`submission/`** is the final result: **valid primary 0.60420**, +0.00258 over
the official FM baseline (0.60161), from a hand-implemented multi-task FM with
`is_click`/`is_like` auxiliary heads on the shared embeddings. See
[`submission/README.md`](submission/README.md) for the full comparison table,
exclusion rationale, and independent re-verification of every claimed number
against the vendored `harness/evaluate.py` (the sole scoring authority).

| Model | valid primary | vs. baseline |
|---|---|---|
| **Multi-task FM (submitted)** | **0.60420** | **+0.00258** |
| Combined (sequential attention + multi-task) | 0.60252 | +0.00091 |
| Multi-task, all 6 labels | ~0.60253 | +0.00092 |
| Sequential attention (self-attention over watch history) | 0.60137 | -0.00024 |
| Official FM baseline (`solution/train.py`) | 0.60161 | — |

## Repository layout

| path | what it is |
|---|---|
| `agent/` | The three-agent loop: `agent/research/` (proposes ideas), `agent/coding/` (implements them as diffs), `agent/evaluator/` (accepts/reverts/abandons). `agent/orchestrator.py`, `agent/executor.py`, `agent/records.py`, `agent/registry.py`, `agent/convergence.py` are the harness itself. |
| `harness/` | Vendored, unmodified competition harness — `evaluate.py` (sole scoring authority), `data.py`. Never edited. |
| `solution/` | The baseline FM implementation the loop starts from and measures against. |
| `experiments/` | Hand-implemented, non-agentic experiments testing specific hypotheses in isolation (multi-task, sequential attention, combined, all-labels) after the autonomous loop's own attempts stayed under the convergence threshold. |
| `submission/` | The final, durable answer — checkpoint, config, predictions, and a fully-verified README. This directory is what gets graded. |
| `runlog/` | Renders `logs/registry.json` + `logs/runs.jsonl` into a human-readable run report. |
| `scripts/run_loop.py` | Entry point for the autonomous loop. |
| `docs/` | Deep dives on each agent and on the pre-run exploration campaign — see below. |
| `tests/` | Unit + integration coverage for the harness and agents. |

## Running the loop

```bash
# offline: no API key, no spend, deterministic Research + template-based Coding/Evaluator
python scripts/run_loop.py --offline

# live Coding/Evaluator, deterministic offline Research (default Research mode)
python scripts/run_loop.py --model gpt-5

# fully live: Research/Coding/Evaluator all LLM-backed, grounded in a bundled citation catalog
python scripts/run_loop.py --live-research --model gpt-5
```

Requires `KUAIRAND_PATH` (and `OPENAI_API_KEY` for live modes) in `.env`. The
loop stops on the competition's convergence rule: `eps=0.002` not cleared over
the trailing 3 scored iterations, OR 50 concluded iterations, OR 6h wall-clock.

## Docs

- [`docs/research-agent.md`](docs/research-agent.md) — Research agent design, offline vs. live modes
- [`docs/coding-agent.md`](docs/coding-agent.md) — Coding agent, diff generation, retry handling
- [`docs/exploration-campaign.md`](docs/exploration-campaign.md) — the pre-run pass that seeded `agent/research/findings.jsonl` for real
- [`docs/results.md`](docs/results.md) — environment/baseline verification log

## Compute & cost

Per-run resource usage (LLM tokens, cost, CPU-hours) is written to
`logs/summary.json` at the end of each `run_loop.py` invocation and is
regenerated fresh every run — it is not a running total across the project's
history. The hand-implemented work in `experiments/` and `submission/` used no
LLM API calls (pure local PyTorch); its cost is real CPU-time only, not
captured in any `logs/` file since it ran outside the orchestrator.
