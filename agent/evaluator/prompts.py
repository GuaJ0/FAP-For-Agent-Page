"""Prompt construction for the LLM Evaluator.

Separate from agent/coding/prompts.py -- this agent judges a result, it
doesn't write code, so the contract, the inputs, and the failure modes it
needs to reason about are entirely different. Reuses agent.coding.llm's
generic LLM/usage plumbing (agent/coding/__init__.py's public exports), not
anything code-generation-specific.
"""
from __future__ import annotations

from agent.records import RunRecord

SYSTEM_PROMPT = """\
You are the Evaluator agent in an autonomous ML research loop for the \
KuaiRand-Pure ranking competition (task: rank videos per user; metric: \
primary = mean(GAUC, nDCG@5); a Factorization Machine trained with pointwise \
logloss is the baseline).

You are shown one experiment: a hypothesis the Coding agent implemented, and \
the validation metrics it produced, compared against the current best result \
in the run so far. Decide what happens to it:

  ACCEPT  - this result is better than the current best; make it the new
            incumbent the rest of the run builds on and compares against.
  REVERT  - this result did not beat the current best; discard it, keep the
            current best, and let the Research agent try something else.
  ABANDON - not just "this attempt didn't work" but "this whole research
            direction is a dead end" -- e.g. the same underlying idea has now
            underperformed multiple times in different forms, or the result
            is dramatically worse in a way that suggests a fundamental
            mismatch with the data/task, not something worth refining further.
            Use this sparingly: REVERT is the default for "didn't beat the
            baseline this time"; ABANDON is for "stop spending iterations on
            this idea at all."

Respond with EXACTLY one JSON object and nothing else, on one line or pretty
printed, no markdown fence, no commentary outside the JSON:

    {"decision": "accept" | "revert" | "abandon", "commentary": "..."}

`commentary` is 1-3 sentences. It is written back into the shared run history \
and is what the Research agent reads to decide what to try next -- so say \
something a hypothesis-generator could act on (what likely caused this \
result, what direction looks more or less promising), not just a restatement \
of the numbers already shown to you.\
"""

MAX_HISTORY_ITEMS = 5   # bounded context: see build_judge_prompt's docstring
MAX_COMMENTARY_CHARS_PER_ITEM = 300


def _format_history_item(r: RunRecord) -> str:
    if r.aggregate is None:
        outcome = f"status={r.status.value}, no validation metrics (run failed)"
    else:
        decision = r.decision.value if r.decision else "pending"
        outcome = f"primary={r.aggregate.primary_mean:.4f}, decision={decision}"
    commentary = ""
    for e in r.events:
        if e.type == "evaluator_commentary":
            commentary = f" | prior evaluator note: {e.detail[:MAX_COMMENTARY_CHARS_PER_ITEM]}"
            break
    return f"- iteration {r.iteration}: \"{r.hypothesis}\" -> {outcome}{commentary}"


def build_judge_prompt(record: RunRecord, history: list[RunRecord], current_best_primary: float | None) -> str:
    """The one prompt this agent ever builds: judge `record` against
    `current_best_primary`, with a bounded slice of `history` for context on
    what has already been tried.

    Bounded on purpose: only the last MAX_HISTORY_ITEMS records are included,
    and each prior evaluator note is truncated. Unbounded context growth as
    iteration count rises was flagged in the harness's own audit as a real
    risk for token-cost scoring; this keeps the Evaluator's prompt size flat
    regardless of how long the run has been going, the same way the Coding
    agent's ideas.md read is capped.
    """
    a = record.aggregate
    parts = [
        "## Hypothesis being judged\n", record.hypothesis.strip(), "\n\n",
        "## This experiment's validation metrics\n",
        f"primary={a.primary_mean:.4f} (std {a.primary_std:.4f}), "
        f"GAUC={a.gauc_mean:.4f}, nDCG@5={a.ndcg5_mean:.4f}, over {a.n_seeds} seed(s)\n",
    ]
    if current_best_primary is not None:
        delta = a.primary_mean - current_best_primary
        parts.append(
            f"\nCurrent best (incumbent) validation primary: {current_best_primary:.4f}\n"
            f"Delta (this experiment - incumbent): {delta:+.4f}\n"
        )
    else:
        parts.append("\nNo incumbent exists yet -- this would be the first accepted result.\n")

    recent = history[-MAX_HISTORY_ITEMS:]
    if recent:
        parts.append("\n## Recent history (most recent last)\n")
        parts.extend(_format_history_item(r) + "\n" for r in recent)

    parts.append(
        "\n## Your task\nReturn the JSON verdict described in the system prompt. "
        "Nothing else."
    )
    return "".join(parts)
