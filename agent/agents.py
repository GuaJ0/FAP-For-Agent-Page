"""Pluggable interfaces for the three LLM agents, plus deterministic fakes
for testing the harness without any LLM calls.

Real implementations (backed by actual model calls) get wired in later by
satisfying these Protocols -- nothing in orchestrator.py should ever import
an LLM client directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from agent.records import Decision, RunRecord


@dataclass(frozen=True)
class Idea:
    hypothesis: str
    parent_iteration: Optional[int]


@dataclass(frozen=True)
class AgentUsage:
    """What one implement() call cost. Optional -- a CodingAgent that doesn't
    track usage simply leaves Diff.usage as None.

    `cost_usd` is carried here but deliberately does NOT get a field on
    ResourceUsage. Token counts are ground truth from the API; a dollar figure
    is derived from a mutable list-price table, so persisting one into an
    append-only log freezes a number that silently goes stale as prices change.
    Tokens are what's stored; cost stays derivable (agent/coding/llm.py's
    pricing table) and is logged alongside the model name in
    logs/coding_agent_usage.jsonl. The orchestrator also writes it into a
    per-iteration Event so it's visible in runs.jsonl without a schema change.
    """
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class Diff:
    diff_path: str      # where the change is recorded (patch file, commit ref, ...)
    solution_dir: str    # directory containing train.py + config.yaml, ready for the executor
    # Optional: what producing this cost. orchestrator.py folds it into
    # RunRecord.resources; None means "not tracked" and yields today's
    # wall_s-only ResourceUsage.
    usage: Optional[AgentUsage] = None


class ResearchAgent(Protocol):
    def propose(self, history: list[RunRecord]) -> Idea:
        """Propose the next idea given everything tried so far."""
        ...


class CodingAgent(Protocol):
    def implement(self, idea: Idea, feedback: Optional[str]) -> Diff:
        """Turn an idea into a runnable solution dir. `feedback` is the prior
        attempt's failure (traceback tail / evaluator note) on a retry, or
        None on the first attempt at this idea."""
        ...


class EvaluatorAgent(Protocol):
    def judge(self, record: RunRecord, history: list[RunRecord]) -> Decision:
        """Decide what to do with a successfully-run iteration: keep it
        (ACCEPT), discard it (REVERT), or give up on this line entirely
        (ABANDON). Only called for iterations that actually produced
        validation metrics -- executor-level failures are handled by the
        orchestrator's retry policy, not this method."""
        ...


class FakeResearchAgent:
    """Cycles through a fixed list of hypotheses deterministically, so tests
    don't depend on iteration order beyond that list."""

    def __init__(self, hypotheses: list[str]):
        self._hypotheses = hypotheses
        self._i = 0

    def propose(self, history: list[RunRecord]) -> Idea:
        h = self._hypotheses[self._i % len(self._hypotheses)]
        self._i += 1
        parent = history[-1].iteration if history else None
        return Idea(hypothesis=h, parent_iteration=parent)


class FakeCodingAgent:
    """Writes a solution dir pointing at fixtures/fake_train.py with a config
    controlled by `outcomes` -- a queue of dicts merged into the fixture's
    JSON config, popped one per `implement()` call. Lets a test script an
    exact sequence of crash/timeout/bad-output/success outcomes."""

    def __init__(self, work_dir, outcomes: list[dict]):
        import itertools
        self._work_dir = work_dir
        self._outcomes = iter(outcomes)
        self._counter = itertools.count()

    def implement(self, idea: Idea, feedback: Optional[str]) -> Diff:
        import json
        import shutil
        from pathlib import Path

        n = next(self._counter)
        sol_dir = Path(self._work_dir) / f"attempt_{n}"
        sol_dir.mkdir(parents=True, exist_ok=True)
        fixture = Path(__file__).resolve().parent.parent / "fixtures" / "fake_train.py"
        shutil.copy(fixture, sol_dir / "train.py")

        outcome = next(self._outcomes, {"mode": "normal"})
        config_path = sol_dir / "config.json"
        config_path.write_text(json.dumps(outcome))

        return Diff(diff_path=str(config_path), solution_dir=str(sol_dir))


class FakeEvaluatorAgent:
    """Deterministic rule: accept if this iteration beats the best validation
    primary seen so far by more than a margin, else revert."""

    def __init__(self, margin: float = 0.0):
        self._margin = margin

    def judge(self, record: RunRecord, history: list[RunRecord]) -> Decision:
        if record.aggregate is None:
            raise ValueError("judge() called on a record with no aggregate metrics")
        if record.delta_vs_current_best is not None and record.delta_vs_current_best > self._margin:
            return Decision.ACCEPT
        return Decision.REVERT
