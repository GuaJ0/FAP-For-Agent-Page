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
class Diff:
    diff_path: str      # where the change is recorded (patch file, commit ref, ...)
    solution_dir: str    # directory containing train.py + config.yaml, ready for the executor


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
