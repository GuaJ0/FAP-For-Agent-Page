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
    """What a CodingAgent hands back: a runnable solution, plus provenance.

    NAMING -- this class used to have a single `diff_path` field, documented as
    "where the change is recorded (patch file, commit ref, ...)" but in fact
    consumed by orchestrator.py as the config path handed to the executor. The
    documented meaning and the executable meaning had drifted apart, and the
    executable one was load-bearing.

    Split into two explicitly named fields rather than repurposing the old
    name. `diff_path` deliberately no longer exists here: RunRecord.diff_path
    (agent/records.py) still means "the config the executor ran", and having
    the same name mean a patch file one layer up and a config path one layer
    down would be a worse trap than the original ambiguity. With the name gone
    from this class there is exactly one `diff_path` in the codebase and it has
    one meaning.

    Renaming RunRecord's own field is a separate, larger decision -- it changes
    what an already-written line of runs.jsonl means -- and is deliberately not
    taken here.
    """
    config_path: str     # the config file the executor passes to train.py
    solution_dir: str    # directory containing train.py + config, ready for the executor
    # Optional: a real unified diff against the source this was built from.
    # None when the agent doesn't produce one (FakeCodingAgent, the seeded
    # baseline) -- nothing in the harness requires it, it's for humans.
    patch_path: Optional[str] = None
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


@dataclass(frozen=True)
class Verdict:
    """What an EvaluatorAgent hands back for one judged iteration.

    Not a bare Decision, on purpose: an Evaluator that reasons about a result
    (rather than just comparing two floats) has something to say about *why*,
    and the harness's own audit flagged that this commentary had nowhere to
    go -- Research could not see it because nothing carried it. `commentary`
    is written into the RunRecord's events (agent_action="evaluator"), so any
    ResearchAgent reading `history` on its next propose() call sees it, with
    no RunRecord schema change needed.

    `usage` follows the same shape as Diff.usage (AgentUsage) for the same
    reason: an LLM-backed Evaluator spends real tokens judging a result, and
    that cost belongs in the same per-iteration ResourceUsage the CodingAgent's
    usage already folds into -- not a second, uncounted channel.
    """
    decision: Decision
    commentary: str = ""
    usage: Optional[AgentUsage] = None


class EvaluatorAgent(Protocol):
    def judge(self, record: RunRecord, history: list[RunRecord]) -> Verdict:
        """Decide what to do with a successfully-run iteration: keep it
        (ACCEPT), discard it (REVERT), or give up on this line entirely
        (ABANDON). Only called for iterations that actually produced
        validation metrics -- executor-level failures are handled by the
        orchestrator's retry policy, not this method.

        ABANDON here counts toward the same tier-2 consecutive-abandonment
        streak that tier-1 attempt-cap exhaustion does (Orchestrator ties
        _close_idea's `abandoned` flag to this decision) -- a hypothesis a
        real Evaluator judges as a dead end is exactly as costly to the run's
        escalation budget as one that never got past 3 fix attempts."""
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

        # No patch_path: this fixture copies a fixed file rather than editing
        # anything, so there is no diff to point at.
        return Diff(config_path=str(config_path), solution_dir=str(sol_dir))


class FakeEvaluatorAgent:
    """Deterministic rule: accept if this iteration beats the best validation
    primary seen so far by more than a margin, else revert. Never abandons --
    a bare margin comparison has no basis to judge a whole hypothesis a dead
    end, so it leaves that call to a real Evaluator (see agent/evaluator/)."""

    def __init__(self, margin: float = 0.0):
        self._margin = margin

    def judge(self, record: RunRecord, history: list[RunRecord]) -> Verdict:
        if record.aggregate is None:
            raise ValueError("judge() called on a record with no aggregate metrics")
        if record.delta_vs_current_best is not None and record.delta_vs_current_best > self._margin:
            return Verdict(Decision.ACCEPT)
        return Verdict(Decision.REVERT)
