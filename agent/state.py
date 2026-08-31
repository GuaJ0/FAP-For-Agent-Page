"""Orchestrator's on-disk checkpoint.

Persisted after every step so a crash mid-idea resumes with fix_attempts,
the in-flight idea, and the abandonment streak intact -- the alternative
(losing this on crash) would silently reset tier-1/tier-2 counters and let
a bad idea consume far more than its attempt budget across restarts.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from agent.agents import AgentUsage, Idea


@dataclass
class OrchestratorState:
    iteration: int = 0
    current_idea: Optional[dict] = None        # Idea.__dict__ while an idea is in flight, else None
    fix_attempts: int = 0
    idea_start_time: Optional[float] = None     # epoch seconds; set when the idea was first proposed
    last_failure_feedback: Optional[str] = None
    consecutive_abandonments: int = 0
    # Consecutive ResearchAgent.propose() failures -- distinct from
    # consecutive_abandonments, which counts failed IDEAS. This counts failing
    # to get an idea at all, and resets on the next successful propose().
    consecutive_research_failures: int = 0
    halted: bool = False
    halt_reason: Optional[str] = None
    # Set when ResearchAgent.propose() reported it has nothing left to propose.
    # Terminal and deliberately persisted: a resumed run whose backlog was
    # already exhausted must stop again rather than re-derive the same finish.
    research_exhausted: bool = False
    research_exhausted_reason: Optional[str] = None
    run_start_time: Optional[float] = None       # epoch seconds; set once, for adaptive-seeding projections
    seed_costs: list = field(default_factory=list)  # observed wall_s per completed seed run
    # One-shot: set True by resume_after_human(), consumed (reset to False) by
    # the next RunRecord produced, which is stamped manual_intervention=True.
    # Marks exactly the iteration that only happened because a human cleared
    # a tier-2 halt -- not every iteration for the rest of the run.
    manual_intervention_pending: bool = False

    def get_current_idea(self) -> Optional[Idea]:
        """Rehydrate the in-flight Idea, nested AgentUsage included.

        set_current_idea() persists via asdict(), which flattens a nested
        AgentUsage into a plain dict; Idea(**d) would hand that dict straight
        back as Idea.usage, so a resumed idea would carry something that looks
        like usage and has no attributes. Rebuilt explicitly instead.

        A state file written before Idea grew a `usage` field simply has no
        such key, and the field's default (None) applies -- old checkpoints
        resume unchanged.
        """
        if not self.current_idea:
            return None
        fields = dict(self.current_idea)
        usage = fields.get("usage")
        if isinstance(usage, dict):
            fields["usage"] = AgentUsage(**usage)
        return Idea(**fields)

    def set_current_idea(self, idea: Optional[Idea]) -> None:
        self.current_idea = asdict(idea) if idea else None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "OrchestratorState":
        return cls(**d)


class StateStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> OrchestratorState:
        if not self.path.exists():
            return OrchestratorState()
        return OrchestratorState.from_json(json.loads(self.path.read_text()))

    def save(self, state: OrchestratorState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(state.to_json(), indent=2))
        os.replace(tmp, self.path)  # atomic on POSIX: a crash mid-write can't corrupt the live file
