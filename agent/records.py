"""The result record schema: one RunRecord per orchestrator iteration.

RunRecord is the sole shape that crosses from executor.py into the agent
loop (research/coding/evaluator) and into logs/runs.jsonl. It structurally
cannot hold a hidden-test metric -- there is no field for it anywhere in this
module. Test-split numbers live only in the quarantine file written by
executor.py, which nothing here ever reads.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from runlog.emit import append_line, read_lines


class Status(str, Enum):
    SUCCESS = "success"
    SUCCESS_AFTER_RETRY = "success_after_retry"
    FAILED = "failed"
    ABANDONED = "abandoned"


class Decision(str, Enum):
    ACCEPT = "accept"
    REVERT = "revert"
    ABANDON = "abandon"


class FailureKind(str, Enum):
    """Classification produced by executor.py when a subprocess run doesn't
    yield usable validation metrics."""
    TIMEOUT = "timeout"
    OOM = "oom"
    CRASH = "crash"
    BAD_OUTPUT = "bad_output"


@dataclass(frozen=True)
class SeedMetrics:
    """One subprocess run. Validation-split metrics only -- see module docstring."""
    seed: int
    primary: Optional[float]
    gauc: Optional[float]
    ndcg5: Optional[float]
    epochs_run: Optional[int]
    wall_s: float
    failure_kind: Optional[FailureKind] = None
    traceback_tail: Optional[str] = None  # last N stderr lines, only set on failure
    artifact_dir: Optional[str] = None    # persistent dir holding result.json (+ any checkpoint), set on success only

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["failure_kind"] = self.failure_kind.value if self.failure_kind else None
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "SeedMetrics":
        d = dict(d)
        d.setdefault("artifact_dir", None)
        if d.get("failure_kind") is not None:
            d["failure_kind"] = FailureKind(d["failure_kind"])
        return cls(**d)


@dataclass(frozen=True)
class AggregateMetrics:
    """Mean/std across seeds that produced usable metrics. Validation only."""
    primary_mean: float
    primary_std: float
    gauc_mean: float
    ndcg5_mean: float
    n_seeds: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "AggregateMetrics":
        return cls(**d)


@dataclass(frozen=True)
class Event:
    type: str            # "research_proposed" | "coding_attempt" | "eval_started" | "retry" | "abandon" | "escalate" | ...
    detail: str
    agent_action: str    # "research" | "coding" | "evaluator" | "orchestrator"
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Event":
        return cls(**d)


@dataclass(frozen=True)
class ResourceUsage:
    wall_s: float
    gpu_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ResourceUsage":
        return cls(**d)


@dataclass
class RunRecord:
    iteration: int
    parent_iteration: Optional[int]
    timestamp: str                        # ISO 8601
    hypothesis: str
    diff_path: Optional[str]
    status: Status
    seeds: list[SeedMetrics]
    aggregate: Optional[AggregateMetrics]  # None iff every seed failed
    delta_vs_current_best: Optional[float]
    decision: Optional[Decision]
    events: list[Event]
    resources: ResourceUsage
    manual_intervention: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "parent_iteration": self.parent_iteration,
            "timestamp": self.timestamp,
            "hypothesis": self.hypothesis,
            "diff_path": self.diff_path,
            "status": self.status.value,
            "seeds": [s.to_json() for s in self.seeds],
            "aggregate": self.aggregate.to_json() if self.aggregate else None,
            "delta_vs_current_best": self.delta_vs_current_best,
            "decision": self.decision.value if self.decision else None,
            "events": [e.to_json() for e in self.events],
            "resources": self.resources.to_json(),
            "manual_intervention": self.manual_intervention,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "RunRecord":
        return cls(
            iteration=d["iteration"],
            parent_iteration=d["parent_iteration"],
            timestamp=d["timestamp"],
            hypothesis=d["hypothesis"],
            diff_path=d["diff_path"],
            status=Status(d["status"]),
            seeds=[SeedMetrics.from_json(s) for s in d["seeds"]],
            aggregate=AggregateMetrics.from_json(d["aggregate"]) if d["aggregate"] else None,
            delta_vs_current_best=d["delta_vs_current_best"],
            decision=Decision(d["decision"]) if d["decision"] else None,
            events=[Event.from_json(e) for e in d["events"]],
            resources=ResourceUsage.from_json(d["resources"]),
            manual_intervention=d.get("manual_intervention", False),
        )


class RunLog:
    """Append-only JSONL log of RunRecords. This is the agent-facing history:
    everything Research/Coding/Evaluator see about past iterations comes from
    reading this file, never from the quarantine file."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, record: RunRecord) -> None:
        append_line(self.path, record.to_json())

    def read_all(self) -> list[RunRecord]:
        return [RunRecord.from_json(d) for d in read_lines(self.path)]
