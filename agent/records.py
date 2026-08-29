"""The result record schema: one RunRecord per orchestrator iteration.

RunRecord is the sole shape that crosses from executor.py into the agent
loop (research/coding/evaluator) and into logs/runs.jsonl. It carries two
paths and they are not the same thing: `diff_path` is the config file the
executor ran, `patch_path` the unified diff of the code change (when there
was one). Both are optional to a reader -- see from_json. It structurally
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
    # The run finished and produced a well-formed result.json, but the metrics
    # it claimed did not survive an independent re-score of the raw predictions
    # it persisted, through the vendored (authoritative) evaluate.py. Distinct
    # from BAD_OUTPUT -- the output is *shaped* fine, it just isn't true -- so
    # the Coding agent gets "your numbers are wrong" as feedback rather than
    # "your file is malformed". See agent/verification.py.
    METRIC_MISMATCH = "metric_mismatch"


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
    # Real measured CPU time (user+sys) the subprocess consumed, from
    # resource.getrusage(RUSAGE_CHILDREN) deltas -- not wall_s. On a
    # single-threaded workload these are close, but a BLAS backend that uses
    # multiple threads for matrix ops would make wall_s understate actual
    # compute consumed. Set on every path, including failures/timeouts: a
    # crashed or timed-out run still burned real CPU. See agent/executor.py.
    cpu_s: float = 0.0

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["failure_kind"] = self.failure_kind.value if self.failure_kind else None
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "SeedMetrics":
        d = dict(d)
        d.setdefault("artifact_dir", None)
        d.setdefault("cpu_s", 0.0)
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
    # Real measured CPU time (sum of SeedMetrics.cpu_s across this iteration's
    # seeds) in hours, not derived from wall_s. This system trains on numpy
    # over CPU only -- gpu_s stays 0.0 by design, not by omission, and
    # cpu_hours is what actually reports the compute Feasibility scoring asks
    # for. See Orchestrator._resources().
    cpu_hours: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ResourceUsage":
        return cls(**d)  # a missing cpu_hours key (old records) uses the field default


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
    # Path to the unified diff for this iteration's code change, when the
    # CodingAgent produced one. `diff_path` above holds the *config* the
    # executor ran (see Orchestrator._record_diff_path for why that name and
    # meaning are left alone); this is the actual code change, so the run log
    # can answer "did this iteration implement what its hypothesis claimed"
    # without hunting for the solution directory by hand.
    #
    # Optional, and appended after the existing defaulted field rather than
    # slotted next to diff_path, so positional construction keeps working.
    # None whenever the producer makes no patch -- FakeCodingAgent, and the
    # bootstrapped baseline, which is a pre-existing solution rather than an
    # edit to one.
    patch_path: Optional[str] = None

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
            "patch_path": self.patch_path,
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
            # .get(), not [], for the same reason as manual_intervention above:
            # runs.jsonl is append-only, so lines written before this field
            # existed must keep reading. Absent means None, i.e. "no patch
            # recorded", which is indistinguishable from a producer that never
            # made one -- and that is the correct reading for old lines.
            patch_path=d.get("patch_path"),
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
