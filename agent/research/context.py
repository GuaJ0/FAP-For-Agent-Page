"""Build the validation-only context used to choose the next hypothesis."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from agent.config import BOOTSTRAP_ITERATION, ConvergenceConfig, DEFAULT_CONFIG
from agent.executor import assert_no_forbidden_keys
from agent.records import Decision, RunRecord, Status


@dataclass(frozen=True)
class IncumbentSummary:
    iteration: int
    hypothesis: str
    primary_mean: float
    primary_std: float
    gauc_mean: float
    ndcg5_mean: float
    n_seeds: int
    diff_path: Optional[str]


@dataclass(frozen=True)
class IterationSummary:
    iteration: int
    parent_iteration: Optional[int]
    hypothesis: str
    status: str
    decision: Optional[str]
    primary_mean: Optional[float]
    primary_std: Optional[float]
    gauc_mean: Optional[float]
    ndcg5_mean: Optional[float]
    delta_vs_current_best: Optional[float]
    n_seeds: int
    failure_kinds: tuple[str, ...]
    evaluator_events: tuple[str, ...]
    wall_s: float


@dataclass(frozen=True)
class ResearchContext:
    incumbent: Optional[IncumbentSummary]
    parent_iteration: Optional[int]
    iterations: tuple[IterationSummary, ...]
    remaining_iterations: int
    remaining_wall_s: float
    minimum_meaningful_delta: float
    history_fingerprint: str
    task: str = "KuaiRand-Pure within-user ranking of logged impressions"
    label: str = "long_view"
    primary_metric: str = "mean(GAUC, nDCG@5)"

    def to_prompt_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        assert_no_forbidden_keys(payload)
        return payload


def build_history_fingerprint(history: Sequence[RunRecord]) -> str:
    """Fingerprint the exact authoritative RunRecord sequence.

    ResearchMemory uses the same digest to prove that its enrichment index was
    reconciled against the context supplied to QueryPlanner.
    """
    payload = json.dumps(
        [record.to_json() for record in history],
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result


def _summarize(record: RunRecord) -> IterationSummary:
    aggregate = record.aggregate
    return IterationSummary(
        iteration=record.iteration,
        parent_iteration=record.parent_iteration,
        hypothesis=record.hypothesis,
        status=record.status.value,
        decision=record.decision.value if record.decision else None,
        primary_mean=aggregate.primary_mean if aggregate else None,
        primary_std=aggregate.primary_std if aggregate else None,
        gauc_mean=aggregate.gauc_mean if aggregate else None,
        ndcg5_mean=aggregate.ndcg5_mean if aggregate else None,
        delta_vs_current_best=record.delta_vs_current_best,
        n_seeds=aggregate.n_seeds if aggregate else 0,
        failure_kinds=tuple(
            sorted({seed.failure_kind.value for seed in record.seeds if seed.failure_kind is not None})
        ),
        evaluator_events=tuple(
            event.detail for event in record.events if event.agent_action == "evaluator"
        ),
        wall_s=record.resources.wall_s,
    )


def _incumbent(history: Sequence[RunRecord]) -> Optional[IncumbentSummary]:
    accepted = [
        record for record in history
        if record.aggregate is not None and record.decision == Decision.ACCEPT
    ]
    if not accepted:
        return None
    # Mirrors CheckpointRegistry: the accepted record with the highest
    # validation primary is the incumbent, regardless of what ran most recently.
    record = max(accepted, key=lambda item: item.aggregate.primary_mean)
    aggregate = record.aggregate
    return IncumbentSummary(
        iteration=record.iteration,
        hypothesis=record.hypothesis,
        primary_mean=aggregate.primary_mean,
        primary_std=aggregate.primary_std,
        gauc_mean=aggregate.gauc_mean,
        ndcg5_mean=aggregate.ndcg5_mean,
        n_seeds=aggregate.n_seeds,
        diff_path=record.diff_path,
    )


def build_research_context(
    history: Sequence[RunRecord],
    cfg: ConvergenceConfig = DEFAULT_CONFIG.convergence,
) -> ResearchContext:
    """Return only data already permitted in the agent-facing RunRecord.

    It intentionally does not read ``docs/results.md``, ``solution/ideas.md``,
    or the quarantine directory: those files contain or may contain
    split-specific numbers that are not part of the Research Agent boundary.
    """
    records = list(history)
    incumbent = _incumbent(records)
    # Match convergence.should_stop() exactly: concluded research experiments
    # consume max_iterations, but the bootstrap incumbent at iteration 0 does
    # not. Keep the bootstrap in ``records`` for metrics, incumbent selection,
    # prompt history, wall-clock accounting, and the authoritative fingerprint.
    concluded_research = sum(
        record.status != Status.FAILED
        and record.iteration != BOOTSTRAP_ITERATION
        for record in records
    )
    remaining_iterations = max(0, cfg.max_iterations - concluded_research)
    elapsed = 0.0
    if len(records) >= 2:
        elapsed = max(
            0.0,
            (_parse_timestamp(records[-1].timestamp) - _parse_timestamp(records[0].timestamp)).total_seconds(),
        )
    remaining_wall_s = max(0.0, cfg.max_wall_s - elapsed)
    context = ResearchContext(
        incumbent=incumbent,
        parent_iteration=incumbent.iteration if incumbent else None,
        iterations=tuple(_summarize(record) for record in records),
        remaining_iterations=remaining_iterations,
        remaining_wall_s=remaining_wall_s,
        minimum_meaningful_delta=cfg.epsilon,
        history_fingerprint=build_history_fingerprint(records),
    )
    # Structural backstop: this is the exact JSON-compatible payload that will
    # later be placed in the Research prompt.
    context.to_prompt_dict()
    return context
