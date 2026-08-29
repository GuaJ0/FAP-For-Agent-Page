"""Compact Research memory derived from authoritative ``RunRecord`` history."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from agent.records import Decision, RunRecord, Status
from agent.research.context import build_history_fingerprint
from agent.research.retrieval.models import (
    QueryPlan,
    ResearchIntent,
    RetrievalValidationError,
    deterministic_fingerprint,
)
from agent.research.retrieval.safety import ResearchSafetyScanner


MEMORY_SCHEMA_VERSION = 2
_SHA256 = re.compile(r"[0-9a-f]{64}")


class StaleResearchMemoryError(RetrievalValidationError):
    """Memory was not reconciled against the exact supplied RunRecord history."""


class GapStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    SUPERSEDED = "superseded"


_METHOD_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pairwise_loss", ("pairwise", "bpr", "positive-negative pair")),
    ("listwise_loss", ("listwise", "lambdarank", "listnet", "top-k loss")),
    ("pointwise_loss", ("pointwise", "logloss", "binary cross entropy")),
    ("sequence_modeling", ("sequence", "recent history", "behavior history", "din", "attention")),
    ("feature_interaction", ("factorization machine", "deepfm", "feature interaction", "cross network")),
    ("multi_task", ("multi-task", "multitask", "auxiliary task", "auxiliary head")),
    ("watch_time", ("watch time", "watch-time", "play_time", "duration-aware")),
    ("debiasing", ("debias", "propensity", "inverse propensity", "exposure bias", "ips")),
    ("sampling", ("sampler", "sampling", "negative sample", "hard negative")),
    ("calibration", ("calibration", "calibrated", "isotonic")),
    ("regularization", ("regularization", "dropout", "weight decay", "early stopping")),
    ("temporal", ("temporal", "time-aware", "recency", "drift")),
    ("ensemble", ("ensemble", "blending", "stacking")),
    ("context_features", ("context feature", "exposure context", "tab feature")),
    ("content_features", ("content feature", "video metadata", "tag embedding")),
)


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9@._:+-]+", text.casefold()))


def _section(text: str, heading: str) -> Optional[str]:
    match = re.search(
        rf"(?ms)^{re.escape(heading)}:\s*\n(.*?)(?=\n[A-Z][A-Z ]+:\s*\n|\Z)",
        text,
    )
    return match.group(1).strip() if match else None


def _proposal_method_text(text: str) -> str:
    if "[RESEARCH_PROPOSAL" not in text:
        return text
    return "\n".join(
        section
        for heading in ("HYPOTHESIS", "IMPLEMENTATION", "TARGET COMPONENTS", "HYPERPARAMETERS")
        if (section := _section(text, heading))
    )


def _hypothesis_id(text: str) -> Optional[str]:
    match = re.search(r"(?m)^ID:\s*([^\r\n]+?)\s*$", text)
    return match.group(1) if match else None


def _evidence_source_ids(text: str) -> tuple[str, ...]:
    result = []
    for reference in re.findall(r"(?m)^-\s+\[([^\]]+)\]", text):
        source_id, separator, _ = reference.rpartition("/")
        if separator and source_id:
            result.append(source_id)
    return tuple(dict.fromkeys(result))


def classify_method_families(text: str) -> tuple[str, ...]:
    """Classify only the proposed mechanism, excluding evidence/risks/controls."""
    normalized = _normalized(_proposal_method_text(text))
    families = [
        family
        for family, keywords in _METHOD_KEYWORDS
        if any(_normalized(keyword) in normalized for keyword in keywords)
    ]
    return tuple(families or ["unknown"])


def method_fingerprint(hypothesis: str) -> str:
    relevant = {
        "hypothesis": _normalized(_section(hypothesis, "HYPOTHESIS") or hypothesis),
        "implementation": _normalized(_section(hypothesis, "IMPLEMENTATION") or ""),
        "target_components": _normalized(_section(hypothesis, "TARGET COMPONENTS") or ""),
        "hyperparameters": _normalized(_section(hypothesis, "HYPERPARAMETERS") or ""),
    }
    return deterministic_fingerprint(relevant)


@dataclass(frozen=True)
class ExperimentMemory:
    iteration: int
    parent_iteration: Optional[int]
    lineage_root_iteration: int
    hypothesis_id: Optional[str]
    method_fingerprint: str
    method_families: tuple[str, ...]
    status: str
    decision: Optional[str]
    primary_mean: Optional[float]
    gauc_mean: Optional[float]
    ndcg5_mean: Optional[float]
    gauc_delta_from_parent: Optional[float]
    ndcg5_delta_from_parent: Optional[float]
    evaluator_diagnostics: tuple[str, ...]
    evidence_source_ids: tuple[str, ...]

    @property
    def attempted(self) -> bool:
        return self.decision in {"accept", "revert", "abandon"} or self.status == "abandoned"


@dataclass(frozen=True)
class ResearchGap:
    gap_id: str
    kind: str
    statement: str
    source_iteration: int
    lineage_root_iteration: int
    method_families: tuple[str, ...]
    status: GapStatus = GapStatus.OPEN
    resolved_by_iteration: Optional[int] = None


@dataclass(frozen=True)
class QueryHistoryEntry:
    context_fingerprint: str
    query_signature: str
    query_id: str
    intent: str
    text: str


def _require_mapping(value: Any, *, path: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RetrievalValidationError(f"{path} keys must be exactly {sorted(keys)!r}")
    return value


def _require_str(value: Any, *, path: str, optional: bool = False) -> Optional[str]:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RetrievalValidationError(f"{path} must be a non-empty string")
    return value


def _require_sha(value: Any, *, path: str, optional: bool = False) -> Optional[str]:
    result = _require_str(value, path=path, optional=optional)
    if result is not None and _SHA256.fullmatch(result) is None:
        raise RetrievalValidationError(f"{path} must be a SHA-256 digest")
    return result


def _require_int(value: Any, *, path: str, optional: bool = False) -> Optional[int]:
    if value is None and optional:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RetrievalValidationError(f"{path} must be a non-negative integer")
    return value


def _require_number(value: Any, *, path: str) -> Optional[float]:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise RetrievalValidationError(f"{path} must be finite or null")
    return float(value)


def _require_strings(value: Any, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise RetrievalValidationError(f"{path} must be an array of non-empty strings")
    if len(value) != len(set(value)):
        raise RetrievalValidationError(f"{path} must not contain duplicates")
    return tuple(value)


@dataclass
class ResearchMemory:
    """Persistable enrichment index; RunRecord remains the outcome authority."""

    path: Optional[Path] = None
    authoritative_history_fingerprint: Optional[str] = None
    authoritative_run_fingerprint: Optional[str] = None
    experiments: dict[int, ExperimentMemory] = field(default_factory=dict)
    attempted_method_fingerprints: set[str] = field(default_factory=set)
    attempted_method_families: set[str] = field(default_factory=set)
    evidence_source_ids: set[str] = field(default_factory=set)
    unresolved_gaps: dict[str, ResearchGap] = field(default_factory=dict)
    query_history: dict[str, QueryHistoryEntry] = field(default_factory=dict)
    scanner: ResearchSafetyScanner = field(default_factory=ResearchSafetyScanner, repr=False)
    _reconciled: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path = Path(self.path)

    @property
    def active_gaps(self) -> tuple[ResearchGap, ...]:
        return tuple(
            gap for _, gap in sorted(self.unresolved_gaps.items())
            if gap.status == GapStatus.OPEN
        )

    def ancestor_chain(self, iteration: int) -> tuple[int, ...]:
        """Return ``iteration`` followed by its actual parent chain.

        This deliberately does not use the shared root as a proxy for
        ancestry: sibling branches share a root but are not ancestors of one
        another.
        """
        if iteration not in self.experiments:
            raise StaleResearchMemoryError(
                f"ResearchMemory does not contain iteration {iteration}"
            )
        chain: list[int] = []
        seen: set[int] = set()
        current: Optional[int] = iteration
        while current is not None:
            if current in seen:
                raise RetrievalValidationError("Research memory contains a parent cycle")
            experiment = self.experiments.get(current)
            if experiment is None:
                break
            seen.add(current)
            chain.append(current)
            current = experiment.parent_iteration
        return tuple(chain)

    def _is_ancestor(self, ancestor_iteration: int, descendant_iteration: int) -> bool:
        return ancestor_iteration in self.ancestor_chain(descendant_iteration)

    def is_branch_comparable(self, iteration: int, incumbent_iteration: int) -> bool:
        """Whether either iteration lies on the other's actual parent chain."""
        return (
            self._is_ancestor(iteration, incumbent_iteration)
            or self._is_ancestor(incumbent_iteration, iteration)
        )

    def active_gaps_for_incumbent(self, incumbent_iteration: Optional[int]) -> tuple[ResearchGap, ...]:
        """Return open gaps on branches comparable with the incumbent.

        Ancestor gaps and gaps produced by direct descendants of the current
        incumbent are relevant. Sibling-branch gaps are not.
        """
        if incumbent_iteration is None:
            return ()
        incumbent = self.experiments.get(incumbent_iteration)
        if incumbent is None:
            raise StaleResearchMemoryError(
                "ResearchMemory does not contain the ResearchContext incumbent iteration"
            )
        return tuple(
            gap for gap in self.active_gaps
            if self.is_branch_comparable(gap.source_iteration, incumbent_iteration)
        )

    def assert_matches(self, history_fingerprint: str) -> None:
        if not self._reconciled:
            raise StaleResearchMemoryError(
                "ResearchMemory is stale or loaded but unreconciled; call reconcile(history) before plan()"
            )
        if self.authoritative_history_fingerprint != history_fingerprint:
            raise StaleResearchMemoryError(
                "ResearchMemory does not match the authoritative ResearchContext history"
            )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        scanner: Optional[ResearchSafetyScanner] = None,
    ) -> "ResearchMemory":
        memory_path = Path(path)
        safety = scanner or ResearchSafetyScanner()
        if not memory_path.exists():
            return cls(path=memory_path, scanner=safety)
        try:
            raw = json.loads(memory_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RetrievalValidationError("Research memory is not valid UTF-8 JSON") from exc
        safety.scan_value(raw, origin="Research memory")
        top_keys = {
            "schema_version", "authoritative_history_fingerprint",
            "authoritative_run_fingerprint", "experiments",
            "attempted_method_fingerprints", "attempted_method_families",
            "evidence_source_ids", "unresolved_gaps", "query_history",
        }
        root = _require_mapping(raw, path="Research memory", keys=top_keys)
        if root["schema_version"] != MEMORY_SCHEMA_VERSION:
            raise RetrievalValidationError("unsupported Research memory schema_version")
        memory = cls(path=memory_path, scanner=safety)
        memory.authoritative_history_fingerprint = _require_sha(
            root["authoritative_history_fingerprint"],
            path="authoritative_history_fingerprint",
            optional=True,
        )
        memory.authoritative_run_fingerprint = _require_sha(
            root["authoritative_run_fingerprint"],
            path="authoritative_run_fingerprint",
            optional=True,
        )

        if not isinstance(root["experiments"], list):
            raise RetrievalValidationError("experiments must be an array")
        experiment_keys = {
            "iteration", "parent_iteration", "lineage_root_iteration", "hypothesis_id",
            "method_fingerprint", "method_families", "status", "decision", "primary_mean",
            "gauc_mean", "ndcg5_mean", "gauc_delta_from_parent", "ndcg5_delta_from_parent",
            "evaluator_diagnostics", "evidence_source_ids",
        }
        for index, raw_item in enumerate(root["experiments"]):
            item = _require_mapping(raw_item, path=f"experiments[{index}]", keys=experiment_keys)
            iteration = _require_int(item["iteration"], path=f"experiments[{index}].iteration")
            if iteration in memory.experiments:
                raise RetrievalValidationError("Research memory contains duplicate experiment iterations")
            status = _require_str(item["status"], path=f"experiments[{index}].status")
            decision = _require_str(item["decision"], path=f"experiments[{index}].decision", optional=True)
            if status not in {member.value for member in Status}:
                raise RetrievalValidationError(f"experiments[{index}].status is invalid")
            if decision not in {None, *(member.value for member in Decision)}:
                raise RetrievalValidationError(f"experiments[{index}].decision is invalid")
            memory.experiments[iteration] = ExperimentMemory(
                iteration=iteration,
                parent_iteration=_require_int(item["parent_iteration"], path="parent_iteration", optional=True),
                lineage_root_iteration=_require_int(item["lineage_root_iteration"], path="lineage_root_iteration"),
                hypothesis_id=_require_str(item["hypothesis_id"], path="hypothesis_id", optional=True),
                method_fingerprint=_require_sha(item["method_fingerprint"], path="method_fingerprint"),
                method_families=_require_strings(item["method_families"], path="method_families"),
                status=status,
                decision=decision,
                primary_mean=_require_number(item["primary_mean"], path="primary_mean"),
                gauc_mean=_require_number(item["gauc_mean"], path="gauc_mean"),
                ndcg5_mean=_require_number(item["ndcg5_mean"], path="ndcg5_mean"),
                gauc_delta_from_parent=_require_number(item["gauc_delta_from_parent"], path="gauc_delta_from_parent"),
                ndcg5_delta_from_parent=_require_number(item["ndcg5_delta_from_parent"], path="ndcg5_delta_from_parent"),
                evaluator_diagnostics=_require_strings(item["evaluator_diagnostics"], path="evaluator_diagnostics"),
                evidence_source_ids=_require_strings(item["evidence_source_ids"], path="experiment evidence_source_ids"),
            )

        memory.attempted_method_fingerprints = set(_require_strings(
            root["attempted_method_fingerprints"], path="attempted_method_fingerprints"
        ))
        if any(_SHA256.fullmatch(value) is None for value in memory.attempted_method_fingerprints):
            raise RetrievalValidationError("attempted_method_fingerprints contains an invalid digest")
        memory.attempted_method_families = set(_require_strings(
            root["attempted_method_families"], path="attempted_method_families"
        ))
        memory.evidence_source_ids = set(_require_strings(
            root["evidence_source_ids"], path="evidence_source_ids"
        ))

        if not isinstance(root["unresolved_gaps"], list):
            raise RetrievalValidationError("unresolved_gaps must be an array")
        gap_keys = {
            "gap_id", "kind", "statement", "source_iteration", "lineage_root_iteration",
            "method_families", "status", "resolved_by_iteration",
        }
        for index, raw_item in enumerate(root["unresolved_gaps"]):
            item = _require_mapping(raw_item, path=f"unresolved_gaps[{index}]", keys=gap_keys)
            gap_id = _require_str(item["gap_id"], path="gap_id")
            if gap_id in memory.unresolved_gaps:
                raise RetrievalValidationError("Research memory contains duplicate gap IDs")
            try:
                gap_status = GapStatus(item["status"])
            except (TypeError, ValueError) as exc:
                raise RetrievalValidationError("Research memory contains an invalid gap status") from exc
            resolved_by = _require_int(item["resolved_by_iteration"], path="resolved_by_iteration", optional=True)
            if gap_status == GapStatus.OPEN and resolved_by is not None:
                raise RetrievalValidationError("open Research gaps cannot have resolved_by_iteration")
            if gap_status != GapStatus.OPEN and resolved_by is None:
                raise RetrievalValidationError("closed Research gaps require resolved_by_iteration")
            memory.unresolved_gaps[gap_id] = ResearchGap(
                gap_id=gap_id,
                kind=_require_str(item["kind"], path="gap.kind"),
                statement=_require_str(item["statement"], path="gap.statement"),
                source_iteration=_require_int(item["source_iteration"], path="gap.source_iteration"),
                lineage_root_iteration=_require_int(item["lineage_root_iteration"], path="gap.lineage_root_iteration"),
                method_families=_require_strings(item["method_families"], path="gap.method_families"),
                status=gap_status,
                resolved_by_iteration=resolved_by,
            )

        if not isinstance(root["query_history"], list):
            raise RetrievalValidationError("query_history must be an array")
        query_keys = {"context_fingerprint", "query_signature", "query_id", "intent", "text"}
        for index, raw_item in enumerate(root["query_history"]):
            item = _require_mapping(raw_item, path=f"query_history[{index}]", keys=query_keys)
            entry = QueryHistoryEntry(
                context_fingerprint=_require_sha(item["context_fingerprint"], path="query context_fingerprint"),
                query_signature=_require_sha(item["query_signature"], path="query_signature"),
                query_id=_require_str(item["query_id"], path="query_id"),
                intent=_require_str(item["intent"], path="query intent"),
                text=_require_str(item["text"], path="query text"),
            )
            if entry.intent not in {intent.value for intent in ResearchIntent}:
                raise RetrievalValidationError("Research memory contains an invalid query intent")
            key = cls._query_history_key(entry.context_fingerprint, entry.query_signature)
            if key in memory.query_history:
                raise RetrievalValidationError("Research memory contains duplicate query history")
            memory.query_history[key] = entry
        # Loaded state is intentionally not trusted as current until reconcile().
        return memory

    def reconcile(self, history: Sequence[RunRecord]) -> None:
        """Rebuild outcome state and bind it to the authoritative history."""
        records = sorted(history, key=lambda record: record.iteration)
        if len({record.iteration for record in records}) != len(records):
            raise RetrievalValidationError("authoritative history contains duplicate iterations")
        history_fingerprint = build_history_fingerprint(records)
        run_fingerprint = build_history_fingerprint(records[:1])
        same_run = (
            self.authoritative_run_fingerprint is None
            or self.authoritative_run_fingerprint == run_fingerprint
        )
        previous_history_sources = {
            source_id
            for experiment in self.experiments.values()
            for source_id in experiment.evidence_source_ids
        }
        external_evidence = (
            self.evidence_source_ids - previous_history_sources if same_run else set()
        )
        if not same_run:
            self.query_history.clear()

        by_iteration = {record.iteration: record for record in records}
        experiments: dict[int, ExperimentMemory] = {}
        attempted_fingerprints: set[str] = set()
        attempted_families: set[str] = set()
        history_evidence: set[str] = set()

        def lineage_root(record: RunRecord) -> int:
            current = record
            seen: set[int] = set()
            while current.parent_iteration is not None and current.parent_iteration in by_iteration:
                if current.iteration in seen:
                    raise RetrievalValidationError("authoritative history contains a parent cycle")
                seen.add(current.iteration)
                current = by_iteration[current.parent_iteration]
            return current.iteration

        for record in records:
            self.scanner.scan_text(record.hypothesis, origin=f"Research history iteration {record.iteration} hypothesis")
            diagnostics = tuple(
                event.detail for event in record.events if event.agent_action == "evaluator"
            )
            self.scanner.scan_value(diagnostics, origin=f"Research history iteration {record.iteration} evaluator diagnostics")
            aggregate = record.aggregate
            parent = by_iteration.get(record.parent_iteration)
            parent_aggregate = parent.aggregate if parent is not None else None
            gauc_delta = aggregate.gauc_mean - parent_aggregate.gauc_mean if aggregate and parent_aggregate else None
            ndcg_delta = aggregate.ndcg5_mean - parent_aggregate.ndcg5_mean if aggregate and parent_aggregate else None
            families = classify_method_families(record.hypothesis)
            fingerprint = method_fingerprint(record.hypothesis)
            sources = _evidence_source_ids(record.hypothesis)
            experiment = ExperimentMemory(
                iteration=record.iteration,
                parent_iteration=record.parent_iteration,
                lineage_root_iteration=lineage_root(record),
                hypothesis_id=_hypothesis_id(record.hypothesis),
                method_fingerprint=fingerprint,
                method_families=families,
                status=record.status.value,
                decision=record.decision.value if record.decision else None,
                primary_mean=aggregate.primary_mean if aggregate else None,
                gauc_mean=aggregate.gauc_mean if aggregate else None,
                ndcg5_mean=aggregate.ndcg5_mean if aggregate else None,
                gauc_delta_from_parent=gauc_delta,
                ndcg5_delta_from_parent=ndcg_delta,
                evaluator_diagnostics=diagnostics,
                evidence_source_ids=sources,
            )
            experiments[record.iteration] = experiment
            history_evidence.update(sources)
            if experiment.attempted:
                attempted_fingerprints.add(fingerprint)
                attempted_families.update(families)

        gaps: dict[str, ResearchGap] = {}
        for experiment in experiments.values():
            if not experiment.attempted:
                continue
            for gap in (self._gap_for_experiment(experiment), self._divergence_gap(experiment)):
                if gap is not None:
                    gaps[gap.gap_id] = gap
        gaps = self._apply_gap_lifecycle(gaps, experiments)

        self.experiments = experiments
        self.attempted_method_fingerprints = attempted_fingerprints
        self.attempted_method_families = attempted_families
        self.evidence_source_ids = external_evidence | history_evidence
        self.unresolved_gaps = gaps
        self.authoritative_history_fingerprint = history_fingerprint
        self.authoritative_run_fingerprint = run_fingerprint
        self._reconciled = True
        self.save()

    @staticmethod
    def _gap_for_experiment(experiment: ExperimentMemory) -> Optional[ResearchGap]:
        if experiment.decision == Decision.ACCEPT.value:
            kind = "accepted_follow_up"
            statement = "Find a compatible, attributable follow-up to the accepted method."
        elif experiment.decision == Decision.REVERT.value:
            kind = "reverted_alternative"
            statement = "Find a different mechanism that addresses the reverted method's goal."
        elif experiment.decision == Decision.ABANDON.value or experiment.status == Status.ABANDONED.value:
            kind = "abandoned_feasibility"
            statement = "Find a cheaper or simpler method for the abandoned research direction."
        else:
            return None
        gap_id = deterministic_fingerprint({
            "kind": kind, "iteration": experiment.iteration, "families": experiment.method_families,
        })[:16]
        return ResearchGap(
            gap_id=f"G-{gap_id}", kind=kind, statement=statement,
            source_iteration=experiment.iteration,
            lineage_root_iteration=experiment.lineage_root_iteration,
            method_families=experiment.method_families,
        )

    @staticmethod
    def _divergence_gap(experiment: ExperimentMemory) -> Optional[ResearchGap]:
        gauc_delta = experiment.gauc_delta_from_parent
        ndcg_delta = experiment.ndcg5_delta_from_parent
        if gauc_delta is None or ndcg_delta is None:
            return None
        if gauc_delta > 0 and ndcg_delta < 0:
            kind = "top_list_quality"
            statement = "Preserve the GAUC gain while recovering top-of-list nDCG@5 quality."
            families = ("listwise_loss", "top_k_reranking")
        elif ndcg_delta > 0 and gauc_delta < 0:
            kind = "within_user_consistency"
            statement = "Preserve the nDCG@5 gain while recovering broad within-user GAUC."
            families = ("sampling", "calibration")
        else:
            return None
        gap_id = deterministic_fingerprint({"kind": kind, "iteration": experiment.iteration})[:16]
        return ResearchGap(
            gap_id=f"G-{gap_id}", kind=kind, statement=statement,
            source_iteration=experiment.iteration,
            lineage_root_iteration=experiment.lineage_root_iteration,
            method_families=families,
        )

    @staticmethod
    def _apply_gap_lifecycle(
        gaps: dict[str, ResearchGap],
        experiments: Mapping[int, ExperimentMemory],
    ) -> dict[str, ResearchGap]:
        ordered = sorted(experiments.values(), key=lambda item: item.iteration)
        result = dict(gaps)
        divergence_kinds = {"top_list_quality", "within_user_consistency"}

        def is_descendant(source_iteration: int, candidate_iteration: int) -> bool:
            current = experiments.get(candidate_iteration)
            seen: set[int] = set()
            while current is not None:
                if current.iteration in seen:
                    raise RetrievalValidationError("Research memory contains a parent cycle")
                seen.add(current.iteration)
                if current.iteration == source_iteration:
                    return True
                current = experiments.get(current.parent_iteration)
            return False

        for gap_id, gap in gaps.items():
            later = [
                experiment for experiment in ordered
                if experiment.iteration > gap.source_iteration
                and is_descendant(gap.source_iteration, experiment.iteration)
            ]
            for experiment in later:
                if gap.kind == "accepted_follow_up" and experiment.decision == Decision.ACCEPT.value:
                    result[gap_id] = replace(
                        gap, status=GapStatus.SUPERSEDED,
                        resolved_by_iteration=experiment.iteration,
                    )
                    break
                if gap.kind in divergence_kinds:
                    later_gap = ResearchMemory._divergence_gap(experiment)
                    if later_gap is not None and later_gap.kind == gap.kind:
                        result[gap_id] = replace(
                            gap, status=GapStatus.SUPERSEDED,
                            resolved_by_iteration=experiment.iteration,
                        )
                        break
                    if (
                        experiment.decision == Decision.ACCEPT.value
                        and experiment.gauc_delta_from_parent is not None
                        and experiment.ndcg5_delta_from_parent is not None
                        and experiment.gauc_delta_from_parent > 0
                        and experiment.ndcg5_delta_from_parent > 0
                    ):
                        result[gap_id] = replace(
                            gap, status=GapStatus.RESOLVED,
                            resolved_by_iteration=experiment.iteration,
                        )
                        break
        return result

    def remember_evidence(self, source_ids: Iterable[str]) -> None:
        values = tuple(source_id.strip() for source_id in source_ids if source_id.strip())
        self.scanner.scan_value(values, origin="Research evidence source IDs")
        self.evidence_source_ids.update(values)
        self.save()

    @staticmethod
    def _query_history_key(context_fingerprint: str, query_signature: str) -> str:
        return deterministic_fingerprint({
            "context_fingerprint": context_fingerprint,
            "query_signature": query_signature,
        })

    def has_query(self, context_fingerprint: str, query_signature: str) -> bool:
        return self._query_history_key(context_fingerprint, query_signature) in self.query_history

    def remember_query_plan(self, plan: QueryPlan) -> None:
        if not self._reconciled:
            raise StaleResearchMemoryError("reconcile(history) is required before recording a query plan")
        for query in plan.queries:
            entry = QueryHistoryEntry(
                context_fingerprint=plan.context_fingerprint,
                query_signature=query.signature,
                query_id=query.query_id,
                intent=query.intent.value,
                text=query.text,
            )
            self.query_history[self._query_history_key(plan.context_fingerprint, query.signature)] = entry
        self.save()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "authoritative_history_fingerprint": self.authoritative_history_fingerprint,
            "authoritative_run_fingerprint": self.authoritative_run_fingerprint,
            "experiments": [asdict(self.experiments[iteration]) for iteration in sorted(self.experiments)],
            "attempted_method_fingerprints": sorted(self.attempted_method_fingerprints),
            "attempted_method_families": sorted(self.attempted_method_families),
            "evidence_source_ids": sorted(self.evidence_source_ids),
            "unresolved_gaps": [asdict(self.unresolved_gaps[gap_id]) for gap_id in sorted(self.unresolved_gaps)],
            "query_history": [asdict(self.query_history[key]) for key in sorted(self.query_history)],
        }

    def save(self) -> None:
        if self.path is None:
            return
        payload = self.to_dict()
        self.scanner.scan_value(payload, origin="Research memory")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
