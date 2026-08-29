"""Research-owned models for bounded, deterministic literature retrieval.

These types contain no transport or provider behavior.  They are the offline
contract used by query planning now and by provider adapters in a later phase.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class RetrievalValidationError(ValueError):
    """A retrieval model or operation exceeds its safe deterministic contract."""


class ResearchIntent(str, Enum):
    DATASET_SPECIFIC = "dataset_specific"
    OBJECTIVE_ALIGNMENT = "objective_alignment"
    GAUC_IMPROVEMENT = "gauc_improvement"
    NDCG_IMPROVEMENT = "ndcg_improvement"
    INCUMBENT_EXTENSION = "incumbent_extension"
    FAILED_ALTERNATIVE = "failed_alternative"
    SUCCESS_FOLLOW_UP = "success_follow_up"
    ABANDONED_FEASIBILITY = "abandoned_feasibility"
    UNUSED_SIGNAL = "unused_signal"
    EVALUATOR_DIAGNOSTIC = "evaluator_diagnostic"
    EXPLORATION = "exploration"


def _non_empty_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalValidationError(f"{path} must be a non-empty string")
    return value.strip()


def _text_tuple(value: Any, path: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RetrievalValidationError(f"{path} must be an array of strings")
    result = tuple(_non_empty_text(item, f"{path}[{index}]") for index, item in enumerate(value))
    if not result and not allow_empty:
        raise RetrievalValidationError(f"{path} must not be empty")
    if len(set(result)) != len(result):
        raise RetrievalValidationError(f"{path} must not contain duplicates")
    return result


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise RetrievalValidationError("fingerprint input contains NaN or Infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise RetrievalValidationError(
        f"fingerprint input contains unsupported value {type(value).__name__}"
    )


def deterministic_fingerprint(value: Any) -> str:
    """Return a stable SHA-256 digest of a JSON-compatible semantic value."""
    payload = json.dumps(
        _canonical(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DatasetSignal:
    family: str
    method_family: str
    fields: tuple[str, ...]
    description: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetSignal":
        expected = {"family", "method_family", "fields", "description"}
        if set(value) != expected:
            raise RetrievalValidationError(
                "dataset signal keys must be exactly " + repr(sorted(expected))
            )
        return cls(
            family=_non_empty_text(value["family"], "signal.family"),
            method_family=_non_empty_text(value["method_family"], "signal.method_family"),
            fields=_text_tuple(value["fields"], "signal.fields"),
            description=_non_empty_text(value["description"], "signal.description"),
        )


@dataclass(frozen=True)
class DatasetProfile:
    schema_version: int
    profile_id: str
    dataset_name: str
    task: str
    label: str
    metrics: tuple[str, ...]
    interaction_semantics: str
    allowed_data_boundary: tuple[str, ...]
    available_signals: tuple[DatasetSignal, ...]
    assumptions: tuple[str, ...]
    public_sources: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetProfile":
        expected = {
            "schema_version",
            "profile_id",
            "dataset_name",
            "task",
            "label",
            "metrics",
            "interaction_semantics",
            "allowed_data_boundary",
            "available_signals",
            "assumptions",
            "public_sources",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise RetrievalValidationError(
                "dataset profile keys must be exactly " + repr(sorted(expected))
            )
        if value["schema_version"] != 1:
            raise RetrievalValidationError("dataset profile schema_version must be 1")
        raw_signals = value["available_signals"]
        if not isinstance(raw_signals, Sequence) or isinstance(raw_signals, (str, bytes)):
            raise RetrievalValidationError("available_signals must be an array")
        signals = tuple(DatasetSignal.from_dict(item) for item in raw_signals)
        if not signals:
            raise RetrievalValidationError("available_signals must not be empty")
        families = [signal.family for signal in signals]
        if len(families) != len(set(families)):
            raise RetrievalValidationError("available signal families must be unique")
        metrics = _text_tuple(value["metrics"], "profile.metrics")
        if set(metrics) != {"GAUC", "nDCG@5"}:
            raise RetrievalValidationError("profile.metrics must contain GAUC and nDCG@5")
        boundary = _text_tuple(value["allowed_data_boundary"], "profile.allowed_data_boundary")
        if set(boundary) - {"public_metadata", "train", "validation"}:
            raise RetrievalValidationError(
                "allowed_data_boundary may contain only public_metadata, train, and validation"
            )
        return cls(
            schema_version=1,
            profile_id=_non_empty_text(value["profile_id"], "profile.profile_id"),
            dataset_name=_non_empty_text(value["dataset_name"], "profile.dataset_name"),
            task=_non_empty_text(value["task"], "profile.task"),
            label=_non_empty_text(value["label"], "profile.label"),
            metrics=metrics,
            interaction_semantics=_non_empty_text(
                value["interaction_semantics"], "profile.interaction_semantics"
            ),
            allowed_data_boundary=boundary,
            available_signals=signals,
            assumptions=_text_tuple(value["assumptions"], "profile.assumptions"),
            public_sources=_text_tuple(value["public_sources"], "profile.public_sources"),
        )

    @property
    def fingerprint(self) -> str:
        return deterministic_fingerprint(self)


@dataclass(frozen=True)
class ResearchQuery:
    query_id: str
    intent: ResearchIntent
    text: str
    rationale: str
    method_families: tuple[str, ...]
    priority: int
    max_results: int
    cheap_only: bool = False

    def __post_init__(self) -> None:
        _non_empty_text(self.query_id, "query.query_id")
        _non_empty_text(self.text, "query.text")
        _non_empty_text(self.rationale, "query.rationale")
        _text_tuple(self.method_families, "query.method_families")
        if not isinstance(self.intent, ResearchIntent):
            raise RetrievalValidationError("query.intent must be a ResearchIntent")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise RetrievalValidationError("query.priority must be an integer")
        if not isinstance(self.max_results, int) or isinstance(self.max_results, bool):
            raise RetrievalValidationError("query.max_results must be an integer")
        if self.max_results < 1:
            raise RetrievalValidationError("query.max_results must be positive")

    @classmethod
    def create(
        cls,
        *,
        intent: ResearchIntent,
        text: str,
        rationale: str,
        method_families: Sequence[str],
        priority: int,
        max_results: int,
        cheap_only: bool = False,
    ) -> "ResearchQuery":
        normalized = " ".join(text.split())
        signature = deterministic_fingerprint({
            "intent": intent.value,
            "text": normalized.casefold(),
            "method_families": sorted(set(method_families)),
            "cheap_only": cheap_only,
        })
        return cls(
            query_id=f"Q-{signature[:12]}",
            intent=intent,
            text=normalized,
            rationale=" ".join(rationale.split()),
            method_families=tuple(dict.fromkeys(method_families)),
            priority=priority,
            max_results=max_results,
            cheap_only=cheap_only,
        )

    @property
    def signature(self) -> str:
        return deterministic_fingerprint({
            "intent": self.intent.value,
            "text": " ".join(self.text.casefold().split()),
            "method_families": sorted(self.method_families),
            "cheap_only": self.cheap_only,
        })


@dataclass(frozen=True)
class QueryPlan:
    schema_version: int
    context_fingerprint: str
    queries: tuple[ResearchQuery, ...]
    tight_budget: bool
    suppressed_query_signatures: tuple[str, ...] = ()
    retrieval_wall_s: float = 0.0

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise RetrievalValidationError("query plan schema_version must be 1")
        if re.fullmatch(r"[0-9a-f]{64}", self.context_fingerprint) is None:
            raise RetrievalValidationError("query plan context_fingerprint must be a SHA-256 digest")
        signatures = [query.signature for query in self.queries]
        if len(signatures) != len(set(signatures)):
            raise RetrievalValidationError("query plan contains duplicate queries")
        if (
            not isinstance(self.retrieval_wall_s, (int, float))
            or isinstance(self.retrieval_wall_s, bool)
            or not math.isfinite(float(self.retrieval_wall_s))
            or self.retrieval_wall_s < 0
        ):
            raise RetrievalValidationError("query plan retrieval_wall_s must be finite and non-negative")


@dataclass(frozen=True)
class RetrievalBudget:
    """Hard caps applied before future provider, extraction, and prompt work."""

    max_queries: int = 10
    max_results_per_query: int = 8
    max_total_results: int = 40
    max_retrieval_wall_s: float = 20.0
    max_evidence_items: int = 8
    max_evidence_chars: int = 12_000
    max_prompt_chars: int = 24_000
    max_query_chars: int = 320

    def __post_init__(self) -> None:
        integer_limits = {
            "max_queries": self.max_queries,
            "max_results_per_query": self.max_results_per_query,
            "max_total_results": self.max_total_results,
            "max_evidence_items": self.max_evidence_items,
            "max_evidence_chars": self.max_evidence_chars,
            "max_prompt_chars": self.max_prompt_chars,
            "max_query_chars": self.max_query_chars,
        }
        for name, value in integer_limits.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise RetrievalValidationError(f"{name} must be a positive integer")
        if (
            not isinstance(self.max_retrieval_wall_s, (int, float))
            or isinstance(self.max_retrieval_wall_s, bool)
            or not math.isfinite(float(self.max_retrieval_wall_s))
            or self.max_retrieval_wall_s <= 0
        ):
            raise RetrievalValidationError("max_retrieval_wall_s must be finite and positive")
        if self.max_total_results < self.max_results_per_query:
            raise RetrievalValidationError(
                "max_total_results must be at least max_results_per_query"
            )
        if self.max_prompt_chars < self.max_evidence_chars:
            raise RetrievalValidationError(
                "max_prompt_chars must be at least max_evidence_chars"
            )

    def validate_plan(self, plan: QueryPlan) -> None:
        if len(plan.queries) > self.max_queries:
            raise RetrievalValidationError(
                f"query plan has {len(plan.queries)} queries; limit is {self.max_queries}"
            )
        total_results = 0
        for query in plan.queries:
            if len(query.text) > self.max_query_chars:
                raise RetrievalValidationError(
                    f"query {query.query_id} has {len(query.text)} characters; "
                    f"limit is {self.max_query_chars}"
                )
            if query.max_results > self.max_results_per_query:
                raise RetrievalValidationError(
                    f"query {query.query_id} requests {query.max_results} results; "
                    f"per-query limit is {self.max_results_per_query}"
                )
            total_results += query.max_results
        self.validate_result_count(total_results)
        if plan.retrieval_wall_s > self.max_retrieval_wall_s:
            raise RetrievalValidationError(
                f"query plan retrieval wall budget {plan.retrieval_wall_s:g}s exceeds limit "
                f"{self.max_retrieval_wall_s:g}s"
            )

    def validate_result_count(self, count: int) -> None:
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise RetrievalValidationError("result count must be a non-negative integer")
        if count > self.max_total_results:
            raise RetrievalValidationError(
                f"retrieval produced/requested {count} results; limit is {self.max_total_results}"
            )

    def validate_retrieval_time(self, elapsed_s: float) -> None:
        if not isinstance(elapsed_s, (int, float)) or not math.isfinite(float(elapsed_s)):
            raise RetrievalValidationError("retrieval elapsed time must be finite")
        if elapsed_s < 0 or elapsed_s > self.max_retrieval_wall_s:
            raise RetrievalValidationError(
                f"retrieval elapsed time {elapsed_s:g}s exceeds limit "
                f"{self.max_retrieval_wall_s:g}s"
            )

    def validate_evidence(self, count: int, serialized_chars: int) -> None:
        if count < 0 or serialized_chars < 0:
            raise RetrievalValidationError("evidence counts must be non-negative")
        if count > self.max_evidence_items:
            raise RetrievalValidationError(
                f"evidence packet has {count} items; limit is {self.max_evidence_items}"
            )
        if serialized_chars > self.max_evidence_chars:
            raise RetrievalValidationError(
                f"evidence packet has {serialized_chars} characters; "
                f"limit is {self.max_evidence_chars}"
            )

    def validate_prompt(self, prompt: str) -> None:
        if not isinstance(prompt, str):
            raise RetrievalValidationError("prompt must be a string")
        if len(prompt) > self.max_prompt_chars:
            raise RetrievalValidationError(
                f"prompt has {len(prompt)} characters; limit is {self.max_prompt_chars}"
            )
