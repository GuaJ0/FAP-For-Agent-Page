"""Strict, serialisable contract for one Research Agent proposal.

The shared pipeline still consumes ``agent.agents.Idea``.  This richer schema
is deliberately internal to Agent 1: after validation, ``to_handoff_text()``
produces the structured string placed in ``Idea.hypothesis`` and
``parent_iteration`` is copied into the existing field of the same name.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

from agent.config import FORBIDDEN_PAYLOAD_KEYS


class ProposalValidationError(ValueError):
    """The model response does not satisfy the Research proposal contract."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProposalValidationError(f"{path} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {sorted(missing)}")
        if extra:
            parts.append(f"unexpected {sorted(extra)}")
        raise ProposalValidationError(f"{path}: " + "; ".join(parts))


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProposalValidationError(f"{path} must be a non-empty string")
    return value.strip()


def _text_tuple(value: Any, path: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProposalValidationError(f"{path} must be an array of strings")
    out = tuple(_text(item, f"{path}[{i}]") for i, item in enumerate(value))
    if not out and not allow_empty:
        raise ProposalValidationError(f"{path} must not be empty")
    return out


def _iteration(value: Any, path: str) -> Optional[int]:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProposalValidationError(f"{path} must be a non-negative integer or null")
    return value


def _json_safe(value: Any, path: str = "proposal") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProposalValidationError(f"{path} contains a non-string key")
            if key.lower() in FORBIDDEN_PAYLOAD_KEYS:
                raise ProposalValidationError(f"{path} contains forbidden key {key!r}")
            _json_safe(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for i, child in enumerate(value):
            _json_safe(child, f"{path}[{i}]")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ProposalValidationError(f"{path} is not JSON-serialisable")


@dataclass(frozen=True)
class EvidenceReference:
    citation_id: str
    claim_id: str
    application: str

    @classmethod
    def from_dict(cls, value: Any, path: str) -> "EvidenceReference":
        data = _mapping(value, path)
        _exact_keys(data, {"citation_id", "claim_id", "application"}, path)
        return cls(
            citation_id=_text(data["citation_id"], f"{path}.citation_id"),
            claim_id=_text(data["claim_id"], f"{path}.claim_id"),
            application=_text(data["application"], f"{path}.application"),
        )


@dataclass(frozen=True)
class Rationale:
    mechanism: str
    metric_alignment: tuple[str, ...]
    prior_results_used: tuple[int, ...]
    evidence: tuple[EvidenceReference, ...]

    @classmethod
    def from_dict(cls, value: Any) -> "Rationale":
        path = "proposal.rationale"
        data = _mapping(value, path)
        _exact_keys(data, {"mechanism", "metric_alignment", "prior_results_used", "evidence"}, path)
        prior_raw = data["prior_results_used"]
        if not isinstance(prior_raw, Sequence) or isinstance(prior_raw, (str, bytes)):
            raise ProposalValidationError(f"{path}.prior_results_used must be an array")
        prior = tuple(
            _iteration(item, f"{path}.prior_results_used[{i}]")
            for i, item in enumerate(prior_raw)
        )
        if any(item is None for item in prior):
            raise ProposalValidationError(f"{path}.prior_results_used cannot contain null")
        evidence_raw = data["evidence"]
        if not isinstance(evidence_raw, Sequence) or isinstance(evidence_raw, (str, bytes)):
            raise ProposalValidationError(f"{path}.evidence must be an array")
        evidence = tuple(
            EvidenceReference.from_dict(item, f"{path}.evidence[{i}]")
            for i, item in enumerate(evidence_raw)
        )
        if not evidence:
            raise ProposalValidationError(f"{path}.evidence must contain at least one citation")
        return cls(
            mechanism=_text(data["mechanism"], f"{path}.mechanism"),
            metric_alignment=_text_tuple(data["metric_alignment"], f"{path}.metric_alignment"),
            prior_results_used=tuple(int(item) for item in prior),
            evidence=evidence,
        )


@dataclass(frozen=True)
class FeasibilityAssessment:
    dependencies: tuple[str, ...]
    hardware: str
    estimated_runtime_impact: str
    implementation_complexity: str
    notes: str

    @classmethod
    def from_dict(cls, value: Any) -> "FeasibilityAssessment":
        path = "proposal.implementation.feasibility"
        data = _mapping(value, path)
        _exact_keys(
            data,
            {"dependencies", "hardware", "estimated_runtime_impact", "implementation_complexity", "notes"},
            path,
        )
        complexity = _text(data["implementation_complexity"], f"{path}.implementation_complexity").lower()
        if complexity not in {"low", "medium", "high"}:
            raise ProposalValidationError(f"{path}.implementation_complexity must be low, medium, or high")
        return cls(
            dependencies=_text_tuple(data["dependencies"], f"{path}.dependencies", allow_empty=True),
            hardware=_text(data["hardware"], f"{path}.hardware"),
            estimated_runtime_impact=_text(
                data["estimated_runtime_impact"], f"{path}.estimated_runtime_impact"
            ),
            implementation_complexity=complexity,
            notes=_text(data["notes"], f"{path}.notes"),
        )


@dataclass(frozen=True)
class ImplementationPlan:
    target_components: tuple[str, ...]
    steps: tuple[str, ...]
    hyperparameters: dict[str, Any]
    must_hold_constant: tuple[str, ...]
    feasibility: FeasibilityAssessment

    @classmethod
    def from_dict(cls, value: Any) -> "ImplementationPlan":
        path = "proposal.implementation"
        data = _mapping(value, path)
        _exact_keys(
            data,
            {"target_components", "steps", "hyperparameters", "must_hold_constant", "feasibility"},
            path,
        )
        hyperparameters = dict(_mapping(data["hyperparameters"], f"{path}.hyperparameters"))
        _json_safe(hyperparameters, f"{path}.hyperparameters")
        return cls(
            target_components=_text_tuple(data["target_components"], f"{path}.target_components"),
            steps=_text_tuple(data["steps"], f"{path}.steps"),
            hyperparameters=hyperparameters,
            must_hold_constant=_text_tuple(data["must_hold_constant"], f"{path}.must_hold_constant"),
            feasibility=FeasibilityAssessment.from_dict(data["feasibility"]),
        )


@dataclass(frozen=True)
class EvaluationPlan:
    reference_iteration: Optional[int]
    primary_metric: str
    minimum_primary_delta: float
    expected_secondary_effects: dict[str, str]
    ablation: str
    failure_interpretation: str

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationPlan":
        path = "proposal.evaluation"
        data = _mapping(value, path)
        _exact_keys(
            data,
            {
                "reference_iteration",
                "primary_metric",
                "minimum_primary_delta",
                "expected_secondary_effects",
                "ablation",
                "failure_interpretation",
            },
            path,
        )
        delta = data["minimum_primary_delta"]
        if not isinstance(delta, (int, float)) or isinstance(delta, bool) or delta < 0:
            raise ProposalValidationError(f"{path}.minimum_primary_delta must be a non-negative number")
        secondary_raw = _mapping(data["expected_secondary_effects"], f"{path}.expected_secondary_effects")
        secondary = {
            _text(key, f"{path}.expected_secondary_effects key"): _text(
                item, f"{path}.expected_secondary_effects.{key}"
            )
            for key, item in secondary_raw.items()
        }
        primary_metric = _text(data["primary_metric"], f"{path}.primary_metric")
        if primary_metric != "primary":
            raise ProposalValidationError(f"{path}.primary_metric must be 'primary'")
        return cls(
            reference_iteration=_iteration(data["reference_iteration"], f"{path}.reference_iteration"),
            primary_metric=primary_metric,
            minimum_primary_delta=float(delta),
            expected_secondary_effects=secondary,
            ablation=_text(data["ablation"], f"{path}.ablation"),
            failure_interpretation=_text(data["failure_interpretation"], f"{path}.failure_interpretation"),
        )


@dataclass(frozen=True)
class ResearchProposal:
    schema_version: int
    hypothesis_id: str
    parent_iteration: Optional[int]
    title: str
    hypothesis: str
    rationale: Rationale
    implementation: ImplementationPlan
    evaluation: EvaluationPlan
    risks: tuple[str, ...]

    @classmethod
    def from_json(cls, text: str) -> "ResearchProposal":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProposalValidationError(f"proposal is not valid JSON: {exc}") from exc
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value: Any) -> "ResearchProposal":
        path = "proposal"
        data = _mapping(value, path)
        _json_safe(data)
        _exact_keys(
            data,
            {
                "schema_version",
                "hypothesis_id",
                "parent_iteration",
                "title",
                "hypothesis",
                "rationale",
                "implementation",
                "evaluation",
                "risks",
            },
            path,
        )
        if data["schema_version"] != 1:
            raise ProposalValidationError("proposal.schema_version must be 1")
        parent = _iteration(data["parent_iteration"], "proposal.parent_iteration")
        evaluation = EvaluationPlan.from_dict(data["evaluation"])
        if evaluation.reference_iteration != parent:
            raise ProposalValidationError(
                "proposal.evaluation.reference_iteration must equal proposal.parent_iteration; "
                "success is measured against the accepted parent, never a hard-coded baseline"
            )
        return cls(
            schema_version=1,
            hypothesis_id=_text(data["hypothesis_id"], "proposal.hypothesis_id"),
            parent_iteration=parent,
            title=_text(data["title"], "proposal.title"),
            hypothesis=_text(data["hypothesis"], "proposal.hypothesis"),
            rationale=Rationale.from_dict(data["rationale"]),
            implementation=ImplementationPlan.from_dict(data["implementation"]),
            evaluation=evaluation,
            risks=_text_tuple(data["risks"], "proposal.risks"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_handoff_text(self) -> str:
        """Render the existing Coding Agent's ``Idea.hypothesis`` payload."""
        impl = self.implementation
        ev = self.evaluation
        lines = [
            "[RESEARCH_PROPOSAL v1]",
            f"ID: {self.hypothesis_id}",
            f"TITLE: {self.title}",
            f"PARENT ITERATION: {self.parent_iteration if self.parent_iteration is not None else 'none'}",
            "",
            "HYPOTHESIS:",
            self.hypothesis,
            "",
            "WHY THIS SHOULD HELP:",
            self.rationale.mechanism,
            f"Metric alignment: {', '.join(self.rationale.metric_alignment)}",
            "",
            "EVIDENCE:",
        ]
        lines.extend(
            f"- [{item.citation_id}/{item.claim_id}] {item.application}"
            for item in self.rationale.evidence
        )
        if self.rationale.prior_results_used:
            lines.append(
                "Prior validation iterations used: "
                + ", ".join(str(i) for i in self.rationale.prior_results_used)
            )
        lines.extend(["", "IMPLEMENTATION:"])
        lines.extend(f"{i}. {step}" for i, step in enumerate(impl.steps, 1))
        lines.extend([
            "",
            "TARGET COMPONENTS:",
            *[f"- {item}" for item in impl.target_components],
            "",
            "HYPERPARAMETERS:",
        ])
        lines.extend(
            f"- {key}: {json.dumps(value, separators=(',', ':'))}"
            for key, value in sorted(impl.hyperparameters.items())
        )
        lines.extend(["", "KEEP CONSTANT:", *[f"- {item}" for item in impl.must_hold_constant]])
        lines.extend([
            "",
            "FEASIBILITY:",
            f"- Dependencies: {', '.join(impl.feasibility.dependencies) or 'none beyond current environment'}",
            f"- Hardware: {impl.feasibility.hardware}",
            f"- Runtime impact: {impl.feasibility.estimated_runtime_impact}",
            f"- Implementation complexity: {impl.feasibility.implementation_complexity}",
            f"- Notes: {impl.feasibility.notes}",
            "",
            "SUCCESS CRITERION:",
        ])
        if ev.reference_iteration is None:
            lines.append("Establish the first accepted validation incumbent.")
        else:
            lines.append(
                f"Validation {ev.primary_metric} improves by more than "
                f"{ev.minimum_primary_delta:g} relative to accepted parent iteration "
                f"{ev.reference_iteration}."
            )
        if ev.expected_secondary_effects:
            lines.append("Expected secondary effects:")
            lines.extend(f"- {key}: {value}" for key, value in sorted(ev.expected_secondary_effects.items()))
        lines.extend([
            "",
            "ABLATION:",
            ev.ablation,
            "",
            "FAILURE INTERPRETATION:",
            ev.failure_interpretation,
            "",
            "RISKS:",
            *[f"- {risk}" for risk in self.risks],
        ])
        return "\n".join(lines).strip() + "\n"
