"""Research-only strict JSON Schemas for optional Structured Outputs calls."""
from __future__ import annotations

from typing import Any

from agent.research.breadth import (
    MAX_BREADTH_CANDIDATES,
    MAX_BREADTH_EVIDENCE,
    PRIMARY_FAMILIES_BY_STAGE,
    StackStage,
)
from agent.research.context import ResearchContext


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _text(*, max_length: int = 2_000) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": max_length}


def breadth_response_schema(*, exact_candidates: int) -> dict[str, Any]:
    """Strict schema for either an initial breadth pool or replacement batch."""
    if not 1 <= exact_candidates <= MAX_BREADTH_CANDIDATES:
        raise ValueError("exact_candidates is outside the Research breadth bounds")
    family_values = [
        family
        for stage in StackStage
        for family in PRIMARY_FAMILIES_BY_STAGE[stage]
    ]
    evidence = _object({
        "citation_id": _text(max_length=128),
        "claim_id": _text(max_length=128),
    })
    candidate = _object({
        "candidate_id": {
            "type": "string",
            "pattern": r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$",
        },
        "title": _text(max_length=200),
        "stack_stage": {"type": "string", "enum": [stage.value for stage in StackStage]},
        "primary_family": {"type": "string", "enum": family_values},
        "primary_change": _text(max_length=400),
        "mechanism": _text(max_length=1_200),
        "metric_rationale": _text(max_length=800),
        "expected_upside": {"type": "string", "enum": ["low", "medium", "high"]},
        "implementation_risk": {"type": "string", "enum": ["low", "medium", "high"]},
        "experiment_cost": {"type": "string", "enum": ["low", "medium", "high"]},
        "evidence": {
            "type": "array",
            "items": evidence,
            "minItems": 1,
            "maxItems": MAX_BREADTH_EVIDENCE,
        },
    })
    return _object({
        "schema_version": {"type": "integer", "enum": [1]},
        "candidates": {
            "type": "array",
            "items": candidate,
            "minItems": exact_candidates,
            "maxItems": exact_candidates,
        },
    })


def proposal_response_schema(context: ResearchContext) -> dict[str, Any]:
    """Strict schema for the existing internal ``ResearchProposal`` contract.

    ``hyperparameters`` uses one fixed ``parameters`` property because strict
    Structured Outputs cannot safely express arbitrary object keys. The
    existing ResearchProposal validator already accepts this JSON-safe mapping.
    """
    parent = context.parent_iteration
    parent_schema: dict[str, Any] = (
        {"type": "null", "enum": [None]}
        if parent is None
        else {"type": "integer", "enum": [parent]}
    )
    evidence = _object({
        "citation_id": _text(max_length=128),
        "claim_id": _text(max_length=128),
        "application": _text(max_length=1_200),
    })
    parameter = _object({
        "name": _text(max_length=128),
        "values": {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                ]
            },
            "minItems": 1,
            "maxItems": 12,
        },
        "rationale": _text(max_length=800),
    })
    feasibility = _object({
        "dependencies": {"type": "array", "items": _text(max_length=200), "maxItems": 12},
        "hardware": _text(max_length=500),
        "estimated_runtime_impact": _text(max_length=500),
        "implementation_complexity": {"type": "string", "enum": ["low", "medium", "high"]},
        "notes": _text(max_length=1_200),
    })
    return _object({
        "schema_version": {"type": "integer", "enum": [1]},
        "hypothesis_id": {
            "type": "string",
            "pattern": r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
        },
        "parent_iteration": parent_schema,
        "title": _text(max_length=300),
        "hypothesis": _text(max_length=2_000),
        "rationale": _object({
            "mechanism": _text(max_length=2_000),
            "metric_alignment": {
                "type": "array",
                "items": {"type": "string", "enum": ["GAUC", "nDCG@5"]},
                "minItems": 1,
                "maxItems": 2,
            },
            "prior_results_used": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "maxItems": 20,
            },
            "evidence": {"type": "array", "items": evidence, "minItems": 1, "maxItems": 8},
        }),
        "implementation": _object({
            "target_components": {"type": "array", "items": _text(max_length=300), "minItems": 1, "maxItems": 12},
            "steps": {"type": "array", "items": _text(max_length=1_200), "minItems": 1, "maxItems": 20},
            "hyperparameters": _object({
                "parameters": {"type": "array", "items": parameter, "maxItems": 8},
            }),
            "must_hold_constant": {"type": "array", "items": _text(max_length=300), "minItems": 1, "maxItems": 12},
            "feasibility": feasibility,
        }),
        "evaluation": _object({
            "reference_iteration": parent_schema,
            "primary_metric": {"type": "string", "enum": ["primary"]},
            "minimum_primary_delta": {"type": "number", "minimum": 0},
            "expected_secondary_effects": _object({
                "GAUC": _text(max_length=800),
                "nDCG@5": _text(max_length=800),
            }),
            "ablation": _text(max_length=1_200),
            "failure_interpretation": _text(max_length=1_200),
        }),
        "risks": {"type": "array", "items": _text(max_length=800), "minItems": 1, "maxItems": 12},
    })
