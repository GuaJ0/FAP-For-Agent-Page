"""Prompt contract for the future LLM-backed Research Agent."""
from __future__ import annotations

import json
from typing import Optional, Sequence

from agent.research.breadth import (
    BreadthCandidate,
    StackCoverageSummary,
    StackStage,
)
from agent.research.citations import CitationRecord
from agent.research.context import ResearchContext


SYSTEM_PROMPT = """\
You are Agent 1, the Research Agent in an automated ML experimentation loop.
Your job is to propose exactly one literature-grounded experiment for the
Coding Agent. Do not write code and do not make execution or acceptance
decisions owned by the Coding or Evaluator agents.

SCIENTIFIC RULES
  - Use only validation history supplied in the context. Never request, infer,
    or mention hidden-test results.
  - Never use test labels, final-holdout performance, challenge/competition
    results or rankings, leaderboard feedback, or changes to the official
    scoring code as an experiment or model-selection signal.
  - Compare success against the supplied accepted parent iteration. Never
    hard-code iteration 0 as the comparison unless iteration 0 is actually the
    supplied parent.
  - Do not repeat a measured dead end unless the proposal states the material
    mechanism change that makes the new experiment different.
  - Change one attributable mechanism at a time and state what remains fixed.
  - Cite only citation_id/claim_id pairs in the supplied evidence packet.

FEASIBILITY
  - Open-source ML libraries are permitted; NumPy is not an official challenge
    restriction. The current starter happens to be NumPy-based, so any new
    dependency must be named and justified in terms of installation burden,
    hardware, memory, runtime, and the remaining experiment budget.
  - A proposal may use PyTorch, JAX, scikit-learn, or another open-source
    library when its expected benefit justifies that cost. Do not silently
    assume a dependency is already installed.

OUTPUT
  Return exactly one JSON object and no markdown fences or commentary. It must
  follow the supplied schema exactly. Every free-text field must be concrete
  enough for the Coding Agent to implement without guessing.
"""

BREADTH_SYSTEM_PROMPT = """\
You are performing the cheap breadth phase inside Agent 1, the Research Agent.
Generate several shallow, structurally distinct experiment directions for the
current KuaiRand-Pure validation history. Do not write code and do not produce
the full ResearchProposal yet.

Use only the supplied validation context and bundled evidence packet.
Never request, infer, or mention hidden-test results. Do not propose changing the
official evaluation metric, benchmark, or evaluate.py. Never use test labels,
final-holdout performance, challenge/competition results or rankings, or
leaderboard feedback for development or model selection. Treat stack coverage as
a soft diversity preference: explored stages are not forbidden when a clearly
stronger idea justifies revisiting them.

Return exactly one JSON object matching the supplied breadth schema, with no
markdown fence or commentary. Keep every candidate concise.
"""

BREADTH_HISTORY_LIMIT = 8


def _citation_packet(records: Sequence[CitationRecord]) -> list[dict]:
    return [
        {
            "citation_id": record.citation_id,
            "title": record.title,
            "authors": list(record.authors),
            "year": record.year,
            "venue": record.venue,
            "url": record.url,
            "claims": [
                {"claim_id": claim.claim_id, "text": claim.text}
                for claim in record.claims
            ],
            "tags": list(record.tags),
        }
        for record in records
    ]


def proposal_shape(context: ResearchContext) -> dict:
    """Exact response shape, populated with context-relative placeholders."""
    parent = context.parent_iteration
    return {
        "schema_version": 1,
        "hypothesis_id": "H-<unique-id>",
        "parent_iteration": parent,
        "title": "Short experiment title",
        "hypothesis": "If <change>, validation primary should improve because <mechanism>.",
        "rationale": {
            "mechanism": "Task-specific causal or modelling rationale.",
            "metric_alignment": ["GAUC", "nDCG@5"],
            "prior_results_used": [parent] if parent is not None else [],
            "evidence": [
                {
                    "citation_id": "an ID from evidence_packet",
                    "claim_id": "a claim belonging to that citation",
                    "application": "Why that supported claim applies to this model and history.",
                }
            ],
        },
        "implementation": {
            "target_components": ["component to change"],
            "steps": ["Specific implementation step"],
            "hyperparameters": {"parameter": ["values to test"]},
            "must_hold_constant": ["unchanged component"],
            "feasibility": {
                "dependencies": ["new dependency, or an empty array"],
                "hardware": "CPU/GPU requirement",
                "estimated_runtime_impact": "Expected change versus the current run",
                "implementation_complexity": "low|medium|high",
                "notes": "Installation, memory, and budget considerations.",
            },
        },
        "evaluation": {
            "reference_iteration": parent,
            "primary_metric": "primary",
            "minimum_primary_delta": context.minimum_meaningful_delta,
            "expected_secondary_effects": {
                "GAUC": "expected direction and reason",
                "nDCG@5": "expected direction and reason",
            },
            "ablation": "Minimal comparison that isolates the proposed mechanism.",
            "failure_interpretation": "What a neutral or negative result would imply.",
        },
        "risks": ["Concrete scientific or feasibility risk"],
    }


def breadth_shape() -> dict:
    """Compact strict response shape for one bounded breadth call."""
    return {
        "schema_version": 1,
        "candidates": [{
            "candidate_id": "B-<unique-id>",
            "title": "Concise direction",
            "stack_stage": "one of: " + " | ".join(stage.value for stage in StackStage),
            "primary_change": "The single intervention that defines this candidate.",
            "mechanism": "One or two sentences describing the single mechanism changed.",
            "metric_rationale": "Expected effect on GAUC and/or nDCG@5.",
            "expected_upside": "low|medium|high",
            "implementation_risk": "low|medium|high",
            "experiment_cost": "low|medium|high",
            "evidence": [{
                "citation_id": "an ID from evidence_packet",
                "claim_id": "a claim belonging to that citation",
            }],
        }],
    }


def build_breadth_prompt(
    context: ResearchContext,
    citations: Sequence[CitationRecord],
    coverage: StackCoverageSummary,
    *,
    max_candidates: int,
) -> str:
    """Build a bounded breadth prompt without full history or full papers."""
    compact_context = {
        "task": context.task,
        "label": context.label,
        "primary_metric": context.primary_metric,
        "incumbent": context.to_prompt_dict()["incumbent"],
        "parent_iteration": context.parent_iteration,
        "minimum_meaningful_delta": context.minimum_meaningful_delta,
        "remaining_iterations": context.remaining_iterations,
        "remaining_wall_s": context.remaining_wall_s,
        "recent_iterations": [
            {
                "iteration": item.iteration,
                "parent_iteration": item.parent_iteration,
                "hypothesis": item.hypothesis,
                "status": item.status,
                "decision": item.decision,
                "primary_mean": item.primary_mean,
                "gauc_mean": item.gauc_mean,
                "ndcg5_mean": item.ndcg5_mean,
                "evaluator_events": list(item.evaluator_events),
            }
            for item in context.iterations[-BREADTH_HISTORY_LIMIT:]
        ],
        "stack_coverage": coverage.to_prompt_dict(),
    }
    shape = breadth_shape()
    return (
        "## Compact validation-only research context\n"
        + json.dumps(compact_context, indent=2, sort_keys=True)
        + "\n\n## Bundled evidence packet\n"
        + json.dumps(_citation_packet(citations), indent=2, sort_keys=True)
        + "\n\n## Required breadth JSON shape\n"
        + json.dumps(shape, indent=2, sort_keys=True)
        + f"\n\nGenerate between 3 and {max_candidates} candidates in one response. "
        "For each candidate, primary_change must name exactly one authoritative primary "
        "intervention. The mechanism may include ancillary optimizer, regularizer, sampler, "
        "or training details, but they must not replace or conflict with primary_change. "
        "Do not stuff unrelated feature, architecture, objective, optimization, or inference "
        "changes into its mechanism. Cover "
        "multiple stack stages where worthwhile, but do not include a weak idea solely "
        "to fill an unexplored stage. This is shallow selection material, not the full proposal."
    )


def build_proposal_prompt(
    context: ResearchContext,
    citations: Sequence[CitationRecord],
    selected_candidate: Optional[BreadthCandidate] = None,
) -> str:
    """Build a prompt independent of how the evidence records were retrieved."""
    selected = ""
    if selected_candidate is not None:
        selected = (
            "\n\n## Selected breadth direction (binding)\n"
            + json.dumps(selected_candidate.to_prompt_dict(), indent=2, sort_keys=True)
            + "\nDevelop this exact mechanism and stack stage into the full proposal. "
            "State the primary intervention clearly in hypothesis and implementation steps; "
            "keep ancillary optimizer and regularization details secondary. "
            "Retain at least one listed citation/claim pair. Do not silently switch "
            "to another direction.\n"
        )
    return (
        "## Validation-only research context\n"
        + json.dumps(context.to_prompt_dict(), indent=2, sort_keys=True)
        + "\n\n## Evidence packet\n"
        + json.dumps(_citation_packet(citations), indent=2, sort_keys=True)
        + selected
        + "\n\n## Required JSON response shape\n"
        + json.dumps(proposal_shape(context), indent=2, sort_keys=True)
        + "\n\nSelect one experiment. The parent_iteration and "
        "evaluation.reference_iteration in your response must both equal the "
        "parent_iteration supplied in the research context."
    )


def build_breadth_repair_prompt(
    *,
    original_prompt: str,
    original_response: str,
    validation_error: str,
    response_limit: int = 12_000,
) -> str:
    """One bounded correction request for malformed or exhausted breadth."""
    response = original_response
    if len(response) > response_limit:
        response = response[:response_limit] + "\n... (response truncated)"
    return (
        original_prompt
        + "\n\n## Breadth repair required\n"
        + "The previous breadth response was rejected by deterministic validation/filtering.\n"
        + "Validation error:\n"
        + validation_error.strip()
        + "\n\nPrevious response:\n<previous_response>\n"
        + response
        + "\n</previous_response>\n\n"
        + "Return one corrected breadth JSON object only, with at least 3 candidates "
        + "and no more than the stated candidate limit."
    )


def build_repair_prompt(
    *,
    original_prompt: str,
    original_response: str,
    validation_error: str,
    response_limit: int = 24_000,
) -> str:
    """One bounded correction request after parsing or validation fails."""
    response = original_response
    if len(response) > response_limit:
        response = response[:response_limit] + "\n... (response truncated)"
    return (
        original_prompt
        + "\n\n## Repair required\n"
        + "Your previous response was rejected by deterministic validation.\n"
        + "Validation error:\n"
        + validation_error.strip()
        + "\n\nPrevious response:\n<previous_response>\n"
        + response
        + "\n</previous_response>\n\n"
        + "Return one corrected JSON object only. Do not explain the correction, "
        + "use markdown fences, or repeat the invalid response."
    )
