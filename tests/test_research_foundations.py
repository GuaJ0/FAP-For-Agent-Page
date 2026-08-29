"""Fast, offline tests for the Research Agent's non-LLM foundations."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from agent.config import ConvergenceConfig
from agent.records import (
    AggregateMetrics,
    Decision,
    Event,
    ResourceUsage,
    RunRecord,
    Status,
)
from agent.research.citations import (
    CitationValidationError,
    CompositeCitationSource,
    JsonCitationCatalog,
    validate_proposal_citations,
)
from agent.research.context import build_research_context
from agent.research.prompts import SYSTEM_PROMPT, build_proposal_prompt
from agent.research.schemas import ProposalValidationError, ResearchProposal


def _proposal(parent=4):
    return {
        "schema_version": 1,
        "hypothesis_id": "H-0005",
        "parent_iteration": parent,
        "title": "Hybrid pointwise and pairwise objective",
        "hypothesis": "Add a small BPR term to retain pointwise signal while improving within-user ordering.",
        "rationale": {
            "mechanism": "The two losses preserve calibration and add direct ranking pressure.",
            "metric_alignment": ["GAUC", "nDCG@5"],
            "prior_results_used": [parent] if parent is not None else [],
            "evidence": [
                {
                    "citation_id": "rendle2009bpr",
                    "claim_id": "pairwise-ranking-objective",
                    "application": "The pairwise term directly targets ordering within each user.",
                }
            ],
        },
        "implementation": {
            "target_components": ["training objective"],
            "steps": ["Add a weighted within-user BPR term to the incumbent objective."],
            "hyperparameters": {"lambda_bpr": [0.05, 0.1, 0.2]},
            "must_hold_constant": ["features", "model architecture", "optimizer"],
            "feasibility": {
                "dependencies": ["pytorch"],
                "hardware": "CPU initially; GPU optional",
                "estimated_runtime_impact": "Up to 1.5x the incumbent epoch time",
                "implementation_complexity": "medium",
                "notes": "The added dependency is justified only if vectorized NumPy is too slow.",
            },
        },
        "evaluation": {
            "reference_iteration": parent,
            "primary_metric": "primary",
            "minimum_primary_delta": 0.002,
            "expected_secondary_effects": {
                "GAUC": "increase",
                "nDCG@5": "neutral or increase",
            },
            "ablation": "Compare lambda_bpr=0 with the best nonzero value.",
            "failure_interpretation": "If all nonzero values lose, abandon this hybrid-loss family.",
        },
        "risks": ["Sampling may overweight low-activity users."],
    }


def _record(iteration, primary, decision, status=Status.SUCCESS, parent=None, seconds=0):
    aggregate = None if primary is None else AggregateMetrics(
        primary_mean=primary,
        primary_std=0.001,
        gauc_mean=primary + 0.05,
        ndcg5_mean=primary - 0.05,
        n_seeds=2,
    )
    timestamp = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)).isoformat()
    return RunRecord(
        iteration=iteration,
        parent_iteration=parent,
        timestamp=timestamp,
        hypothesis=f"hypothesis {iteration}",
        diff_path=f"solutions/attempt_{iteration}/config.json",
        status=status,
        seeds=[],
        aggregate=aggregate,
        delta_vs_current_best=None if primary is None else primary - 0.6,
        decision=decision,
        events=[Event("eval_finished", "validation analysis", "evaluator")],
        resources=ResourceUsage(wall_s=12.0),
    )


def test_proposal_parses_and_renders_an_incumbent_relative_handoff():
    proposal = ResearchProposal.from_dict(_proposal(parent=4))

    handoff = proposal.to_handoff_text()

    assert proposal.parent_iteration == 4
    assert "relative to accepted parent iteration 4" in handoff
    assert "Dependencies: pytorch" in handoff
    assert "iteration 0" not in handoff


def test_proposal_rejects_a_reference_other_than_its_parent():
    raw = _proposal(parent=7)
    raw["evaluation"]["reference_iteration"] = 0

    with pytest.raises(ProposalValidationError, match="accepted parent"):
        ResearchProposal.from_dict(raw)


def test_proposal_rejects_forbidden_split_keys_recursively():
    raw = _proposal()
    raw["implementation"]["hyperparameters"]["test_primary"] = 0.9

    with pytest.raises(ProposalValidationError, match="forbidden key"):
        ResearchProposal.from_dict(raw)


def test_curated_catalog_resolves_claim_level_evidence():
    proposal = ResearchProposal.from_dict(_proposal())
    catalog = JsonCitationCatalog()

    resolved = validate_proposal_citations(proposal, catalog)

    assert resolved[0].record.citation_id == "rendle2009bpr"
    assert resolved[0].claim.claim_id == "pairwise-ranking-objective"
    assert catalog.search("pairwise implicit ranking")[0].citation_id == "rendle2009bpr"


def test_unknown_catalog_claim_is_rejected():
    raw = _proposal()
    raw["rationale"]["evidence"][0]["claim_id"] = "invented-claim"
    proposal = ResearchProposal.from_dict(raw)

    with pytest.raises(CitationValidationError, match="does not support"):
        validate_proposal_citations(proposal, JsonCitationCatalog())


def test_composite_source_allows_a_future_source_without_schema_changes():
    catalog = JsonCitationCatalog()

    class FutureRetrievalSource:
        def get(self, citation_id):
            return None

        def search(self, query, limit=10):
            return []

    source = CompositeCitationSource([FutureRetrievalSource(), catalog])
    proposal = ResearchProposal.from_dict(_proposal())

    assert validate_proposal_citations(proposal, source)


def test_context_uses_best_accepted_incumbent_not_latest_record_or_iteration_zero():
    history = [
        _record(0, 0.60, Decision.ACCEPT, seconds=0),
        _record(3, 0.63, Decision.ACCEPT, parent=0, seconds=20),
        _record(4, 0.61, Decision.REVERT, parent=3, seconds=40),
        _record(5, None, Decision.ABANDON, Status.ABANDONED, parent=3, seconds=60),
    ]

    context = build_research_context(
        history,
        ConvergenceConfig(max_iterations=10, max_wall_s=100, epsilon=0.003),
    )

    assert context.incumbent.iteration == 3
    assert context.parent_iteration == 3
    assert context.minimum_meaningful_delta == pytest.approx(0.003)
    assert context.remaining_iterations == 6
    assert context.remaining_wall_s == pytest.approx(40)
    assert context.iterations[-1].status == "abandoned"


def test_empty_context_has_no_parent_and_can_establish_first_incumbent():
    context = build_research_context([])
    proposal = ResearchProposal.from_dict(_proposal(parent=None))

    assert context.parent_iteration is None
    assert "Establish the first accepted validation incumbent" in proposal.to_handoff_text()


def test_prompt_permits_open_source_libraries_and_is_parent_relative():
    context = build_research_context([
        _record(0, 0.60, Decision.ACCEPT, seconds=0),
        _record(8, 0.64, Decision.ACCEPT, parent=0, seconds=10),
    ])
    prompt = build_proposal_prompt(context, JsonCitationCatalog().all())
    shape = json.loads(prompt.split("## Required JSON response shape\n", 1)[1].split(
        "\n\nSelect one experiment", 1
    )[0])

    assert "Open-source ML libraries are permitted" in SYSTEM_PROMPT
    assert "new" in SYSTEM_PROMPT and "dependency must be named and justified" in SYSTEM_PROMPT
    assert shape["parent_iteration"] == 8
    assert shape["evaluation"]["reference_iteration"] == 8
    assert "TEST_METRICS" not in prompt
