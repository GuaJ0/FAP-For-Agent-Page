"""Focused, deterministic tests for the production LLMResearchAgent."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agent.coding.llm import ScriptedClient
from agent.config import ConvergenceConfig
from agent.records import AggregateMetrics, Decision, Event, ResourceUsage, RunRecord, Status
from agent.research.agent import (
    LLMResearchAgent,
    ResearchInputError,
    ResearchOutputError,
)
from agent.research.breadth import BreadthCandidate
from agent.research.citations import JsonCitationCatalog
from agent.research.context import build_research_context
from agent.research.schemas import ResearchProposal


def _proposal(
    parent=3,
    *,
    hypothesis=None,
    hypothesis_id="H-0004",
    citation_id="rendle2009bpr",
    claim_id="pairwise-ranking-objective",
    minimum_primary_delta=0.002,
):
    return {
        "schema_version": 1,
        "hypothesis_id": hypothesis_id,
        "parent_iteration": parent,
        "title": "Weighted BPR objective addition",
        "hypothesis": hypothesis or (
            "Add a small BPR term to the incumbent pointwise objective so direct within-user "
            "ranking pressure is added without discarding its useful signal."
        ),
        "rationale": {
            "mechanism": "The hybrid objective adds ordering pressure while retaining pointwise supervision.",
            "metric_alignment": ["GAUC", "nDCG@5"],
            "prior_results_used": [parent] if parent is not None else [],
            "evidence": [{
                "citation_id": citation_id,
                "claim_id": claim_id,
                "application": "Pairwise ordering is applied within each user, matching the ranking task.",
            }],
        },
        "implementation": {
            "target_components": ["training objective", "within-user sampler"],
            "steps": [
                "Retain the incumbent model and pointwise loss.",
                "Add a weighted BPR term using positive-negative pairs from the same user.",
            ],
            "hyperparameters": {"lambda_bpr": [0.05, 0.1, 0.2]},
            "must_hold_constant": ["features", "model architecture", "checkpoint selection"],
            "feasibility": {
                "dependencies": [],
                "hardware": "Existing CPU environment",
                "estimated_runtime_impact": "At most 1.5x the incumbent epoch time",
                "implementation_complexity": "medium",
                "notes": "Use vectorized pair sampling to remain within the wall-clock budget.",
            },
        },
        "evaluation": {
            "reference_iteration": parent,
            "primary_metric": "primary",
            "minimum_primary_delta": minimum_primary_delta,
            "expected_secondary_effects": {
                "GAUC": "increase through improved within-user ordering",
                "nDCG@5": "remain neutral or improve",
            },
            "ablation": "Compare lambda_bpr=0 with the best nonzero value.",
            "failure_interpretation": "If every nonzero value loses, abandon this hybrid-loss family.",
        },
        "risks": ["Pair sampling may overweight low-activity users."],
    }


def _breadth():
    return {
        "schema_version": 1,
        "candidates": [{
            "candidate_id": "B-HYBRID-RANKING",
            "title": "Hybrid pointwise and pairwise ranking objective",
            "stack_stage": "objective_sampling",
            "primary_change": "Add a weighted BPR pairwise objective term.",
            "mechanism": (
                "Blend retained pointwise supervision with a weighted within-user "
                "pairwise term."
            ),
            "metric_rationale": "Direct ordering pressure should improve GAUC and nDCG@5.",
            "expected_upside": "high",
            "implementation_risk": "low",
            "experiment_cost": "medium",
            "evidence": [
                {
                    "citation_id": "rendle2009bpr",
                    "claim_id": "pairwise-ranking-objective",
                },
                {
                    "citation_id": "covington2016youtube",
                    "claim_id": "ranking-objective-and-watch-time",
                },
            ],
        }, {
            "candidate_id": "B-CONTENT-FEATURES",
            "title": "Candidate content features",
            "stack_stage": "features",
            "primary_change": "Add candidate video metadata content features.",
            "mechanism": "Add video metadata content features for each candidate item.",
            "metric_rationale": "Metadata may improve validation ranking.",
            "expected_upside": "medium",
            "implementation_risk": "medium",
            "experiment_cost": "medium",
            "evidence": [{
                "citation_id": "covington2016youtube",
                "claim_id": "ranking-objective-and-watch-time",
            }],
        }, {
            "candidate_id": "B-DEEPFM",
            "title": "DeepFM interaction architecture",
            "stack_stage": "architecture",
            "primary_change": "Add a DeepFM interaction architecture.",
            "mechanism": "Add a DeepFM interaction tower architecture.",
            "metric_rationale": "Feature interactions may improve validation ranking.",
            "expected_upside": "medium",
            "implementation_risk": "medium",
            "experiment_cost": "medium",
            "evidence": [{
                "citation_id": "guo2017deepfm",
                "claim_id": "joint-low-high-order-interactions",
            }],
        }],
    }


def _record(
    iteration,
    primary,
    decision,
    *,
    hypothesis=None,
    status=Status.SUCCESS,
    event_detail="validation analysis only",
):
    aggregate = None if primary is None else AggregateMetrics(
        primary_mean=primary,
        primary_std=0.001,
        gauc_mean=primary + 0.04,
        ndcg5_mean=primary - 0.04,
        n_seeds=2,
    )
    return RunRecord(
        iteration=iteration,
        parent_iteration=None if iteration == 0 else 3,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        hypothesis=hypothesis or f"hypothesis {iteration}",
        diff_path=f"solutions/attempt_{iteration}/config.json",
        status=status,
        seeds=[],
        aggregate=aggregate,
        delta_vs_current_best=None if primary is None else primary - 0.6,
        decision=decision,
        events=[Event("eval_finished", event_detail, "evaluator")],
        resources=ResourceUsage(wall_s=10.0),
    )


def _agent(tmp_path, responses, *, breadth=None, **kwargs):
    client = ScriptedClient([json.dumps(breadth or _breadth()), *responses])
    agent = LLMResearchAgent(
        llm=client,
        citation_source=JsonCitationCatalog(),
        usage_log_path=tmp_path / "research_usage.jsonl",
        **kwargs,
    )
    return agent, client


def test_valid_proposal_returns_existing_idea_contract(tmp_path):
    history = [_record(3, 0.62, Decision.ACCEPT)]
    agent, client = _agent(tmp_path, [json.dumps(_proposal(parent=3))])

    idea = agent.propose(history)

    assert idea.parent_iteration == 3
    assert idea.hypothesis.startswith("[RESEARCH_PROPOSAL v1]")
    assert "relative to accepted parent iteration 3" in idea.hypothesis
    assert [purpose for _, _, purpose in client.calls] == [
        "research_breadth",
        "research_depth",
    ]


def test_custom_convergence_config_is_used_in_llm_research_context(tmp_path):
    history = [_record(3, 0.62, Decision.ACCEPT)]
    convergence = ConvergenceConfig(
        epsilon=0.007,
        max_iterations=7,
        max_wall_s=123.0,
    )
    agent, client = _agent(
        tmp_path,
        [json.dumps(_proposal(parent=3, minimum_primary_delta=0.007))],
        convergence=convergence,
    )

    agent.propose(history)

    prompt = client.calls[0][1]
    assert '"minimum_meaningful_delta": 0.007' in prompt
    assert '"remaining_iterations": 6' in prompt
    assert '"remaining_wall_s": 123.0' in prompt


def test_llm_research_agent_satisfies_existing_protocol_signature():
    import inspect

    from agent.agents import ResearchAgent

    expected = inspect.signature(ResearchAgent.propose)
    actual = inspect.signature(LLMResearchAgent.propose)
    assert list(actual.parameters) == list(expected.parameters) == ["self", "history"]


def test_malformed_json_is_not_silently_accepted(tmp_path):
    agent, _ = _agent(tmp_path, ["not json"], max_repair_attempts=0)

    with pytest.raises(ResearchOutputError, match="not valid JSON"):
        agent.propose([_record(3, 0.62, Decision.ACCEPT)])


def test_malformed_json_can_be_repaired_once(tmp_path):
    valid = json.dumps(_proposal(parent=3))
    agent, client = _agent(tmp_path, ["```json\n{broken}\n```", valid])

    idea = agent.propose([_record(3, 0.62, Decision.ACCEPT)])

    assert idea.parent_iteration == 3
    assert [purpose for _, _, purpose in client.calls] == [
        "research_breadth",
        "research_depth",
        "research_depth_repair",
    ]
    repair_prompt = client.calls[2][1]
    assert "proposal is not valid JSON" in repair_prompt
    assert "<previous_response>" in repair_prompt
    assert "{broken}" in repair_prompt


def test_failed_repair_raises_clear_research_error(tmp_path):
    agent, _ = _agent(tmp_path, ["not json", "still not json"])

    with pytest.raises(ResearchOutputError, match=r"after 2 call\(s\)") as exc:
        agent.propose([_record(3, 0.62, Decision.ACCEPT)])

    assert "attempt 1" in str(exc.value) and "attempt 2" in str(exc.value)


def test_invalid_citation_is_rejected(tmp_path):
    invalid = json.dumps(_proposal(parent=3, citation_id="invented-paper"))
    agent, _ = _agent(tmp_path, [invalid], max_repair_attempts=0)

    with pytest.raises(ResearchOutputError, match="unknown citation_id"):
        agent.propose([_record(3, 0.62, Decision.ACCEPT)])


def test_prior_results_must_exist_in_research_history(tmp_path):
    raw = _proposal(parent=3)
    raw["rationale"]["prior_results_used"] = [3, 999]
    agent, _ = _agent(tmp_path, [json.dumps(raw)], max_repair_attempts=0)

    with pytest.raises(ResearchOutputError, match="iterations not available"):
        agent.propose([_record(3, 0.62, Decision.ACCEPT)])


def test_hypothesis_id_cannot_collide_with_an_attempted_id(tmp_path):
    historical_data = _proposal(
        parent=3,
        hypothesis_id="H-USED",
        hypothesis="Calibrate the incumbent scores with a held-out monotonic mapping.",
    )
    historical_data["title"] = "Monotonic score calibration"
    historical_data["implementation"]["target_components"] = ["score calibration"]
    historical_data["implementation"]["steps"] = [
        "Fit a monotonic score calibration mapping on validation-safe data."
    ]
    historical = ResearchProposal.from_dict(historical_data).to_handoff_text()
    history = [
        _record(3, 0.62, Decision.ACCEPT),
        _record(4, 0.64, Decision.ACCEPT, hypothesis=historical),
    ]
    candidate = _proposal(
        parent=4,
        hypothesis_id="h-used",
        hypothesis="Add a recent behavior encoder to represent short-term user intent.",
    )
    agent, _ = _agent(tmp_path, [json.dumps(candidate)], max_repair_attempts=0)

    with pytest.raises(ResearchOutputError, match="already used"):
        agent.propose(history)


def test_minimum_delta_must_match_configured_meaningful_threshold(tmp_path):
    agent, _ = _agent(
        tmp_path,
        [json.dumps(_proposal(parent=3, minimum_primary_delta=0.002))],
        max_repair_attempts=0,
        convergence=ConvergenceConfig(epsilon=0.005),
    )

    with pytest.raises(ResearchOutputError, match="minimum meaningful improvement threshold"):
        agent.propose([_record(3, 0.62, Decision.ACCEPT)])


def test_duplicate_reverted_hypothesis_is_rejected(tmp_path):
    duplicate = _proposal(parent=3)["hypothesis"]
    history = [
        _record(3, 0.62, Decision.ACCEPT),
        _record(4, 0.61, Decision.REVERT, hypothesis=duplicate),
    ]
    agent, _ = _agent(
        tmp_path,
        [json.dumps(_proposal(parent=3, hypothesis=duplicate))],
        max_repair_attempts=0,
    )

    with pytest.raises(ResearchOutputError, match="duplicates"):
        agent.propose(history)


def test_meaningful_follow_up_to_accepted_hypothesis_is_allowed_with_a_new_id(tmp_path):
    original = _proposal(parent=3, hypothesis_id="H-ORIGINAL")
    history = [
        _record(3, 0.62, Decision.ACCEPT),
        _record(
            4,
            0.64,
            Decision.ACCEPT,
            hypothesis=ResearchProposal.from_dict(original).to_handoff_text(),
        ),
    ]
    follow_up = _proposal(
        parent=4,
        hypothesis=original["hypothesis"],
        hypothesis_id="H-FOLLOW-UP",
    )
    follow_up["implementation"]["steps"].append(
        "Anneal the pairwise coefficient linearly during the first four epochs."
    )
    follow_up["implementation"]["hyperparameters"]["anneal_epochs"] = 4
    follow_up_breadth = _breadth()
    follow_up_breadth["candidates"][0]["title"] = "Annealed hybrid pairwise objective"
    follow_up_breadth["candidates"][0]["primary_change"] = (
        "Anneal the coefficient of the existing BPR pairwise objective term."
    )
    follow_up_breadth["candidates"][0]["mechanism"] = (
        "Add a BPR pairwise term and anneal its coefficient over four epochs."
    )
    agent, _ = _agent(
        tmp_path,
        [],
        breadth=follow_up_breadth,
    )
    selected = BreadthCandidate.from_dict(
        follow_up_breadth["candidates"][0], "candidate"
    )
    proposal = agent._validate_response(
        json.dumps(follow_up),
        build_research_context(history),
        history,
        selected=selected,
    )

    assert proposal.parent_iteration == 4
    assert proposal.hypothesis_id == "H-FOLLOW-UP"


def test_parent_is_selected_from_best_accepted_incumbent(tmp_path):
    history = [
        _record(0, 0.60, Decision.ACCEPT),
        _record(3, 0.64, Decision.ACCEPT),
        _record(9, 0.63, Decision.REVERT),
    ]
    agent, _ = _agent(tmp_path, [json.dumps(_proposal(parent=3))])

    idea = agent.propose(history)

    assert idea.parent_iteration == 3
    assert "PARENT ITERATION: 3" in idea.hypothesis


def test_handoff_rendering_is_deterministic(tmp_path):
    response = json.dumps(_proposal(parent=3))
    history = [_record(3, 0.62, Decision.ACCEPT)]
    first, _ = _agent(tmp_path / "a", [response])
    second, _ = _agent(tmp_path / "b", [response])

    assert first.propose(history) == second.propose(history)


def test_research_usage_accounting_is_scoped_and_persisted(tmp_path):
    response = json.dumps(_proposal(parent=3))
    agent, _ = _agent(tmp_path, [response])

    agent.propose([_record(3, 0.62, Decision.ACCEPT)])

    assert agent.last_usage["llm_calls"] == 2
    assert agent.last_usage["tokens_in"] > 0
    assert agent.last_usage["tokens_out"] > 0
    # ScriptedClient uses the existing estimator; unknown scripted model price
    # is deliberately zero rather than a fabricated dollar amount.
    assert agent.last_usage["cost_usd"] == 0.0
    rows = [json.loads(line) for line in (tmp_path / "research_usage.jsonl").read_text().splitlines()]
    assert rows[0]["agent"] == "research"
    assert [row["purpose"] for row in rows] == ["research_breadth", "research_depth"]
    assert agent.usage.totals()["tokens_in"] == agent.last_usage["tokens_in"]


def test_hidden_test_material_is_blocked_before_the_llm_call(tmp_path):
    history = [
        _record(
            3,
            0.62,
            Decision.ACCEPT,
            event_detail='TEST_METRICS: {"primary": 0.99}',
        )
    ]
    agent, client = _agent(tmp_path, [json.dumps(_proposal(parent=3))])

    with pytest.raises(ResearchInputError, match="hidden-test material"):
        agent.propose(history)

    assert client.calls == []


def test_normal_prompt_contains_no_hidden_metric_payload(tmp_path):
    agent, client = _agent(tmp_path, [json.dumps(_proposal(parent=3))])

    agent.propose([_record(3, 0.62, Decision.ACCEPT)])

    system, user, _ = client.calls[0]
    assert "TEST_METRICS:" not in user
    assert "test_primary" not in user.lower()
    assert "hidden_test" not in user.lower()
    # A policy statement is allowed; actual hidden-test material is not.
    assert "Never request" in system
