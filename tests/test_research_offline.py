"""Focused tests for the deterministic OfflineResearchAgent."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent.config import ConvergenceConfig
from agent.records import AggregateMetrics, Decision, Event, ResourceUsage, RunRecord, Status
from agent.research.agent import ResearchInputError
from agent.research.citations import JsonCitationCatalog, validate_proposal_citations
from agent.research.offline import (
    DEFAULT_BACKLOG,
    OfflineBacklogExhausted,
    OfflineResearchAgent,
)


def _record(
    iteration,
    primary,
    decision,
    *,
    hypothesis=None,
    status=Status.SUCCESS,
    parent=None,
    event_detail="validation analysis only",
    elapsed_s=None,
):
    aggregate = None if primary is None else AggregateMetrics(
        primary_mean=primary,
        primary_std=0.001,
        gauc_mean=primary + 0.04,
        ndcg5_mean=primary - 0.04,
        n_seeds=2,
    )
    seconds = iteration if elapsed_s is None else elapsed_s
    timestamp = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)).isoformat()
    return RunRecord(
        iteration=iteration,
        parent_iteration=parent,
        timestamp=timestamp,
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


def _baseline(iteration=3, primary=0.62):
    return _record(iteration, primary, Decision.ACCEPT, parent=0 if iteration else None)


def _concluded_from_idea(iteration, idea, *, abandoned=False):
    return _record(
        iteration,
        None if abandoned else 0.61,
        Decision.ABANDON if abandoned else Decision.REVERT,
        hypothesis=idea.hypothesis,
        status=Status.ABANDONED if abandoned else Status.SUCCESS,
        parent=idea.parent_iteration,
    )


def _accepted_from_idea(iteration, idea):
    return _record(
        iteration,
        0.64,
        Decision.ACCEPT,
        hypothesis=idea.hypothesis,
        parent=idea.parent_iteration,
    )


def test_offline_output_is_deterministic_for_the_same_history():
    history = [_baseline()]

    first = OfflineResearchAgent().propose(history)
    second = OfflineResearchAgent().propose(history)

    assert first == second
    assert "OFFLINE-HYBRID-BPR" in first.hypothesis


def test_offline_agent_can_propose_with_one_iteration_cap_after_bootstrap():
    agent = OfflineResearchAgent(
        convergence=ConvergenceConfig(max_iterations=1),
    )

    idea = agent.propose([_baseline(iteration=0, primary=0.60)])

    assert idea.parent_iteration == 0
    assert idea.hypothesis.startswith("[RESEARCH_PROPOSAL v1]\n")


def test_offline_agent_exhausts_one_iteration_cap_after_research_experiment():
    agent = OfflineResearchAgent(
        convergence=ConvergenceConfig(max_iterations=1),
    )
    history = [_baseline(iteration=0, primary=0.60)]
    idea = agent.propose(history)
    history.append(_concluded_from_idea(1, idea))

    with pytest.raises(OfflineBacklogExhausted, match="budget is exhausted"):
        agent.propose(history)


def test_offline_proposal_uses_current_best_accepted_parent():
    history = [
        _record(0, 0.60, Decision.ACCEPT),
        _record(7, 0.65, Decision.ACCEPT, parent=0),
        _record(8, 0.63, Decision.REVERT, parent=7),
    ]

    idea = OfflineResearchAgent().propose(history)

    assert idea.parent_iteration == 7
    assert "PARENT ITERATION: 7" in idea.hypothesis
    assert "relative to accepted parent iteration 7" in idea.hypothesis


def test_offline_agent_skips_an_abandoned_backlog_idea():
    agent = OfflineResearchAgent()
    history = [_baseline()]
    first = agent.propose(history)
    history.append(_concluded_from_idea(4, first, abandoned=True))

    second = agent.propose(history)

    assert "OFFLINE-HYBRID-BPR" in first.hypothesis
    assert "OFFLINE-HYBRID-BPR" not in second.hypothesis
    assert "OFFLINE-GAUC-WEIGHTED-BPR" in second.hypothesis


def test_offline_agent_does_not_repeat_an_accepted_proposal():
    agent = OfflineResearchAgent()
    history = [_baseline()]
    first = agent.propose(history)
    history.append(_accepted_from_idea(4, first))

    second = agent.propose(history)

    assert "OFFLINE-HYBRID-BPR" in first.hypothesis
    assert "OFFLINE-HYBRID-BPR" not in second.hypothesis
    assert "OFFLINE-GAUC-WEIGHTED-BPR" in second.hypothesis


def test_offline_agent_advances_as_history_grows():
    agent = OfflineResearchAgent()
    history = [_baseline()]

    first = agent.propose(history)
    history.append(_concluded_from_idea(4, first))
    second = agent.propose(history)
    history.append(_concluded_from_idea(5, second))
    third = agent.propose(history)

    assert first.hypothesis != second.hypothesis != third.hypothesis
    assert "OFFLINE-DIN-SHORT-HISTORY" in third.hypothesis


def test_offline_proposals_use_valid_catalog_claims():
    source = JsonCitationCatalog()
    agent = OfflineResearchAgent(citation_source=source)

    proposal = agent.select_proposal([_baseline()])
    resolved = validate_proposal_citations(proposal, source)

    assert resolved
    assert resolved[0].record.citation_id == proposal.rationale.evidence[0].citation_id
    assert resolved[0].claim.claim_id == proposal.rationale.evidence[0].claim_id


def test_offline_agent_blocks_hidden_test_history():
    history = [
        _record(
            3,
            0.62,
            Decision.ACCEPT,
            event_detail='TEST_METRICS: {"primary": 0.99}',
        )
    ]

    with pytest.raises(ResearchInputError, match="hidden-test material"):
        OfflineResearchAgent().propose(history)


def test_offline_agent_uses_the_shared_coding_handoff_renderer():
    agent = OfflineResearchAgent()
    history = [_baseline()]

    proposal = agent.select_proposal(history)
    idea = agent.propose(history)

    assert idea.hypothesis == proposal.to_handoff_text()
    assert idea.hypothesis.startswith("[RESEARCH_PROPOSAL v1]\n")
    for heading in (
        "HYPOTHESIS:",
        "EVIDENCE:",
        "IMPLEMENTATION:",
        "FEASIBILITY:",
        "SUCCESS CRITERION:",
        "FAILURE INTERPRETATION:",
    ):
        assert heading in idea.hypothesis
    assert "TEST_METRICS:" not in idea.hypothesis


def test_offline_backlog_exhaustion_is_explicit():
    agent = OfflineResearchAgent()
    history = [_baseline()]

    for offset in range(len(DEFAULT_BACKLOG)):
        idea = agent.propose(history)
        history.append(_concluded_from_idea(4 + offset, idea))

    with pytest.raises(OfflineBacklogExhausted, match="backlog exhausted"):
        agent.propose(history)


def test_tight_budget_prefers_the_cheaper_viable_experiment():
    # Both ranking proposals fit, but only two iterations and 1000 seconds
    # remain. The deterministic feasibility ordering should choose the cheaper
    # weighted-sampler experiment rather than blindly taking rank 1.
    cfg = ConvergenceConfig(max_iterations=3, max_wall_s=1000.0)
    agent = OfflineResearchAgent(convergence=cfg)

    idea = agent.propose([_baseline()])

    assert "OFFLINE-GAUC-WEIGHTED-BPR" in idea.hypothesis


def test_offline_agent_satisfies_existing_research_protocol_signature():
    import inspect

    from agent.agents import ResearchAgent

    assert list(inspect.signature(OfflineResearchAgent.propose).parameters) == list(
        inspect.signature(ResearchAgent.propose).parameters
    ) == ["self", "history"]
