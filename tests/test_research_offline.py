"""Focused tests for the deterministic OfflineResearchAgent."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import json

from agent.agents import Idea
from agent.config import ConvergenceConfig
from agent.records import AggregateMetrics, Decision, Event, ResourceUsage, RunRecord, Status
from agent.research.agent import ResearchInputError
from agent.research.citations import JsonCitationCatalog, validate_proposal_citations
from agent.research.context import build_research_context
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


# ---------------------------------------------------------------------------
# Multiple real variants per complex direction.
#
# The risk being defended against: one Coding Agent generation at one setting
# loses, and the ledger records a "Don't" for the whole mechanism. Each complex
# direction therefore carries 2-3 deliberately different entries. That only
# works if the duplicate-hypothesis guard treats them as legitimately distinct
# proposals -- these tests pin that down, including the pathological case.
# ---------------------------------------------------------------------------

COMPLEX_FAMILIES = {
    "DIN-SEQUENCE": ("DIN-SHORT-HISTORY", "DIN-LONG-HISTORY", "DIN-MEAN-POOL"),
    "MULTITASK": ("MULTITASK-ENGAGEMENT", "MULTITASK-ALL-ENGAGEMENT", "MULTITASK-CLICK-HEAVY"),
    "WATCHTIME": ("WATCHTIME-AUXILIARY", "WATCHTIME-CENSORED", "WATCHTIME-RATIO"),
}


def _entry(key):
    return next(e for e in DEFAULT_BACKLOG if e.key == key)


def test_each_complex_direction_carries_several_real_variants():
    """Not a single attempt each -- a false Don't on these is the expensive
    failure mode, because they are the directions most likely to be
    implemented badly on the first try."""
    for family, members in COMPLEX_FAMILIES.items():
        keys = {e.key for e in DEFAULT_BACKLOG}
        assert set(members) <= keys, f"{family} is missing variants"
        assert len(members) >= 2


def test_variants_of_one_direction_are_not_rejected_as_duplicates():
    """The real question for the campaign: after variant A has been attempted
    and reverted, will the backlog still offer B and C?"""
    from agent.research.agent import _validate_proposal_against_context
    from agent.research.context import build_research_context

    agent = OfflineResearchAgent(convergence=ConvergenceConfig(max_iterations=50))
    for members in COMPLEX_FAMILIES.values():
        history = [_baseline(iteration=0, primary=0.60)]
        for offset, key in enumerate(members, start=1):
            context = build_research_context(history, agent.convergence)
            proposal = _entry(key).build(context)
            # Must not raise: each variant is a distinct experiment.
            _validate_proposal_against_context(proposal, context, history)
            history.append(_concluded_from_idea(
                offset, Idea(proposal.to_handoff_text(), proposal.parent_iteration)))


def test_variants_differ_in_hyperparameters_not_only_in_wording():
    """A variant that changes only its prose is not a second measurement of
    anything -- it is the same experiment with a new name."""
    for members in COMPLEX_FAMILIES.values():
        seen = []
        for key in members:
            hp = json.dumps(_entry(key).hyperparameters, sort_keys=True)
            assert hp not in seen, f"{key} duplicates another variant's settings"
            seen.append(hp)


def test_an_exact_clone_of_an_attempted_entry_is_still_rejected():
    """The guard must not have been loosened: identical wording AND identical
    settings is a repeat, and still has to be refused."""
    import dataclasses

    from agent.research.agent import DuplicateHypothesisError, _validate_proposal_against_context
    from agent.research.context import build_research_context

    agent = OfflineResearchAgent(convergence=ConvergenceConfig(max_iterations=50))
    original = _entry("DIN-SHORT-HISTORY")
    context = build_research_context([], agent.convergence)
    attempted = original.build(context)
    history = [_concluded_from_idea(1, Idea(attempted.to_handoff_text(), attempted.parent_iteration))]

    clone = dataclasses.replace(original, key="DIN-SHORT-HISTORY-COPY")
    context = build_research_context(history, agent.convergence)
    with pytest.raises(DuplicateHypothesisError):
        _validate_proposal_against_context(clone.build(context), context, history)


# ---------------------------------------------------------------------------
# The two directions from solution/ideas.md the backlog never covered.
# ---------------------------------------------------------------------------

def test_the_backlog_covers_every_unexplored_direction_from_ideas_md():
    keys = {e.key for e in DEFAULT_BACKLOG}
    # ideas.md "Unexplored directions", items 1-7 in order.
    assert {"HYBRID-BPR", "GAUC-WEIGHTED-BPR"} <= keys        # 1 ranking loss
    assert "DIN-SHORT-HISTORY" in keys                        # 2 behaviour sequences
    assert "MULTITASK-ENGAGEMENT" in keys                     # 3 multi-task
    assert "WATCHTIME-AUXILIARY" in keys                      # 4 watch-time
    assert "DEEPFM" in keys                                   # 5 architecture
    assert "TIME-DRIFT" in keys                               # 6 time features and drift
    assert "LOG-RANDOM-DIAGNOSTIC" in keys                    # 7 unbiased validation


def test_every_backlog_entry_resolves_to_a_declared_findings_family():
    """A new backlog entry must not silently become its own ungrouped family --
    that is exactly how a variant set degrades back into unrelated one-shot
    verdicts."""
    from agent.research.findings import DIRECTION_FAMILIES, resolve_family

    for entry in DEFAULT_BACKLOG:
        assert resolve_family(entry.key) in DIRECTION_FAMILIES, entry.key


def test_the_unbiased_validation_entry_never_changes_the_selection_metric():
    """It is a read-only diagnostic. If it could move checkpoint selection it
    would silently redefine what every delta in the run log means."""
    entry = _entry("LOG-RANDOM-DIAGNOSTIC")
    assert entry.hyperparameters["report_only"] == [True]
    assert any("selection metric" in item for item in entry.must_hold_constant)
    assert "never as training data" in " ".join(entry.steps)


def test_every_backlog_entry_builds_and_validates_against_the_catalog():
    """Including the two citations added for the new directions."""
    agent = OfflineResearchAgent(convergence=ConvergenceConfig(max_iterations=50))
    context = build_research_context([_baseline(iteration=0, primary=0.60)], agent.convergence)
    for entry in DEFAULT_BACKLOG:
        validate_proposal_citations(entry.build(context), agent.citation_source)
