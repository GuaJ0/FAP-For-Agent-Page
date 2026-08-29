"""Phase 4A.2 tests: adaptive planning and authoritative memory reconciliation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from agent.config import ConvergenceConfig
from agent.records import AggregateMetrics, Decision, Event, ResourceUsage, RunRecord, Status
from agent.research.context import build_research_context
from agent.research.retrieval import (
    GapStatus,
    QueryPlanner,
    ResearchIntent,
    ResearchMemory,
    RetrievalBudget,
    StaleResearchMemoryError,
    classify_method_families,
    load_dataset_profile,
)


def _handoff(
    hypothesis_id: str,
    hypothesis: str,
    *,
    implementation: str,
    source_id: str = "paper-one",
) -> str:
    return (
        "[RESEARCH_PROPOSAL v1]\n"
        f"ID: {hypothesis_id}\n"
        "TITLE: Planned experiment\n"
        "PARENT ITERATION: 0\n\n"
        "HYPOTHESIS:\n"
        f"{hypothesis}\n\n"
        "EVIDENCE:\n"
        f"- [{source_id}/supported-claim] Applicable method evidence.\n\n"
        "IMPLEMENTATION:\n"
        f"1. {implementation}\n\n"
        "TARGET COMPONENTS:\n"
        "- training objective\n\n"
        "HYPERPARAMETERS:\n"
        "- strength: [0.1]\n"
    )


def _record(
    iteration: int,
    *,
    parent: Optional[int],
    hypothesis: str,
    decision: Optional[Decision],
    status: Status = Status.SUCCESS,
    gauc: Optional[float] = 0.60,
    ndcg: Optional[float] = 0.60,
    diagnostic: str = "validation evidence only",
    elapsed_s: Optional[float] = None,
) -> RunRecord:
    aggregate = None if gauc is None or ndcg is None else AggregateMetrics(
        primary_mean=(gauc + ndcg) / 2,
        primary_std=0.001,
        gauc_mean=gauc,
        ndcg5_mean=ndcg,
        n_seeds=2,
    )
    seconds = iteration * 10 if elapsed_s is None else elapsed_s
    return RunRecord(
        iteration=iteration,
        parent_iteration=parent,
        timestamp=(datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)).isoformat(),
        hypothesis=hypothesis,
        diff_path=f"solutions/attempt_{iteration}/config.yaml",
        status=status,
        seeds=[],
        aggregate=aggregate,
        delta_vs_current_best=None,
        decision=decision,
        events=[Event("eval_finished", diagnostic, "evaluator")],
        resources=ResourceUsage(wall_s=10.0),
    )


def _baseline() -> RunRecord:
    return _record(
        0,
        parent=None,
        hypothesis="Factorization machine with pointwise logloss over identity and context fields.",
        decision=Decision.ACCEPT,
    )


def _plan(history, *, convergence=None, memory=None, budget=None):
    profile = load_dataset_profile()
    cfg = convergence or ConvergenceConfig(max_iterations=20, max_wall_s=20_000.0)
    context = build_research_context(history, cfg)
    research_memory = memory or ResearchMemory()
    research_memory.reconcile(history)
    plan = QueryPlanner().plan(
        context=context,
        profile=profile,
        memory=research_memory,
        budget=budget or RetrievalBudget(),
    )
    return plan, research_memory, context


def _queries_for(plan, intent):
    return [query for query in plan.queries if query.intent == intent]


def test_queries_adapt_after_accept_with_compatible_follow_up():
    accepted = _record(
        1,
        parent=0,
        hypothesis=_handoff(
            "H-PAIRWISE-ACCEPT",
            "Add a hybrid pairwise ranking loss to the pointwise incumbent.",
            implementation="Add a bounded pairwise objective term.",
        ),
        decision=Decision.ACCEPT,
        gauc=0.63,
        ndcg=0.62,
    )

    plan, _, _ = _plan([_baseline(), accepted])
    follow_ups = _queries_for(plan, ResearchIntent.SUCCESS_FOLLOW_UP)

    assert follow_ups
    assert "compatible follow-up" in follow_ups[0].text
    assert "pairwise" in follow_ups[0].text


def test_queries_adapt_after_reverted_pairwise_without_repeating_bpr():
    reverted = _record(
        1,
        parent=0,
        hypothesis=_handoff(
            "H-BPR-REVERT",
            "Replace pointwise logloss with pure BPR pairwise ranking.",
            implementation="Train on positive-negative pairs.",
        ),
        decision=Decision.REVERT,
        gauc=0.59,
        ndcg=0.58,
    )

    plan, _, _ = _plan([_baseline(), reverted])
    alternatives = _queries_for(plan, ResearchIntent.FAILED_ALTERNATIVE)

    assert alternatives
    assert "non-pairwise" in alternatives[0].text
    assert "bpr" not in alternatives[0].text.casefold()
    assert "listwise" in alternatives[0].text


def test_queries_adapt_after_abandon_toward_feasible_variants():
    abandoned = _record(
        1,
        parent=0,
        hypothesis=_handoff(
            "H-SEQUENCE-ABANDON",
            "Add a deep attention behavior sequence model.",
            implementation="Build a long candidate-conditioned behavior sequence.",
        ),
        decision=Decision.ABANDON,
        status=Status.ABANDONED,
        gauc=None,
        ndcg=None,
        diagnostic="runtime timeout during validation execution",
    )

    plan, _, _ = _plan([_baseline(), abandoned])
    feasibility = _queries_for(plan, ResearchIntent.ABANDONED_FEASIBILITY)

    assert feasibility
    assert any("lightweight" in query.text and "simpler" in query.text for query in feasibility)


def test_gauc_gain_and_ndcg_drop_prioritizes_top_of_list_research():
    divergent = _record(
        1,
        parent=0,
        hypothesis=_handoff(
            "H-DIVERGENT",
            "Change user weighting in the ranking sampler.",
            implementation="Reweight sampled users by positive activity.",
        ),
        decision=Decision.REVERT,
        gauc=0.64,
        ndcg=0.57,
    )

    plan, memory, _ = _plan([_baseline(), divergent])
    ndcg_queries = _queries_for(plan, ResearchIntent.NDCG_IMPROVEMENT)

    assert any(gap.kind == "top_list_quality" for gap in memory.unresolved_gaps.values())
    assert ndcg_queries
    assert "preserve GAUC" in ndcg_queries[0].text
    assert "recover nDCG@5" in ndcg_queries[0].text


def test_tight_budget_marks_queries_cheap_and_prioritizes_low_cost_methods():
    history = [_baseline()]
    cfg = ConvergenceConfig(max_iterations=3, max_wall_s=1200.0)

    plan, _, _ = _plan(history, convergence=cfg)

    assert plan.tight_budget
    assert plan.queries
    assert all(query.cheap_only for query in plan.queries)
    assert all("lightweight" in query.text for query in plan.queries)
    assert _queries_for(plan, ResearchIntent.ABANDONED_FEASIBILITY)


def test_plan_contains_distinct_intents_unused_signals_and_exploration():
    plan, _, _ = _plan([_baseline()])

    signatures = [query.signature for query in plan.queries]
    assert len(signatures) == len(set(signatures))
    expected_intents = {
        ResearchIntent.DATASET_SPECIFIC,
        ResearchIntent.OBJECTIVE_ALIGNMENT,
        ResearchIntent.GAUC_IMPROVEMENT,
        ResearchIntent.NDCG_IMPROVEMENT,
        ResearchIntent.INCUMBENT_EXTENSION,
        ResearchIntent.SUCCESS_FOLLOW_UP,
        ResearchIntent.UNUSED_SIGNAL,
        ResearchIntent.EXPLORATION,
    }
    assert expected_intents <= {query.intent for query in plan.queries}


def test_identical_queries_are_suppressed_for_unchanged_context(tmp_path):
    history = [_baseline()]
    memory = ResearchMemory(path=tmp_path / "memory.json")
    first, memory, context = _plan(history, memory=memory)
    memory.remember_query_plan(first)
    reloaded = ResearchMemory.load(tmp_path / "memory.json")
    reloaded.reconcile(history)

    second = QueryPlanner().plan(
        context=context,
        profile=load_dataset_profile(),
        memory=reloaded,
        budget=RetrievalBudget(),
    )

    assert first.queries
    assert second.queries == ()
    assert set(second.suppressed_query_signatures) == {
        query.signature for query in first.queries
    }


def test_memory_reconciliation_uses_runrecord_as_authority_and_persists_enrichment(tmp_path):
    baseline = _baseline()
    accepted = _record(
        1,
        parent=0,
        hypothesis=_handoff(
            "H-ACCEPT",
            "Add a hybrid pairwise loss.",
            implementation="Add a pairwise objective.",
            source_id="accepted-paper",
        ),
        decision=Decision.ACCEPT,
        gauc=0.62,
        ndcg=0.61,
    )
    reverted = _record(
        2,
        parent=1,
        hypothesis=_handoff(
            "H-REVERT",
            "Add a long behavior sequence attention encoder.",
            implementation="Encode recent history with attention.",
            source_id="sequence-paper",
        ),
        decision=Decision.REVERT,
        gauc=0.61,
        ndcg=0.60,
    )
    abandoned = _record(
        3,
        parent=1,
        hypothesis=_handoff(
            "H-ABANDON",
            "Add a large multi-task auxiliary tower.",
            implementation="Train multiple auxiliary heads.",
            source_id="multitask-paper",
        ),
        decision=Decision.ABANDON,
        status=Status.ABANDONED,
        gauc=None,
        ndcg=None,
    )
    memory_path = tmp_path / "research_memory.json"
    memory = ResearchMemory(path=memory_path)

    memory.reconcile([baseline, accepted, reverted, abandoned])
    memory.remember_evidence(["external-source"])
    reloaded = ResearchMemory.load(memory_path)

    assert set(reloaded.experiments) == {0, 1, 2, 3}
    assert reloaded.experiments[1].decision == "accept"
    assert reloaded.experiments[2].decision == "revert"
    assert reloaded.experiments[3].decision == "abandon"
    assert len(reloaded.attempted_method_fingerprints) == 4
    assert {"accepted-paper", "sequence-paper", "multitask-paper", "external-source"} <= (
        reloaded.evidence_source_ids
    )
    assert {gap.kind for gap in reloaded.unresolved_gaps.values()} >= {
        "accepted_follow_up",
        "reverted_alternative",
        "abandoned_feasibility",
    }

    # Removing records from authoritative history removes their outcome state,
    # while separately remembered evidence remains enrichment.
    reloaded.reconcile([baseline, accepted])
    assert set(reloaded.experiments) == {0, 1}
    assert "external-source" in reloaded.evidence_source_ids
    assert not any(gap.source_iteration == 3 for gap in reloaded.unresolved_gaps.values())


def test_evaluator_diagnostics_generate_targeted_query():
    record = _record(
        1,
        parent=0,
        hypothesis=_handoff(
            "H-SLOW",
            "Add a candidate-conditioned sequence encoder.",
            implementation="Build a recent-history attention layer.",
        ),
        decision=Decision.ABANDON,
        status=Status.ABANDONED,
        gauc=None,
        ndcg=None,
        diagnostic="OOM and runtime timeout indicate excessive memory use",
    )

    plan, _, _ = _plan([_baseline(), record])
    diagnostic_queries = _queries_for(plan, ResearchIntent.EVALUATOR_DIAGNOSTIC)

    assert diagnostic_queries
    assert "bounded-memory" in diagnostic_queries[0].text


def test_stale_and_loaded_memory_must_be_reconciled_before_planning(tmp_path):
    history = [_baseline()]
    context = build_research_context(history)
    profile = load_dataset_profile()
    memory = ResearchMemory(path=tmp_path / "memory.json")

    with pytest.raises(StaleResearchMemoryError, match="reconcile"):
        QueryPlanner().plan(context=context, profile=profile, memory=memory)

    memory.reconcile(history)
    QueryPlanner().plan(context=context, profile=profile, memory=memory)
    loaded = ResearchMemory.load(tmp_path / "memory.json")
    with pytest.raises(StaleResearchMemoryError, match="unreconciled"):
        QueryPlanner().plan(context=context, profile=profile, memory=loaded)

    loaded.reconcile(history)
    QueryPlanner().plan(context=context, profile=profile, memory=loaded)


def test_memory_reconciled_to_different_history_cannot_plan():
    baseline = _baseline()
    memory = ResearchMemory()
    memory.reconcile([baseline])
    changed = _record(
        1,
        parent=0,
        hypothesis="A different accepted pointwise experiment.",
        decision=Decision.ACCEPT,
        gauc=0.62,
        ndcg=0.62,
    )
    changed_context = build_research_context([baseline, changed])

    with pytest.raises(StaleResearchMemoryError, match="does not match"):
        QueryPlanner().plan(
            context=changed_context,
            profile=load_dataset_profile(),
            memory=memory,
        )


def test_loaded_memory_from_different_run_is_cleared_during_reconcile(tmp_path):
    path = tmp_path / "memory.json"
    first_history = [_baseline()]
    memory = ResearchMemory(path=path)
    first_plan, memory, _ = _plan(first_history, memory=memory)
    memory.remember_query_plan(first_plan)
    memory.remember_evidence(["run-one-only-source"])

    loaded = ResearchMemory.load(path)
    second_baseline = _record(
        0,
        parent=None,
        hypothesis="Independent run baseline with calibrated pointwise scores.",
        decision=Decision.ACCEPT,
    )
    second_history = [second_baseline]
    loaded.reconcile(second_history)

    assert "run-one-only-source" not in loaded.evidence_source_ids
    assert loaded.query_history == {}
    plan = QueryPlanner().plan(
        context=build_research_context(second_history),
        profile=load_dataset_profile(),
        memory=loaded,
    )
    assert plan.queries


def test_metric_divergence_gap_resolves_after_both_metrics_improve():
    baseline = _baseline()
    divergent = _record(
        1,
        parent=0,
        hypothesis=_handoff(
            "H-NDCG-REGRESSION",
            "Change user weighting in the sampler.",
            implementation="Reweight users by activity.",
        ),
        decision=Decision.REVERT,
        gauc=0.64,
        ndcg=0.57,
    )
    recovered = _record(
        2,
        parent=1,
        hypothesis=_handoff(
            "H-BOTH-IMPROVE",
            "Add calibrated pointwise score shaping.",
            implementation="Calibrate validation-safe pointwise scores.",
        ),
        decision=Decision.ACCEPT,
        gauc=0.65,
        ndcg=0.60,
    )

    plan, memory, _ = _plan([baseline, divergent, recovered])
    old_gap = next(
        gap for gap in memory.unresolved_gaps.values()
        if gap.kind == "top_list_quality" and gap.source_iteration == 1
    )

    assert old_gap.status == GapStatus.RESOLVED
    assert old_gap.resolved_by_iteration == 2
    assert old_gap not in memory.active_gaps
    assert not any("preserve GAUC improvement recover nDCG" in query.text for query in plan.queries)


def test_newer_metric_divergence_supersedes_older_gap_in_same_lineage():
    first = _record(
        1,
        parent=0,
        hypothesis="First sampling change.",
        decision=Decision.REVERT,
        gauc=0.64,
        ndcg=0.57,
    )
    second = _record(
        2,
        parent=1,
        hypothesis="Different user weighting change.",
        decision=Decision.REVERT,
        gauc=0.65,
        ndcg=0.56,
    )

    _, memory, _ = _plan([_baseline(), first, second])
    old_gap = next(
        gap for gap in memory.unresolved_gaps.values()
        if gap.kind == "top_list_quality" and gap.source_iteration == 1
    )
    new_gap = next(
        gap for gap in memory.unresolved_gaps.values()
        if gap.kind == "top_list_quality" and gap.source_iteration == 2
    )

    assert old_gap.status == GapStatus.SUPERSEDED
    assert old_gap.resolved_by_iteration == 2
    assert new_gap.status == GapStatus.OPEN


def test_sibling_improvement_cannot_resolve_metric_gap():
    divergent = _record(
        1,
        parent=0,
        hypothesis="Divergent sampling branch.",
        decision=Decision.REVERT,
        gauc=0.64,
        ndcg=0.57,
    )
    improving_sibling = _record(
        2,
        parent=0,
        hypothesis="Independent calibrated sibling branch.",
        decision=Decision.ACCEPT,
        gauc=0.63,
        ndcg=0.63,
    )

    _, memory, _ = _plan([_baseline(), divergent, improving_sibling])
    divergent_gap = next(
        gap for gap in memory.unresolved_gaps.values()
        if gap.kind == "top_list_quality" and gap.source_iteration == 1
    )

    assert divergent_gap.status == GapStatus.OPEN
    assert divergent_gap.resolved_by_iteration is None


def test_divergence_gap_from_non_incumbent_lineage_does_not_drive_planning():
    other_root = _record(
        10,
        parent=None,
        hypothesis="Independent low-scoring pointwise root.",
        decision=Decision.ACCEPT,
        gauc=0.50,
        ndcg=0.50,
    )
    other_divergence = _record(
        11,
        parent=10,
        hypothesis="Independent-lineage sampling experiment.",
        decision=Decision.REVERT,
        gauc=0.55,
        ndcg=0.45,
    )

    plan, memory, context = _plan([_baseline(), other_root, other_divergence])

    assert context.parent_iteration == 0
    assert any(gap.kind == "top_list_quality" for gap in memory.active_gaps)
    assert not any("preserve GAUC improvement recover nDCG" in query.text for query in plan.queries)


def test_sibling_divergence_does_not_contaminate_current_incumbent_planning():
    accepted_sibling = _record(
        1,
        parent=0,
        hypothesis="Accepted pointwise calibration branch.",
        decision=Decision.ACCEPT,
        gauc=0.63,
        ndcg=0.62,
    )
    reverted_sibling = _record(
        2,
        parent=0,
        hypothesis="Sibling sampling branch with divergent validation metrics.",
        decision=Decision.REVERT,
        gauc=0.65,
        ndcg=0.55,
    )

    plan, memory, context = _plan([_baseline(), accepted_sibling, reverted_sibling])
    sibling_gap = next(
        gap for gap in memory.unresolved_gaps.values()
        if gap.kind == "top_list_quality" and gap.source_iteration == 2
    )

    assert context.parent_iteration == 1
    assert memory.ancestor_chain(1) == (1, 0)
    assert memory.ancestor_chain(2) == (2, 0)
    assert sibling_gap not in memory.active_gaps_for_incumbent(1)
    assert not _queries_for(plan, ResearchIntent.FAILED_ALTERNATIVE)
    assert not any("preserve GAUC improvement recover nDCG" in query.text for query in plan.queries)


def test_true_descendant_can_resolve_ancestor_metric_gap():
    divergent_accept = _record(
        1,
        parent=0,
        hypothesis="Accepted user-weighting change with a top-list tradeoff.",
        decision=Decision.ACCEPT,
        gauc=0.64,
        ndcg=0.59,
    )
    intermediate_plan, intermediate_memory, _ = _plan([_baseline(), divergent_accept])
    ancestor_gap = next(
        gap for gap in intermediate_memory.active_gaps
        if gap.kind == "top_list_quality" and gap.source_iteration == 1
    )
    assert ancestor_gap in intermediate_memory.active_gaps_for_incumbent(1)
    assert any(
        "preserve GAUC improvement recover nDCG" in query.text
        for query in intermediate_plan.queries
    )

    descendant_accept = _record(
        3,
        parent=1,
        hypothesis="Compatible descendant that improves both validation metrics.",
        decision=Decision.ACCEPT,
        gauc=0.66,
        ndcg=0.62,
    )
    final_plan, final_memory, final_context = _plan(
        [_baseline(), divergent_accept, descendant_accept]
    )
    resolved_gap = next(
        gap for gap in final_memory.unresolved_gaps.values()
        if gap.kind == "top_list_quality" and gap.source_iteration == 1
    )

    assert final_context.parent_iteration == 3
    assert final_memory.ancestor_chain(3) == (3, 1, 0)
    assert resolved_gap.status == GapStatus.RESOLVED
    assert resolved_gap.resolved_by_iteration == 3
    assert resolved_gap not in final_memory.active_gaps_for_incumbent(3)
    assert not any(
        "preserve GAUC improvement recover nDCG" in query.text
        for query in final_plan.queries
    )


def test_reverse_metric_divergence_prioritizes_gauc_recovery():
    divergent = _record(
        1,
        parent=0,
        hypothesis=_handoff(
            "H-GAUC-REGRESSION",
            "Add top-k listwise score shaping.",
            implementation="Optimize a listwise top-k term.",
        ),
        decision=Decision.REVERT,
        gauc=0.57,
        ndcg=0.64,
    )

    plan, memory, _ = _plan([_baseline(), divergent])

    assert any(gap.kind == "within_user_consistency" for gap in memory.active_gaps)
    assert any(
        "preserve nDCG@5 improvement recover GAUC" in query.text
        for query in _queries_for(plan, ResearchIntent.GAUC_IMPROVEMENT)
    )


def test_zero_remaining_iterations_produces_no_retrieval_work():
    completed_research = _record(
        1,
        parent=0,
        hypothesis="Completed research experiment.",
        decision=Decision.REVERT,
    )
    plan, _, _ = _plan(
        [_baseline(), completed_research],
        convergence=ConvergenceConfig(max_iterations=1, max_wall_s=1000.0),
    )

    assert plan.queries == ()
    assert plan.retrieval_wall_s == 0.0


def test_zero_remaining_wall_time_produces_no_retrieval_work():
    plan, _, _ = _plan(
        [_baseline()],
        convergence=ConvergenceConfig(max_iterations=5, max_wall_s=0.0),
    )

    assert plan.queries == ()
    assert plan.retrieval_wall_s == 0.0


def test_retrieval_wall_budget_is_clamped_to_actual_remaining_time():
    baseline = _baseline()
    later = _record(
        1,
        parent=0,
        hypothesis="Accepted pointwise calibration.",
        decision=Decision.ACCEPT,
        elapsed_s=5.0,
    )
    plan, _, context = _plan(
        [baseline, later],
        convergence=ConvergenceConfig(max_iterations=10, max_wall_s=8.0),
        budget=RetrievalBudget(max_retrieval_wall_s=20.0),
    )

    assert context.remaining_wall_s == 3.0
    assert plan.retrieval_wall_s == 3.0


def test_suppressed_queries_are_backfilled_from_lower_ranked_candidates():
    history = [_baseline()]
    budget = RetrievalBudget(max_queries=3, max_results_per_query=2, max_total_results=6)
    first, memory, context = _plan(history, budget=budget)
    memory.remember_query_plan(first)

    second = QueryPlanner().plan(
        context=context,
        profile=load_dataset_profile(),
        memory=memory,
        budget=budget,
    )

    assert len(first.queries) == 3
    assert len(second.queries) == 3
    assert {query.signature for query in first.queries}.isdisjoint(
        query.signature for query in second.queries
    )


def test_method_classification_ignores_evidence_risks_and_keep_constant_sections():
    handoff = _handoff(
        "H-POINTWISE-ONLY",
        "Adjust the pointwise logloss weighting.",
        implementation="Change only the pointwise objective weights.",
    ) + (
        "\nKEEP CONSTANT:\n- sequence attention and BPR remain unchanged.\n"
        "\nRISKS:\n- multi-task learning may be unstable.\n"
    )

    families = classify_method_families(handoff)

    assert "pointwise_loss" in families
    assert "pairwise_loss" not in families
    assert "sequence_modeling" not in families
    assert "multi_task" not in families
