"""Deterministic, history-adaptive query planning for Agent 1."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from agent.research.context import ResearchContext
from agent.research.retrieval.memory import ResearchMemory, classify_method_families
from agent.research.retrieval.models import (
    DatasetProfile,
    QueryPlan,
    ResearchIntent,
    ResearchQuery,
    RetrievalBudget,
    deterministic_fingerprint,
)
from agent.research.retrieval.safety import ResearchSafetyScanner


_FAMILY_TERMS = {
    "pairwise_loss": "pairwise ranking loss",
    "listwise_loss": "listwise top-k ranking loss",
    "pointwise_loss": "pointwise classification objective",
    "sequence_modeling": "behavior sequence modeling",
    "feature_interaction": "factorization and feature-interaction model",
    "multi_task": "multi-task learning",
    "watch_time": "watch-time supervision",
    "debiasing": "exposure debiasing",
    "sampling": "ranking-aware sampling",
    "calibration": "score calibration",
    "regularization": "regularization",
    "temporal": "temporal preference modeling",
    "ensemble": "model ensembling",
    "context_features": "exposure-context features",
    "content_features": "content-side features",
    "top_k_reranking": "top-k reranking",
    "unknown": "current incumbent architecture",
}

_EXPLORATION_ORDER = (
    "debiasing",
    "sequence_modeling",
    "multi_task",
    "watch_time",
    "temporal",
    "calibration",
    "content_features",
    "ensemble",
)


def build_context_fingerprint(
    context: ResearchContext,
    profile: DatasetProfile,
) -> str:
    """Fingerprint only authoritative context and versioned dataset metadata."""
    return deterministic_fingerprint({
        "dataset_profile_id": profile.profile_id,
        "dataset_profile_fingerprint": profile.fingerprint,
        "research_context": context.to_prompt_dict(),
    })


@dataclass(frozen=True)
class _Candidate:
    intent: ResearchIntent
    text: str
    rationale: str
    method_families: tuple[str, ...]
    priority: int


class QueryPlanner:
    """Produce bounded, distinct queries from validation-only Research state."""

    def __init__(self, *, scanner: Optional[ResearchSafetyScanner] = None):
        self.scanner = scanner or ResearchSafetyScanner()

    def plan(
        self,
        *,
        context: ResearchContext,
        profile: DatasetProfile,
        memory: ResearchMemory,
        budget: RetrievalBudget = RetrievalBudget(),
    ) -> QueryPlan:
        memory.assert_matches(context.history_fingerprint)
        self.scanner.scan_value(profile, origin="Research dataset profile")
        self.scanner.scan_value(
            context.to_prompt_dict(),
            origin="validation-only Research query context",
        )
        context_id = build_context_fingerprint(context, profile)
        if context.remaining_iterations <= 0 or context.remaining_wall_s <= 0:
            plan = QueryPlan(
                schema_version=1,
                context_fingerprint=context_id,
                queries=(),
                tight_budget=True,
                suppressed_query_signatures=(),
                retrieval_wall_s=0.0,
            )
            budget.validate_plan(plan)
            return plan

        tight_budget = context.remaining_iterations <= 3 or context.remaining_wall_s <= 3600.0
        retrieval_wall_s = min(budget.max_retrieval_wall_s, context.remaining_wall_s)
        candidates = self._candidates(context, profile, memory, tight_budget)

        distinct: list[_Candidate] = []
        seen_text: set[str] = set()
        for candidate in sorted(
            candidates,
            key=lambda item: (-item.priority, item.intent.value, item.text.casefold()),
        ):
            text = self._fit_query(candidate.text, budget.max_query_chars)
            normalized = " ".join(text.casefold().split())
            if normalized in seen_text:
                continue
            seen_text.add(normalized)
            distinct.append(_Candidate(
                intent=candidate.intent,
                text=text,
                rationale=candidate.rationale,
                method_families=candidate.method_families,
                priority=candidate.priority,
            ))

        available: list[tuple[_Candidate, ResearchQuery]] = []
        suppressed: list[str] = []
        for candidate in distinct:
            text = candidate.text
            if tight_budget:
                text = self._fit_query(
                    text + " lightweight CPU-efficient low-complexity bounded-runtime",
                    budget.max_query_chars,
                )
            query = ResearchQuery.create(
                intent=candidate.intent,
                text=text,
                rationale=candidate.rationale,
                method_families=candidate.method_families,
                priority=candidate.priority,
                max_results=1,
                cheap_only=tight_budget,
            )
            if memory.has_query(context_id, query.signature):
                suppressed.append(query.signature)
                continue
            self.scanner.scan_value(query, origin=f"planned Research query {query.query_id}")
            available.append((candidate, query))

        max_query_count = min(budget.max_queries, budget.max_total_results)
        selected_pairs = self._select_with_exploration(available, max_query_count)
        per_query_results = min(
            budget.max_results_per_query,
            max(1, budget.max_total_results // len(selected_pairs)),
        ) if selected_pairs else 1
        queries = tuple(
            ResearchQuery.create(
                intent=query.intent,
                text=query.text,
                rationale=query.rationale,
                method_families=query.method_families,
                priority=query.priority,
                max_results=per_query_results,
                cheap_only=query.cheap_only,
            )
            for _, query in selected_pairs
        )

        plan = QueryPlan(
            schema_version=1,
            context_fingerprint=context_id,
            queries=queries,
            tight_budget=tight_budget,
            suppressed_query_signatures=tuple(sorted(suppressed)),
            retrieval_wall_s=retrieval_wall_s if queries else 0.0,
        )
        budget.validate_plan(plan)
        return plan

    def _candidates(
        self,
        context: ResearchContext,
        profile: DatasetProfile,
        memory: ResearchMemory,
        tight_budget: bool,
    ) -> list[_Candidate]:
        dataset = profile.dataset_name
        label = profile.label
        task = profile.task
        candidates = [
            _Candidate(
                ResearchIntent.DATASET_SPECIFIC,
                f"{dataset} {label} {task} recommender methods logged impressions",
                "Find methods whose assumptions match the current dataset and ranking task.",
                ("dataset_specific",),
                82,
            ),
            _Candidate(
                ResearchIntent.OBJECTIVE_ALIGNMENT,
                f"{label} implicit-feedback objective alignment watch behavior within-user ranking",
                "Align training supervision with the long-view ranking objective.",
                ("watch_time", "ranking_objective"),
                84,
            ),
            _Candidate(
                ResearchIntent.GAUC_IMPROVEMENT,
                "GAUC group AUC optimization recommender within-user ranking robust user weighting",
                "Investigate broad within-user ordering improvements for GAUC.",
                ("sampling", "ranking_objective"),
                78,
            ),
            _Candidate(
                ResearchIntent.NDCG_IMPROVEMENT,
                "nDCG@5 top-of-list recommendation listwise learning top-k ranking quality",
                "Investigate methods targeted at the highest-ranked impressions.",
                ("listwise_loss", "top_k_reranking"),
                79,
            ),
        ]

        incumbent_families = self._incumbent_families(context)
        incumbent_terms = " and ".join(_FAMILY_TERMS.get(item, item) for item in incumbent_families[:2])
        candidates.append(_Candidate(
            ResearchIntent.INCUMBENT_EXTENSION,
            f"{incumbent_terms} compatible extension {label} ranking GAUC nDCG@5",
            "Prefer an attributable extension compatible with the accepted incumbent.",
            incumbent_families,
            80,
        ))

        latest_accept = self._latest(
            memory,
            decision="accept",
            incumbent_iteration=context.parent_iteration,
        )
        if latest_accept is not None:
            terms = " and ".join(
                _FAMILY_TERMS.get(item, item) for item in latest_accept.method_families[:2]
            )
            candidates.append(_Candidate(
                ResearchIntent.SUCCESS_FOLLOW_UP,
                f"compatible follow-up improvements after successful {terms} recommender experiment",
                f"Iteration {latest_accept.iteration} was accepted; search for a compatible next mechanism.",
                latest_accept.method_families,
                91,
            ))

        latest_revert = self._latest(
            memory,
            decision="revert",
            incumbent_iteration=context.parent_iteration,
        )
        if latest_revert is not None:
            if "pairwise_loss" in latest_revert.method_families:
                text = (
                    "non-pairwise alternative ranking mechanisms listwise top-k calibration "
                    "for implicit-feedback recommendation"
                )
                families = ("listwise_loss", "top_k_reranking", "calibration")
            else:
                failed_terms = " and ".join(
                    _FAMILY_TERMS.get(item, item) for item in latest_revert.method_families[:2]
                )
                text = f"alternative mechanisms after unsuccessful {failed_terms} recommender experiment"
                families = tuple(
                    family for family in _EXPLORATION_ORDER
                    if family not in latest_revert.method_families
                )[:3] or ("ranking_objective",)
            candidates.append(_Candidate(
                ResearchIntent.FAILED_ALTERNATIVE,
                text,
                f"Iteration {latest_revert.iteration} was reverted; avoid repeating its mechanism unchanged.",
                families,
                98,
            ))

        latest_abandon = self._latest(
            memory,
            decision="abandon",
            include_abandoned_status=True,
            incumbent_iteration=context.parent_iteration,
        )
        if latest_abandon is not None:
            terms = " and ".join(
                _FAMILY_TERMS.get(item, item) for item in latest_abandon.method_families[:2]
            )
            candidates.append(_Candidate(
                ResearchIntent.ABANDONED_FEASIBILITY,
                f"lightweight simpler alternatives to {terms} recommendation method CPU efficient",
                f"Iteration {latest_abandon.iteration} was abandoned; investigate a feasible variant.",
                latest_abandon.method_families,
                101 if tight_budget else 96,
            ))

        active_gaps = memory.active_gaps_for_incumbent(context.parent_iteration)
        if any(gap.kind == "top_list_quality" for gap in active_gaps):
            candidates.append(_Candidate(
                ResearchIntent.NDCG_IMPROVEMENT,
                "preserve GAUC improvement recover nDCG@5 top-of-list quality listwise reranking",
                "Validation behavior improved GAUC while reducing nDCG@5.",
                ("listwise_loss", "top_k_reranking"),
                110,
            ))
        if any(gap.kind == "within_user_consistency" for gap in active_gaps):
            candidates.append(_Candidate(
                ResearchIntent.GAUC_IMPROVEMENT,
                "preserve nDCG@5 improvement recover GAUC within-user ordering sampling calibration",
                "Validation behavior improved nDCG@5 while reducing GAUC.",
                ("sampling", "calibration"),
                109,
            ))

        diagnostic = self._diagnostic_candidate(memory)
        if diagnostic is not None:
            candidates.append(diagnostic)

        unused_signal = next(
            (
                signal for signal in profile.available_signals
                if signal.method_family not in memory.attempted_method_families
            ),
            None,
        )
        if unused_signal is not None:
            fields = " ".join(unused_signal.fields[:4])
            candidates.append(_Candidate(
                ResearchIntent.UNUSED_SIGNAL,
                f"{dataset} {unused_signal.family} {fields} features for {label} recommendation ranking",
                f"The {unused_signal.family} signal family has not appeared in attempted methods.",
                (unused_signal.method_family,),
                76,
            ))

        exploration_family = next(
            (
                family for family in _EXPLORATION_ORDER
                if family not in memory.attempted_method_families
            ),
            "causal_representation",
        )
        candidates.append(_Candidate(
            ResearchIntent.EXPLORATION,
            f"{_FAMILY_TERMS.get(exploration_family, exploration_family)} novel direction for {label} recommendation",
            "Retain one method-family-diverse query outside already attempted directions.",
            (exploration_family,),
            45,
        ))

        if tight_budget:
            candidates.append(_Candidate(
                ResearchIntent.ABANDONED_FEASIBILITY,
                "cheap low-risk parameter-efficient recommender improvements no architecture rewrite",
                "Remaining iterations or wall-clock time require inexpensive experiments.",
                ("regularization", "sampling", "calibration"),
                106,
            ))
        return candidates

    @staticmethod
    def _latest(
        memory: ResearchMemory,
        *,
        decision: str,
        include_abandoned_status: bool = False,
        incumbent_iteration: Optional[int] = None,
    ):
        matching = [
            experiment for experiment in memory.experiments.values()
            if (
                experiment.decision == decision
                or (include_abandoned_status and experiment.status == "abandoned")
            )
            and (
                incumbent_iteration is None
                or memory.is_branch_comparable(experiment.iteration, incumbent_iteration)
            )
        ]
        return max(matching, key=lambda item: item.iteration) if matching else None

    @staticmethod
    def _incumbent_families(context: ResearchContext) -> tuple[str, ...]:
        if context.incumbent is None:
            return ("feature_interaction",)
        return classify_method_families(context.incumbent.hypothesis)

    @staticmethod
    def _diagnostic_candidate(memory: ResearchMemory) -> Optional[_Candidate]:
        diagnostics: list[str] = []
        for iteration in sorted(memory.experiments, reverse=True):
            diagnostics.extend(memory.experiments[iteration].evaluator_diagnostics)
            if len(diagnostics) >= 5:
                break
        normalized = " ".join(diagnostics).casefold()
        patterns = (
            (
                ("timeout", "runtime", "slow", "oom", "memory"),
                "efficient bounded-memory recommendation training",
                ("regularization", "sampling"),
            ),
            (
                ("overfit", "variance", "unstable"),
                "recommender regularization robust validation generalization",
                ("regularization",),
            ),
            (
                ("ndcg", "top-k", "top of list"),
                "nDCG@5 top-of-list listwise recommender improvement",
                ("listwise_loss", "top_k_reranking"),
            ),
            (
                ("gauc", "group auc"),
                "GAUC within-user ranking weighting recommender improvement",
                ("sampling", "calibration"),
            ),
            (
                ("sparse", "cold start"),
                "sparse-feedback recommender side-information representation",
                ("content_features", "context_features"),
            ),
        )
        for keywords, text, families in patterns:
            if any(keyword in normalized for keyword in keywords):
                return _Candidate(
                    ResearchIntent.EVALUATOR_DIAGNOSTIC,
                    text,
                    "Evaluator diagnostics identify this failure or metric pattern.",
                    families,
                    103,
                )
        return None

    @staticmethod
    def _select_with_exploration(
        candidates: Sequence[tuple[_Candidate, ResearchQuery]],
        limit: int,
    ) -> list[tuple[_Candidate, ResearchQuery]]:
        if limit <= 0:
            return []
        exploration = next(
            (pair for pair in candidates if pair[0].intent == ResearchIntent.EXPLORATION),
            None,
        )
        if limit == 1 or exploration is None:
            return list(candidates[:limit])
        selected = [
            pair for pair in candidates
            if pair[0].intent != ResearchIntent.EXPLORATION
        ][:limit - 1]
        selected.append(exploration)
        return selected

    @staticmethod
    def _fit_query(text: str, max_chars: int) -> str:
        compact = " ".join(text.split())
        if len(compact) <= max_chars:
            return compact
        clipped = compact[:max_chars].rsplit(" ", 1)[0].rstrip()
        return clipped or compact[:max_chars]
