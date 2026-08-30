"""Production LLM-backed implementation of the existing ResearchAgent protocol.

This module owns only Agent 1 concerns.  It consumes validation-only
``RunRecord`` history and returns the existing ``Idea`` handoff; it does not
route the pipeline, implement model code, or make evaluation decisions.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from agent.agents import Idea
from agent.coding.llm import LLMClient, LLMResponse
from agent.config import (
    DEFAULT_CONFIG,
    FORBIDDEN_PAYLOAD_KEYS,
    TEST_METRICS_SENTINEL,
    ConvergenceConfig,
)
from agent.records import Decision, RunRecord, Status
from agent.research.breadth import (
    BreadthCandidate,
    BreadthPlan,
    BreadthValidationError,
    MAX_BREADTH_CANDIDATES,
    MIN_BREADTH_CANDIDATES,
    MIN_BREADTH_SURVIVORS,
    build_stack_coverage,
    filter_breadth_candidates,
    normalize_research_text,
    parse_breadth_candidates_individually,
    rank_breadth_candidates,
    research_text_similarity,
    validate_depth_alignment,
    validate_research_direction_safety,
)
from agent.research.citations import (
    CitationSource,
    CitationValidationError,
    JsonCitationCatalog,
    validate_proposal_citations,
)
from agent.research.context import ResearchContext, build_research_context
from agent.research.prompts import (
    BREADTH_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_breadth_prompt,
    build_breadth_repair_prompt,
    build_proposal_prompt,
    build_repair_prompt,
)
from agent.research.schemas import ProposalValidationError, ResearchProposal
from runlog.emit import append_line, read_lines

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BREADTH_CITATION_LIMIT = 10


class ResearchAgentError(RuntimeError):
    """Base error raised by the production Research Agent."""


class ResearchInputError(ResearchAgentError):
    """Agent-facing history is unsafe or insufficient to construct a prompt."""


class ResearchOutputError(ResearchAgentError):
    """The model failed to produce a valid proposal within the repair budget."""


class DuplicateHypothesisError(ProposalValidationError):
    """An attempted hypothesis or hypothesis ID was proposed again unchanged."""


@dataclass
class ResearchUsageLog:
    """Append-only accounting for Research calls only.

    The Coding Agent's UsageLog intentionally remains untouched.  Token counts
    and estimated cost are copied from the shared ``LLMResponse`` abstraction.
    """

    path: Path

    def record(
        self,
        response: LLMResponse,
        *,
        purpose: str,
        attempt: int,
        parent_iteration: Optional[int],
    ) -> None:
        append_line(self.path, {
            "timestamp": time.time(),
            "agent": "research",
            "purpose": purpose,
            "attempt": attempt,
            "parent_iteration": parent_iteration,
            "model": response.model,
            "is_real_model_call": response.is_real_model_call,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "cost_usd": round(response.cost_usd, 6),
        })

    def totals(self) -> dict[str, Any]:
        rows = list(read_lines(self.path))
        return {
            "calls": len(rows),
            "real_model_calls": sum(bool(row.get("is_real_model_call")) for row in rows),
            "tokens_in": sum(int(row.get("tokens_in", 0)) for row in rows),
            "tokens_out": sum(int(row.get("tokens_out", 0)) for row in rows),
            "cost_usd": round(sum(float(row.get("cost_usd", 0.0)) for row in rows), 6),
        }


def _normalise(text: str) -> str:
    return normalize_research_text(text)


def _section(text: str, heading: str) -> Optional[str]:
    """Extract a section from the deterministic Phase 1 handoff format."""
    match = re.search(
        rf"(?ms)^{re.escape(heading)}:\s*\n(.*?)(?=\n[A-Z][A-Z ]+:\s*\n|\Z)",
        text,
    )
    return match.group(1).strip() if match else None


def _historical_hypothesis(record: RunRecord) -> str:
    return _section(record.hypothesis, "HYPOTHESIS") or record.hypothesis


def _historical_hypothesis_id(record: RunRecord) -> Optional[str]:
    if not record.hypothesis.lstrip().startswith("[RESEARCH_PROPOSAL v1]"):
        return None
    match = re.search(r"(?m)^ID:\s*([^\r\n]+?)\s*$", record.hypothesis)
    return match.group(1) if match else None


def _was_attempted(record: RunRecord) -> bool:
    return record.decision in {Decision.ACCEPT, Decision.REVERT, Decision.ABANDON} or (
        record.status == Status.ABANDONED
    )


def _meaningful_variation(proposal: ResearchProposal, record: RunRecord) -> bool:
    """Conservative deterministic check for a changed experiment.

    For a structured historical handoff, changed implementation steps or
    hyperparameters can establish a variation.  For legacy free-text history,
    the variation must be explicit in the new hypothesis itself because the
    old record contains no implementation detail to compare safely.
    """
    current_hypothesis = _normalise(proposal.hypothesis)
    prior_hypothesis = _normalise(_historical_hypothesis(record))
    if not prior_hypothesis:
        return True

    similarity = research_text_similarity(current_hypothesis, prior_hypothesis)
    if similarity < 0.92:
        return True

    if not record.hypothesis.lstrip().startswith("[RESEARCH_PROPOSAL v1]"):
        # Legacy records cannot prove that an unchanged hypothesis carries a
        # genuinely different implementation. Require the variation in the
        # hypothesis wording itself.
        current_tokens = set(current_hypothesis.split())
        prior_tokens = set(prior_hypothesis.split())
        return len(current_tokens - prior_tokens) >= 3

    variation_text = " ".join([
        *proposal.implementation.target_components,
        *proposal.implementation.steps,
        json.dumps(proposal.implementation.hyperparameters, sort_keys=True),
    ])
    prior_text = _normalise(record.hypothesis)
    variation_tokens = set(_normalise(variation_text).split())
    return len(variation_tokens - set(prior_text.split())) >= 3


def _reject_duplicate(proposal: ResearchProposal, history: Sequence[RunRecord]) -> None:
    for record in history:
        if not _was_attempted(record):
            continue
        prior = _historical_hypothesis(record)
        similarity = research_text_similarity(proposal.hypothesis, prior)
        if similarity >= 0.92 and not _meaningful_variation(proposal, record):
            outcome = record.decision.value if record.decision is not None else record.status.value
            raise DuplicateHypothesisError(
                f"proposal duplicates {outcome} iteration {record.iteration} "
                "without a meaningful mechanism, implementation, or hyperparameter variation"
            )


def _validate_proposal_against_context(
    proposal: ResearchProposal,
    context: ResearchContext,
    history: Sequence[RunRecord],
) -> None:
    if proposal.parent_iteration != context.parent_iteration:
        raise ProposalValidationError(
            "proposal.parent_iteration must equal the current accepted incumbent "
            f"({context.parent_iteration!r}), got {proposal.parent_iteration!r}"
        )

    available_iterations = {item.iteration for item in context.iterations}
    unavailable = sorted(set(proposal.rationale.prior_results_used) - available_iterations)
    if unavailable:
        raise ProposalValidationError(
            "proposal.rationale.prior_results_used refers to iterations not available "
            f"in Research history: {unavailable}"
        )

    if not math.isclose(
        proposal.evaluation.minimum_primary_delta,
        context.minimum_meaningful_delta,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ProposalValidationError(
            "proposal.evaluation.minimum_primary_delta must equal the configured "
            "minimum meaningful improvement threshold "
            f"({context.minimum_meaningful_delta:g}); the schema has no explicit "
            "justification field for overriding it"
        )

    proposed_id = proposal.hypothesis_id.casefold()
    for record in history:
        if not _was_attempted(record):
            continue
        prior_id = _historical_hypothesis_id(record)
        if prior_id is not None and prior_id.casefold() == proposed_id:
            raise DuplicateHypothesisError(
                f"proposal.hypothesis_id {proposal.hypothesis_id!r} was already used by "
                f"attempted iteration {record.iteration}"
            )

    _reject_duplicate(proposal, history)


_UNSAFE_TEXT_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    re.escape(TEST_METRICS_SENTINEL),
    r"\bhidden[\s_-]*test\b",
    r"\btest[\s_-]*(?:primary|gauc|ndcg5|metrics)\b",
    r"\b(?:primary|gauc|ndcg5)[\s_-]*test\b",
))


def _assert_validation_only_context(context: ResearchContext) -> None:
    """Fail closed if prohibited split information reached a prompt field."""
    payload = json.dumps(context.to_prompt_dict(), sort_keys=True)
    lowered = payload.lower()
    for key in FORBIDDEN_PAYLOAD_KEYS:
        if key in lowered:
            raise ResearchInputError(
                f"research context contains hidden-test material (forbidden key {key!r})"
            )
    for pattern in _UNSAFE_TEXT_PATTERNS:
        if pattern.search(payload):
            raise ResearchInputError(
                "research context contains hidden-test material; refusing to construct an LLM prompt"
            )


@dataclass
class LLMResearchAgent:
    """ResearchAgent with bounded breadth selection before detailed depth."""

    llm: LLMClient
    citation_source: CitationSource = field(default_factory=JsonCitationCatalog)
    usage_log_path: Path = field(
        default_factory=lambda: REPO_ROOT / "logs" / "research_agent_usage.jsonl"
    )
    max_repair_attempts: int = 1
    citation_limit: int = 20
    convergence: ConvergenceConfig = DEFAULT_CONFIG.convergence
    breadth_candidate_count: int = 5

    last_usage: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.max_repair_attempts not in (0, 1):
            raise ValueError("max_repair_attempts must be 0 or 1")
        if self.citation_limit < 1:
            raise ValueError("citation_limit must be positive")
        if not MIN_BREADTH_CANDIDATES <= self.breadth_candidate_count <= MAX_BREADTH_CANDIDATES:
            raise ValueError(
                "breadth_candidate_count must be between "
                f"{MIN_BREADTH_CANDIDATES} and {MAX_BREADTH_CANDIDATES}"
            )
        self.usage_log_path = Path(self.usage_log_path)
        self.usage = ResearchUsageLog(self.usage_log_path)

    def propose(self, history: list[RunRecord]) -> Idea:
        """Return one validated, evidence-backed proposal as the existing Idea."""
        context = build_research_context(history, self.convergence)
        _assert_validation_only_context(context)

        query = self._citation_query(context)
        citations = self.citation_source.search(query, limit=self.citation_limit)
        if not citations:
            raise ResearchInputError(
                "the configured CitationSource returned no evidence for the Research prompt"
            )

        calls: list[LLMResponse] = []
        coverage = build_stack_coverage(history)
        try:
            selected = self._generate_breadth(
                context=context,
                history=history,
                citations=citations[:BREADTH_CITATION_LIMIT],
                coverage=coverage,
                calls=calls,
            )
            proposal = self._generate_depth(
                context=context,
                history=history,
                citations=citations,
                selected=selected,
                calls=calls,
            )
        except ResearchAgentError:
            self._set_last_usage(calls)
            raise

        self._set_last_usage(calls)
        return Idea(
            hypothesis=proposal.to_handoff_text(),
            # Use the selected incumbent from trusted context, rather than
            # trusting the model to choose its own parent.
            parent_iteration=context.parent_iteration,
        )

    def _complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        purpose: str,
        attempt: int,
        context: ResearchContext,
        calls: list[LLMResponse],
    ) -> LLMResponse:
        try:
            response = self.llm.complete(system_prompt, user_prompt, purpose=purpose)
        except Exception as exc:
            raise ResearchAgentError(
                f"Research Agent LLM call failed during {purpose}: {exc}"
            ) from exc
        calls.append(response)
        self.usage.record(
            response,
            purpose=purpose,
            attempt=attempt,
            parent_iteration=context.parent_iteration,
        )
        return response

    def _generate_breadth(
        self,
        *,
        context: ResearchContext,
        history: Sequence[RunRecord],
        citations,
        coverage,
        calls: list[LLMResponse],
    ) -> BreadthCandidate:
        initial_prompt = build_breadth_prompt(
            context,
            citations,
            coverage,
            max_candidates=self.breadth_candidate_count,
        )
        errors: list[str] = []
        response = self._complete(
            system_prompt=BREADTH_SYSTEM_PROMPT,
            user_prompt=initial_prompt,
            purpose="research_breadth",
            attempt=0,
            context=context,
            calls=calls,
        )
        retained: tuple[BreadthCandidate, ...] = ()
        rejections = ()
        initial_candidate_ids: set[str] = set()
        initial_count = 0
        try:
            # Keep individually valid work from a sound JSON envelope. A
            # malformed top level still has no trustworthy candidates to keep.
            parsed = parse_breadth_candidates_individually(
                response.text,
                min_candidates=1,
                max_candidates=self.breadth_candidate_count,
            )
            initial_count = parsed.raw_candidate_count
            initial_candidate_ids = set(parsed.reserved_candidate_ids)
            retained, filter_rejections = filter_breadth_candidates(
                parsed.plan,
                history=history,
                citation_source=self.citation_source,
            )
            rejections = (*parsed.rejections, *filter_rejections)
            if (
                initial_count == self.breadth_candidate_count
                and not parsed.rejections
                and len(retained) >= MIN_BREADTH_SURVIVORS
            ):
                ranked = rank_breadth_candidates(
                    retained,
                    history=history,
                    coverage=coverage,
                    context=context,
                    citation_source=self.citation_source,
                )
                return ranked[0].candidate
            detail = "; ".join(
                f"{item.candidate_id}: {item.reason}" for item in rejections
            ) or "no candidate-specific rejection"
            count_detail = (
                f"configured batch requires exactly {self.breadth_candidate_count}; "
                f"model returned {initial_count}. "
                if initial_count != self.breadth_candidate_count
                else ""
            )
            raise BreadthValidationError(
                count_detail
                + "breadth requires at least "
                f"{MIN_BREADTH_SURVIVORS} post-filter survivors; got "
                f"{len(retained)}: {detail}"
            )
        except BreadthValidationError as exc:
            errors.append(str(exc))
            if self.max_repair_attempts == 0:
                raise ResearchOutputError(
                    "Research Agent breadth phase failed after 1 call(s): "
                    f"attempt 1: {exc}"
                ) from exc

        replacement_count = self.breadth_candidate_count - len(retained)
        repair_prompt = build_breadth_repair_prompt(
            original_prompt=initial_prompt,
            original_response=response.text,
            validation_error=errors[-1],
            retained_candidates=retained,
            rejections=rejections,
            replacement_count=replacement_count,
            configured_candidate_count=self.breadth_candidate_count,
        )
        repair_response = self._complete(
            system_prompt=BREADTH_SYSTEM_PROMPT,
            user_prompt=repair_prompt,
            purpose="research_breadth_repair",
            attempt=1,
            context=context,
            calls=calls,
        )
        try:
            replacements = BreadthPlan.from_json_with_bounds(
                repair_response.text,
                min_candidates=1,
                max_candidates=replacement_count,
                exact_candidates=replacement_count,
            )
            reused_ids = sorted(
                item.candidate_id
                for item in replacements.candidates
                if item.candidate_id.casefold() in initial_candidate_ids
            )
            if reused_ids:
                raise BreadthValidationError(
                    "breadth repair candidate IDs must be new; reused "
                    + ", ".join(reused_ids)
                )
            combined_json = json.dumps({
                "schema_version": 1,
                "candidates": [
                    candidate.to_prompt_dict()
                    for candidate in (*retained, *replacements.candidates)
                ],
            })
            combined = BreadthPlan.from_json_with_bounds(
                combined_json,
                min_candidates=MIN_BREADTH_CANDIDATES,
                max_candidates=self.breadth_candidate_count,
                exact_candidates=self.breadth_candidate_count,
            )
            final_survivors, final_rejections = filter_breadth_candidates(
                combined,
                history=history,
                citation_source=self.citation_source,
                protected_candidate_ids=frozenset(
                    item.candidate_id for item in retained
                ),
            )
            if len(final_survivors) < MIN_BREADTH_SURVIVORS:
                detail = "; ".join(
                    f"{item.candidate_id}: {item.reason}"
                    for item in final_rejections
                ) or "too few genuinely different candidates survived"
                raise BreadthValidationError(
                    "breadth requires at least "
                    f"{MIN_BREADTH_SURVIVORS} post-filter survivors after repair; got "
                    f"{len(final_survivors)}: {detail}"
                )
            ranked = rank_breadth_candidates(
                final_survivors,
                history=history,
                coverage=coverage,
                context=context,
                citation_source=self.citation_source,
            )
            return ranked[0].candidate
        except BreadthValidationError as exc:
            errors.append(str(exc))
            raise ResearchOutputError(
                "Research Agent breadth phase failed after 2 call(s): "
                + " | ".join(
                    f"attempt {index + 1}: {error}"
                    for index, error in enumerate(errors)
                )
            ) from exc

    def _generate_depth(
        self,
        *,
        context: ResearchContext,
        history: Sequence[RunRecord],
        citations,
        selected: BreadthCandidate,
        calls: list[LLMResponse],
    ) -> ResearchProposal:
        initial_prompt = build_proposal_prompt(context, citations, selected)
        user_prompt = initial_prompt
        errors: list[str] = []
        for attempt in range(self.max_repair_attempts + 1):
            purpose = "research_depth" if attempt == 0 else "research_depth_repair"
            response = self._complete(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                purpose=purpose,
                attempt=attempt,
                context=context,
                calls=calls,
            )
            try:
                return self._validate_response(
                    response.text,
                    context,
                    history,
                    selected=selected,
                )
            except (ProposalValidationError, CitationValidationError) as exc:
                errors.append(str(exc))
                if attempt >= self.max_repair_attempts:
                    raise ResearchOutputError(
                        "Research Agent depth phase failed after "
                        f"{attempt + 1} call(s): " + " | ".join(
                            f"attempt {index + 1}: {error}"
                            for index, error in enumerate(errors)
                        )
                    ) from exc
                user_prompt = build_repair_prompt(
                    original_prompt=initial_prompt,
                    original_response=response.text,
                    validation_error=str(exc),
                    selected_candidate=selected,
                )
        raise AssertionError("unreachable")

    def _validate_response(
        self,
        text: str,
        context: ResearchContext,
        history: Sequence[RunRecord],
        *,
        selected: Optional[BreadthCandidate] = None,
    ) -> ResearchProposal:
        proposal = ResearchProposal.from_json(text)
        validate_research_direction_safety(
            proposal.to_dict(),
            origin="depth Research proposal",
        )
        _validate_proposal_against_context(proposal, context, history)
        validate_proposal_citations(proposal, self.citation_source)
        if selected is not None:
            validate_depth_alignment(proposal, selected)
        return proposal

    @staticmethod
    def _citation_query(context: ResearchContext) -> str:
        recent = " ".join(item.hypothesis for item in context.iterations[-5:])
        return (
            f"{context.task} ranking recommendation behavior sequence multi-task "
            f"watch time feature interactions {recent}"
        )

    def _set_last_usage(self, calls: Sequence[LLMResponse]) -> None:
        self.last_usage = {
            "llm_calls": len(calls),
            "real_model_calls": sum(response.is_real_model_call for response in calls),
            "tokens_in": sum(response.tokens_in for response in calls),
            "tokens_out": sum(response.tokens_out for response in calls),
            "cost_usd": round(sum(response.cost_usd for response in calls), 6),
        }
