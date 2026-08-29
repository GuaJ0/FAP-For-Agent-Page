"""Deterministic, zero-API Research Agent.

The offline agent selects from a ranked backlog of complete, citation-backed
``ResearchProposal`` templates.  It is history-aware rather than round-robin:
each call rebuilds validation-only context, filters ideas that do not fit the
remaining budget, and skips reverted/abandoned duplicates before returning the
highest-ranked viable proposal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from agent.agents import Idea
from agent.config import ConvergenceConfig, DEFAULT_CONFIG
from agent.records import RunRecord
from agent.research.agent import (
    DuplicateHypothesisError,
    ResearchAgentError,
    _assert_validation_only_context,
    _reject_duplicate,
)
from agent.research.citations import (
    CitationSource,
    CitationValidationError,
    JsonCitationCatalog,
    validate_proposal_citations,
)
from agent.research.context import ResearchContext, build_research_context
from agent.research.schemas import ProposalValidationError, ResearchProposal


class OfflineResearchError(ResearchAgentError):
    """The deterministic Research configuration itself is invalid."""


class OfflineBacklogExhausted(OfflineResearchError):
    """No untried, citation-valid idea fits the remaining experiment budget."""


@dataclass(frozen=True)
class BacklogEntry:
    rank: int
    key: str
    expected_wall_s: float
    minimum_remaining_iterations: int
    title: str
    hypothesis: str
    mechanism: str
    citation_id: str
    claim_id: str
    evidence_application: str
    target_components: tuple[str, ...]
    steps: tuple[str, ...]
    hyperparameters: dict[str, Any]
    must_hold_constant: tuple[str, ...]
    dependencies: tuple[str, ...]
    hardware: str
    runtime_impact: str
    complexity: str
    feasibility_notes: str
    secondary_effects: dict[str, str]
    ablation: str
    failure_interpretation: str
    risks: tuple[str, ...]

    def fits(self, context: ResearchContext) -> bool:
        return (
            context.remaining_iterations >= self.minimum_remaining_iterations
            and context.remaining_wall_s >= self.expected_wall_s
        )

    def build(self, context: ResearchContext) -> ResearchProposal:
        parent = context.parent_iteration
        raw = {
            "schema_version": 1,
            "hypothesis_id": f"OFFLINE-{self.key}",
            "parent_iteration": parent,
            "title": self.title,
            "hypothesis": self.hypothesis,
            "rationale": {
                "mechanism": self.mechanism,
                "metric_alignment": ["GAUC", "nDCG@5"],
                "prior_results_used": [parent] if parent is not None else [],
                "evidence": [{
                    "citation_id": self.citation_id,
                    "claim_id": self.claim_id,
                    "application": self.evidence_application,
                }],
            },
            "implementation": {
                "target_components": list(self.target_components),
                "steps": list(self.steps),
                "hyperparameters": self.hyperparameters,
                "must_hold_constant": list(self.must_hold_constant),
                "feasibility": {
                    "dependencies": list(self.dependencies),
                    "hardware": self.hardware,
                    "estimated_runtime_impact": self.runtime_impact,
                    "implementation_complexity": self.complexity,
                    "notes": self.feasibility_notes,
                },
            },
            "evaluation": {
                "reference_iteration": parent,
                "primary_metric": "primary",
                "minimum_primary_delta": context.minimum_meaningful_delta,
                "expected_secondary_effects": self.secondary_effects,
                "ablation": self.ablation,
                "failure_interpretation": self.failure_interpretation,
            },
            "risks": list(self.risks),
        }
        # Even hand-authored offline entries go through the same strict schema
        # as model output. A malformed backlog entry is a code defect, not a
        # special offline format that downstream code has to tolerate.
        return ResearchProposal.from_dict(raw)


DEFAULT_BACKLOG: tuple[BacklogEntry, ...] = (
    BacklogEntry(
        rank=1,
        key="HYBRID-BPR",
        expected_wall_s=900.0,
        minimum_remaining_iterations=1,
        title="Hybrid pointwise and pairwise ranking objective",
        hypothesis=(
            "Add a small within-user BPR term to the accepted incumbent's pointwise objective "
            "so ranking pressure is introduced without discarding its useful pointwise signal."
        ),
        mechanism=(
            "Pure pointwise training is not aligned directly with within-user ordering, while pure "
            "BPR can discard useful absolute supervision. A weighted hybrid isolates whether a modest "
            "pairwise gradient improves ordering without replacing the incumbent objective."
        ),
        citation_id="rendle2009bpr",
        claim_id="pairwise-ranking-objective",
        evidence_application=(
            "BPR supplies the pairwise ranking component; retaining the incumbent loss makes this a "
            "material variation rather than repeating a pure-BPR experiment."
        ),
        target_components=("training objective", "within-user pair sampler"),
        steps=(
            "Keep the accepted incumbent scorer, features, and pointwise objective unchanged.",
            "Sample positive-negative impressions from the same user within each training epoch.",
            "Add lambda_bpr times the BPR loss to the incumbent loss and early-stop on validation primary.",
        ),
        hyperparameters={"lambda_bpr": [0.05, 0.1, 0.2], "pairs_per_positive": 1},
        must_hold_constant=("feature set", "model capacity", "optimizer", "checkpoint selection"),
        dependencies=(),
        hardware="Existing CPU environment",
        runtime_impact="Approximately 1.2x-1.5x incumbent training time with vectorized pair sampling",
        complexity="medium",
        feasibility_notes="Requires no new library; abandon expensive sweeps if the first nonzero weight loses clearly.",
        secondary_effects={
            "GAUC": "Expected to increase if within-user pair ordering improves.",
            "nDCG@5": "Expected to remain stable or increase if top-ranked positives improve.",
        },
        ablation="Compare lambda_bpr=0 with each nonzero value using the same seeds and all other settings fixed.",
        failure_interpretation=(
            "If every nonzero weight reduces validation primary, ranking-loss mismatch is not the main "
            "limitation for the incumbent and this objective family should be deprioritized."
        ),
        risks=(
            "Pair sampling can change user weighting relative to GAUC.",
            "The pairwise term may improve GAUC while reducing nDCG@5.",
        ),
    ),
    BacklogEntry(
        rank=2,
        key="GAUC-WEIGHTED-BPR",
        expected_wall_s=600.0,
        minimum_remaining_iterations=1,
        title="Positive-count-weighted within-user BPR sampling",
        hypothesis=(
            "Weight within-user BPR sampling by each user's positive-impression count so the training "
            "distribution more closely matches GAUC's positive-count weighting."
        ),
        mechanism=(
            "Uniform user sampling and GAUC optimize different user mixtures. Reweighting user selection "
            "while leaving the scorer and pairwise loss fixed tests that mismatch directly."
        ),
        citation_id="rendle2009bpr",
        claim_id="pairwise-ranking-objective",
        evidence_application=(
            "BPR defines the within-user pairwise objective; this experiment changes only how users are "
            "sampled to better match the validation metric's aggregation."
        ),
        target_components=("within-user pair sampler",),
        steps=(
            "Keep the existing pairwise scorer, loss, optimizer, and pair construction unchanged.",
            "Sample users in proportion to their positive-impression count instead of uniformly.",
            "Compare weighted and uniform sampling with identical pair counts and seeds.",
        ),
        hyperparameters={"user_sampling": ["uniform", "positive_count_weighted"]},
        must_hold_constant=("pairwise loss", "pair count", "features", "model capacity", "optimizer"),
        dependencies=(),
        hardware="Existing CPU environment",
        runtime_impact="Near-neutral versus an existing vectorized BPR run",
        complexity="low",
        feasibility_notes="A contained sampler change suitable when little wall-clock budget remains.",
        secondary_effects={
            "GAUC": "Expected to benefit most because its aggregation weights positive counts.",
            "nDCG@5": "May be neutral or slightly lower if heavy users receive more training weight.",
        },
        ablation="Run uniform and positive-count-weighted sampling with the same pair count and seeds.",
        failure_interpretation=(
            "If weighting does not improve GAUC, the earlier ranking-loss gap is unlikely to be explained "
            "by user-sampling mismatch alone."
        ),
        risks=("Optimizing GAUC weighting may trade off equally weighted per-user nDCG@5.",),
    ),
    BacklogEntry(
        rank=3,
        key="DIN-SHORT-HISTORY",
        expected_wall_s=1800.0,
        minimum_remaining_iterations=2,
        title="Candidate-conditioned short behavior history",
        hypothesis=(
            "Add a candidate-conditioned summary of each user's recent training interactions so the model "
            "can represent transient interests that static user and item IDs cannot express."
        ),
        mechanism=(
            "A user's relevant interest varies with the candidate. A short chronological history with "
            "candidate-conditioned weighting can expose intent that is lost in a single static user embedding."
        ),
        citation_id="zhou2018din",
        claim_id="candidate-conditioned-interest",
        evidence_application=(
            "DIN's candidate-adaptive interest representation motivates conditioning recent behavior on "
            "the video being ranked rather than pooling history identically for every candidate."
        ),
        target_components=("training-data history construction", "candidate-conditioned interaction features"),
        steps=(
            "Build each user's chronological history using training rows only, excluding the current impression.",
            "Represent the most recent items and weight their similarity to the candidate item.",
            "Concatenate the resulting interest summary with incumbent features and select on validation primary.",
        ),
        hyperparameters={"history_length": [20, 50], "interest_dimension": [16]},
        must_hold_constant=("label", "validation split", "base ID features", "selection metric"),
        dependencies=("PyTorch or another open-source autograd library if justified by the Coding owner",),
        hardware="CPU-capable; GPU optional for faster sequence batching",
        runtime_impact="Approximately 2x-3x incumbent time, including history preprocessing",
        complexity="high",
        feasibility_notes="Attempt only with enough budget for construction plus one controlled ablation.",
        secondary_effects={
            "GAUC": "Expected to increase for users whose interests shift over time.",
            "nDCG@5": "Expected to increase if the recent-interest signal sharpens the top of each list.",
        },
        ablation="Compare candidate-conditioned history against an unconditioned mean history using the same length.",
        failure_interpretation=(
            "If neither conditioned nor pooled recent history helps, short-term intent may be weak under the "
            "official split or the chosen history representation may be too lossy."
        ),
        risks=(
            "Incorrect temporal joins could leak future interactions.",
            "Sequence preprocessing and dependencies increase implementation and runtime cost.",
        ),
    ),
    BacklogEntry(
        rank=4,
        key="MULTITASK-ENGAGEMENT",
        expected_wall_s=2100.0,
        minimum_remaining_iterations=2,
        title="Multi-task auxiliary engagement objectives",
        hypothesis=(
            "Jointly predict long_view with related training-only engagement labels through a shared "
            "representation so sparse auxiliary supervision regularizes the primary ranking task."
        ),
        mechanism=(
            "Related response labels can transfer information through a shared representation while the "
            "primary head remains selected exclusively by validation primary."
        ),
        citation_id="ma2018esmm",
        claim_id="shared-representation-transfer",
        evidence_application=(
            "ESMM demonstrates feature-representation transfer across related user-response objectives; "
            "the proposed use is auxiliary regularization, not a claim that its exact conversion setup applies."
        ),
        target_components=("data loader for auxiliary training labels", "shared representation", "task heads"),
        steps=(
            "Load selected auxiliary labels for training rows without changing official split boundaries.",
            "Share the incumbent representation and add one lightweight prediction head per auxiliary task.",
            "Tune one auxiliary-loss weight while selecting checkpoints only on long_view validation primary.",
        ),
        hyperparameters={"auxiliary_tasks": [["is_click", "is_like"]], "lambda_aux": [0.05, 0.1]},
        must_hold_constant=("primary label", "official split", "base features", "primary checkpoint metric"),
        dependencies=("PyTorch or another open-source autograd library if justified by the Coding owner",),
        hardware="CPU-capable; GPU optional",
        runtime_impact="Approximately 1.5x-2.5x incumbent time depending on the shared model",
        complexity="high",
        feasibility_notes="Keep the task set and weight sweep small enough for one attributable iteration.",
        secondary_effects={
            "GAUC": "May increase if auxiliary engagement distinguishes user preference more reliably.",
            "nDCG@5": "May increase if shared features improve top-ranked relevance.",
        },
        ablation="Compare the primary-only model with the same architecture and lambda_aux=0.",
        failure_interpretation=(
            "If auxiliary heads do not help, their labels are either insufficiently aligned with long_view "
            "or the shared representation introduces harmful task interference."
        ),
        risks=("Auxiliary tasks can cause negative transfer.", "Additional labels must remain training-only inputs."),
    ),
    BacklogEntry(
        rank=5,
        key="WATCHTIME-AUXILIARY",
        expected_wall_s=1800.0,
        minimum_remaining_iterations=2,
        title="Watch-time auxiliary objective",
        hypothesis=(
            "Add a bounded watch-time auxiliary prediction target to the accepted incumbent so the shared "
            "representation captures engagement intensity beyond the binary long_view label."
        ),
        mechanism=(
            "Watch time contains graded engagement information that a binary target discards. Used only as "
            "training-side auxiliary supervision, it may refine ordering while validation primary remains authoritative."
        ),
        citation_id="covington2016youtube",
        claim_id="ranking-objective-and-watch-time",
        evidence_application=(
            "The YouTube ranking work motivates expected watch time as a useful engagement signal distinct "
            "from click probability; here it is tested as an auxiliary signal for long_view ranking."
        ),
        target_components=("training data targets", "auxiliary regression head", "joint objective"),
        steps=(
            "Derive a normalized, capped watch-time target from training rows only.",
            "Add a lightweight auxiliary head sharing the incumbent representation.",
            "Tune a single auxiliary-loss weight and retain validation-primary checkpoint selection.",
        ),
        hyperparameters={"watch_time_transform": ["log1p_capped"], "lambda_watch": [0.05, 0.1]},
        must_hold_constant=("primary label", "official split", "base features", "primary selection metric"),
        dependencies=("Open-source autograd library if the incumbent implementation requires it",),
        hardware="CPU-capable; GPU optional",
        runtime_impact="Approximately 1.3x-2x incumbent time",
        complexity="medium",
        feasibility_notes="Use one robust target transform rather than a broad modelling sweep.",
        secondary_effects={
            "GAUC": "May improve if engagement intensity separates positive candidates within users.",
            "nDCG@5": "May improve if highly engaging positives move toward the top five.",
        },
        ablation="Compare lambda_watch=0 with each nonzero weight using identical architecture and seeds.",
        failure_interpretation=(
            "If the auxiliary target hurts validation primary, watch-time magnitude is either misaligned with "
            "long_view ordering or its censoring/noise needs a different treatment."
        ),
        risks=("Watch time is censored by video completion.", "Long videos may dominate without normalization."),
    ),
    BacklogEntry(
        rank=6,
        key="DEEPFM",
        expected_wall_s=2400.0,
        minimum_remaining_iterations=2,
        title="Shared-embedding DeepFM interaction tower",
        hypothesis=(
            "Add a small nonlinear interaction tower sharing the incumbent FM embeddings so higher-order "
            "feature interactions are tested without removing the accepted low-order FM path."
        ),
        mechanism=(
            "The FM path preserves known low-order interactions while a compact shared-embedding tower can "
            "represent higher-order combinations that the incumbent cannot express."
        ),
        citation_id="guo2017deepfm",
        claim_id="joint-low-high-order-interactions",
        evidence_application=(
            "DeepFM motivates jointly learning low- and higher-order interactions with shared embeddings; "
            "the proposed tower is deliberately small because capacity has not been the leading bottleneck."
        ),
        target_components=("model scorer", "shared feature embeddings", "nonlinear interaction tower"),
        steps=(
            "Retain the incumbent FM output as one additive scoring path.",
            "Feed the same field embeddings into a compact two-layer interaction tower.",
            "Add the tower logit to the FM logit and compare against a parameter-matched FM control.",
        ),
        hyperparameters={"hidden_layers": [[64, 32]], "dropout": [0.1], "embedding_dim": [16]},
        must_hold_constant=("feature set", "embedding dimension", "objective", "checkpoint selection"),
        dependencies=("PyTorch or another open-source autograd library if justified by the Coding owner",),
        hardware="CPU-capable but GPU preferred",
        runtime_impact="Approximately 2x-4x incumbent time",
        complexity="high",
        feasibility_notes="Lowest backlog priority because prior evidence suggests capacity is not the main bottleneck.",
        secondary_effects={
            "GAUC": "May increase if cross-field combinations improve within-user ordering.",
            "nDCG@5": "May increase if nonlinear interactions sharpen the highest scores.",
        },
        ablation="Compare the combined model against the unchanged FM and a parameter-matched wider FM.",
        failure_interpretation=(
            "If the compact tower fails to beat FM, additional architecture capacity should remain deprioritized."
        ),
        risks=("Extra capacity may overfit the available training rows.", "Dependency and runtime cost may exceed payoff."),
    ),
)


@dataclass
class OfflineResearchAgent:
    """History-aware deterministic implementation of ``ResearchAgent``."""

    citation_source: CitationSource = field(default_factory=JsonCitationCatalog)
    backlog: Sequence[BacklogEntry] = DEFAULT_BACKLOG
    convergence: ConvergenceConfig = DEFAULT_CONFIG.convergence

    def propose(self, history: list[RunRecord]) -> Idea:
        proposal = self.select_proposal(history)
        return Idea(
            hypothesis=proposal.to_handoff_text(),
            parent_iteration=proposal.parent_iteration,
        )

    def select_proposal(self, history: Sequence[RunRecord]) -> ResearchProposal:
        context = build_research_context(history, self.convergence)
        _assert_validation_only_context(context)

        if context.remaining_iterations <= 0 or context.remaining_wall_s <= 0:
            raise OfflineBacklogExhausted(
                "offline Research backlog cannot propose: the experiment budget is exhausted"
            )

        viable = [entry for entry in self.backlog if entry.fits(context)]
        # Under a tight budget, prefer the cheapest viable experiment. With
        # comfortable budget, preserve the research ranking. Both orderings
        # are stable and depend on context, never on an internal rotation index.
        tight_budget = context.remaining_iterations <= 3 or context.remaining_wall_s <= 3600.0
        viable.sort(
            key=(
                (lambda entry: (entry.expected_wall_s, entry.rank, entry.key))
                if tight_budget else
                (lambda entry: (entry.rank, entry.expected_wall_s, entry.key))
            )
        )

        rejected_duplicates = 0
        for entry in viable:
            try:
                proposal = entry.build(context)
                validate_proposal_citations(proposal, self.citation_source)
                _reject_duplicate(proposal, history)
            except DuplicateHypothesisError:
                rejected_duplicates += 1
                continue
            except (ProposalValidationError, CitationValidationError) as exc:
                raise OfflineResearchError(
                    f"offline backlog entry {entry.key!r} is invalid: {exc}"
                ) from exc
            return proposal

        raise OfflineBacklogExhausted(
            "offline Research backlog exhausted: no untried proposal fits the remaining "
            f"budget ({len(viable)} feasible, {rejected_duplicates} already rejected/abandoned)"
        )
