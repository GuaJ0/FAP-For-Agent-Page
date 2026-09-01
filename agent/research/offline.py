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

from agent.agents import Idea, ResearchExhausted
from agent.config import ConvergenceConfig, DEFAULT_CONFIG
from agent.records import RunRecord
from agent.research.agent import (
    DuplicateHypothesisError,
    ResearchAgentError,
    _assert_validation_only_context,
    _validate_proposal_against_context,
)
from agent.research.citations import (
    CitationSource,
    CitationValidationError,
    JsonCitationCatalog,
    validate_proposal_citations,
)
from agent.research.context import ResearchContext, build_research_context
from agent.research.findings import FindingsLedger
from agent.research.schemas import ProposalValidationError, ResearchProposal


class OfflineResearchError(ResearchAgentError):
    """The deterministic Research configuration itself is invalid."""


class OfflineBacklogExhausted(OfflineResearchError, ResearchExhausted):
    """No untried, citation-valid idea fits the remaining experiment budget.

    Also a ResearchExhausted, which is what tells the Orchestrator this is an
    orderly finish rather than a failure to propose: retrying a backlog that
    has run out cannot succeed, so it must not accumulate failed records or
    escalate to a human. Still an OfflineResearchError so existing callers and
    tests that catch that keep catching it.
    """


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
        # Treatment FIRST, control second. The first non-null value is what
        # reaches config.json and therefore what actually runs; listing
        # "uniform" first made this entry measure the control condition and
        # report the incumbent's behaviour as the idea's result.
        hyperparameters={"user_sampling": ["positive_count_weighted", "uniform"]},
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
        dependencies=(
            "PyTorch, for the gradients this needs. External open-source libraries are permitted "
            "(docs/coding-agent.md) and torch is installed and pinned in requirements.txt, so it may "
            "be imported directly. Pin it to one thread with torch.set_num_threads(1) to stay inside "
            "the single-core budget.",
        ),
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
        dependencies=(
            "PyTorch, for the gradients this needs. External open-source libraries are permitted "
            "(docs/coding-agent.md) and torch is installed and pinned in requirements.txt, so it may "
            "be imported directly. Pin it to one thread with torch.set_num_threads(1) to stay inside "
            "the single-core budget.",
        ),
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
        dependencies=(
            "PyTorch, for the gradients this needs. External open-source libraries are permitted "
            "(docs/coding-agent.md) and torch is installed and pinned in requirements.txt, so it may "
            "be imported directly. Pin it to one thread with torch.set_num_threads(1) to stay inside "
            "the single-core budget.",
        ),
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
        dependencies=(
            "PyTorch, for the gradients this needs. External open-source libraries are permitted "
            "(docs/coding-agent.md) and torch is installed and pinned in requirements.txt, so it may "
            "be imported directly. Pin it to one thread with torch.set_num_threads(1) to stay inside "
            "the single-core budget.",
        ),
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
    # ------------------------------------------------------------------
    # VARIANTS OF THE STRUCTURALLY COMPLEX DIRECTIONS (ranks 7-13).
    #
    # A single Coding Agent generation at one setting cannot settle a complex
    # mechanism: if it loses, the ledger cannot tell "the mechanism is wrong"
    # from "the one implementation we happened to get was weak". These entries
    # give each complex direction 2-3 deliberately different attempts, the way
    # HYBRID-BPR and GAUC-WEIGHTED-BPR already do for the ranking-loss
    # direction, so findings.py can record a confidence tier that reflects how
    # much of the direction was actually measured.
    #
    # Ranks start at 7 so entries 1-6 keep the exact ordering the graded run
    # already depends on -- these are additions, not a reprioritisation.
    #
    # They share a findings family with their sibling (see DIRECTION_FAMILIES
    # in agent/research/findings.py), so their results roll up into one entry
    # per direction rather than several unrelated one-shot verdicts.
    # ------------------------------------------------------------------
    BacklogEntry(
        rank=7,
        key="DIN-LONG-HISTORY",
        expected_wall_s=2400.0,
        minimum_remaining_iterations=2,
        title="Candidate-conditioned long behavior history",
        hypothesis=(
            "Extend the candidate-conditioned behavior summary to a long history window so that "
            "stable long-run preference, not only the most recent handful of impressions, is "
            "available to the scorer."
        ),
        mechanism=(
            "A short window can only express transient intent. If recent-history conditioning fails at "
            "20-50 impressions the cause may be window length rather than the mechanism, since KuaiRand "
            "users have hundreds to thousands of training interactions that a short window discards."
        ),
        citation_id="zhou2018din",
        claim_id="candidate-conditioned-interest",
        evidence_application=(
            "DIN's candidate-adaptive interest representation is applied over a substantially longer "
            "window, testing history length as the variable while the conditioning mechanism is fixed."
        ),
        target_components=("training-data history construction", "candidate-conditioned interaction features"),
        steps=(
            "Reuse the candidate-conditioned history construction, changing only the window length and interest width.",
            "Build each user's chronological history from training rows only, excluding the current impression.",
            "Select on validation primary and report cost against the short-window variant.",
        ),
        hyperparameters={"history_length": [100, 200], "interest_dimension": [32]},
        must_hold_constant=("label", "validation split", "base ID features", "selection metric", "conditioning mechanism"),
        dependencies=(
            "PyTorch, for the gradients this needs. External open-source libraries are permitted "
            "(docs/coding-agent.md) and torch is installed and pinned in requirements.txt, so it may "
            "be imported directly. Pin it to one thread with torch.set_num_threads(1) to stay inside "
            "the single-core budget.",
        ),
        hardware="CPU-capable; GPU optional for faster sequence batching",
        runtime_impact="Approximately 3x-4x incumbent time; the longest sequence variant",
        complexity="high",
        feasibility_notes=(
            "Deliberately the most expensive sequence variant. Run it only when the short-window "
            "attempt has already produced a real measurement to compare against."
        ),
        secondary_effects={
            "GAUC": "Expected to increase over the short window if stable preference dominates transient intent.",
            "nDCG@5": "Expected to increase if longer context sharpens the top of each list.",
        },
        ablation="Compare against the short-window variant with the conditioning mechanism held identical.",
        failure_interpretation=(
            "If the long window also loses, window length is not what limited the short-window attempt, "
            "which materially strengthens a negative verdict on candidate-conditioned history."
        ),
        risks=(
            "Longer histories increase preprocessing cost and the chance of an incorrect temporal join.",
            "Very long windows may dilute the recent signal they are meant to extend.",
        ),
    ),
    BacklogEntry(
        rank=8,
        key="DIN-MEAN-POOL",
        expected_wall_s=1500.0,
        minimum_remaining_iterations=1,
        title="Unconditioned mean-pooled behavior history",
        hypothesis=(
            "Summarize each user's recent training interactions by unconditioned mean pooling, so that "
            "the value of having history at all is separated from the value of conditioning it on the candidate."
        ),
        mechanism=(
            "Candidate conditioning and history availability are two different claims bundled into one "
            "experiment. Mean pooling supplies the same history with the attention mechanism removed, "
            "which is the control that makes either result attributable."
        ),
        citation_id="zhou2018din",
        claim_id="candidate-conditioned-interest",
        evidence_application=(
            "DIN's contribution is specifically the candidate-adaptive weighting over history; this entry "
            "is its control, isolating that contribution from the presence of history features."
        ),
        target_components=("training-data history construction", "pooled history features"),
        steps=(
            "Build the same chronological training-only history as the conditioned variants.",
            "Mean-pool the recent item embeddings with no candidate-dependent weighting.",
            "Concatenate the pooled summary with incumbent features and select on validation primary.",
        ),
        hyperparameters={"history_length": [50], "pooling": ["mean"], "interest_dimension": [16]},
        must_hold_constant=("label", "validation split", "base ID features", "selection metric", "history window"),
        dependencies=(),
        hardware="Existing CPU environment",
        runtime_impact="Approximately 1.5x-2x incumbent time; the cheapest sequence variant",
        complexity="medium",
        feasibility_notes=(
            "Cheapest of the three sequence attempts and the one worth keeping if budget tightens, "
            "because it is the control the other two are interpreted against."
        ),
        secondary_effects={
            "GAUC": "Expected to increase modestly if history matters at all, regardless of conditioning.",
            "nDCG@5": "Expected to move less than the conditioned variants if conditioning is what pays.",
        },
        ablation="Compare directly against the candidate-conditioned variant at the same history length.",
        failure_interpretation=(
            "If mean pooling helps and conditioning does not, the payoff is history availability, not "
            "attention. If neither helps, behavior history is weak under the official split."
        ),
        risks=(
            "Mean pooling may wash out the recent signal that makes history useful.",
            "A weak control invites over-reading a small difference between the two variants.",
        ),
    ),
    BacklogEntry(
        rank=9,
        key="MULTITASK-ALL-ENGAGEMENT",
        expected_wall_s=2400.0,
        minimum_remaining_iterations=2,
        title="Broad multi-task engagement supervision at low weight",
        hypothesis=(
            "Predict the full set of observed engagement labels as auxiliary tasks at a low shared weight, "
            "so that sparse but diverse supervision regularizes the primary long_view representation."
        ),
        mechanism=(
            "A two-task setup may simply have too little auxiliary signal to regularize anything. Widening "
            "the task set while lowering the per-task weight tests breadth of supervision rather than its strength."
        ),
        citation_id="ma2018esmm",
        claim_id="shared-representation-transfer",
        evidence_application=(
            "ESMM's feature-representation transfer across related user-response objectives is applied at "
            "greater task breadth; the auxiliary heads remain regularizers, not conversion-rate estimators."
        ),
        target_components=("data loader for auxiliary training labels", "shared representation", "task heads"),
        steps=(
            "Load all five observed engagement labels for training rows without changing official split boundaries.",
            "Share the incumbent representation and add one lightweight head per auxiliary task.",
            "Use a single low weight shared across auxiliary heads and select checkpoints only on long_view validation primary.",
        ),
        hyperparameters={
            "auxiliary_tasks": [["is_click", "is_like", "is_follow", "is_comment", "is_forward"]],
            "lambda_aux": [0.02],
        },
        must_hold_constant=("primary label", "official split", "base features", "primary checkpoint metric"),
        dependencies=(
            "PyTorch, for the gradients this needs. External open-source libraries are permitted "
            "(docs/coding-agent.md) and torch is installed and pinned in requirements.txt, so it may "
            "be imported directly. Pin it to one thread with torch.set_num_threads(1) to stay inside "
            "the single-core budget.",
        ),
        hardware="CPU-capable; GPU optional",
        runtime_impact="Approximately 2x-3x incumbent time; five heads over a shared trunk",
        complexity="high",
        feasibility_notes=(
            "The rarest labels (is_follow, is_comment, is_forward) are extremely sparse; the low shared "
            "weight is what keeps their gradient noise from dominating."
        ),
        secondary_effects={
            "GAUC": "May increase if diverse engagement sharpens the shared user-item representation.",
            "nDCG@5": "May increase if the regularization reduces overfitting on the head of each list.",
        },
        ablation="Compare against the two-task variant and against lambda_aux=0 at identical architecture.",
        failure_interpretation=(
            "If broad low-weight supervision also loses, the limitation is the transfer itself rather than "
            "how many auxiliary labels were supplied."
        ),
        risks=(
            "Very sparse labels can contribute mostly gradient noise.",
            "Five heads increase the chance of negative transfer onto the primary task.",
        ),
    ),
    BacklogEntry(
        rank=10,
        key="MULTITASK-CLICK-HEAVY",
        expected_wall_s=1800.0,
        minimum_remaining_iterations=1,
        title="Single dense auxiliary task at high weight",
        hypothesis=(
            "Predict is_click alone as an auxiliary task at a substantially higher weight, testing whether "
            "one dense, closely-related label transfers more than several sparse ones."
        ),
        mechanism=(
            "is_click is by far the densest engagement label and the most semantically adjacent to long_view. "
            "If auxiliary supervision helps at all, the strongest effect should appear here, and a low weight "
            "may simply have been too small to produce a measurable one."
        ),
        citation_id="ma2018esmm",
        claim_id="shared-representation-transfer",
        evidence_application=(
            "ESMM's transfer argument is strongest between closely related, densely observed responses; this "
            "entry tests that regime directly rather than diluting it across sparse labels."
        ),
        target_components=("data loader for auxiliary training labels", "shared representation", "task heads"),
        steps=(
            "Load only is_click for training rows without changing official split boundaries.",
            "Share the incumbent representation and add one auxiliary head.",
            "Sweep the auxiliary weight upward while selecting checkpoints only on long_view validation primary.",
        ),
        hyperparameters={"auxiliary_tasks": [["is_click"]], "lambda_aux": [0.3, 0.5]},
        must_hold_constant=("primary label", "official split", "base features", "primary checkpoint metric"),
        dependencies=(
            "PyTorch, for the gradients this needs. External open-source libraries are permitted "
            "(docs/coding-agent.md) and torch is installed and pinned in requirements.txt, so it may "
            "be imported directly. Pin it to one thread with torch.set_num_threads(1) to stay inside "
            "the single-core budget.",
        ),
        hardware="CPU-capable; GPU optional",
        runtime_impact="Approximately 1.5x-2x incumbent time; one head over a shared trunk",
        complexity="medium",
        feasibility_notes=(
            "The cheapest multi-task attempt and the one that spans the widest weight range, which is why "
            "it is the variant to keep if only one multi-task round is affordable."
        ),
        secondary_effects={
            "GAUC": "May increase if click behavior shares ordering structure with long_view.",
            "nDCG@5": "May decrease if a high weight pulls the representation toward click rather than long view.",
        },
        ablation="Compare lambda_aux in {0, 0.3, 0.5} at identical architecture and seeds.",
        failure_interpretation=(
            "If the densest, most related label at high weight does not transfer, auxiliary engagement "
            "supervision is unlikely to help long_view ranking at any task set or weight."
        ),
        risks=(
            "A high auxiliary weight can dominate the primary objective.",
            "Optimizing toward click may actively mis-order long_view positives.",
        ),
    ),
    BacklogEntry(
        rank=11,
        key="WATCHTIME-CENSORED",
        expected_wall_s=2100.0,
        minimum_remaining_iterations=2,
        title="Censored one-sided watch-time objective",
        hypothesis=(
            "Treat watch time as right-censored when a video is watched to completion and fit a one-sided "
            "auxiliary loss, so completed views are not penalized for a watch time they could not exceed."
        ),
        mechanism=(
            "Watch time is truncated by video duration: a fully-watched short video records a small watch time "
            "that squared error reads as weak engagement. A one-sided loss penalizes under-prediction only on "
            "censored examples, which is the modelling correction plain regression omits."
        ),
        citation_id="covington2016youtube",
        claim_id="ranking-objective-and-watch-time",
        evidence_application=(
            "The YouTube ranking work motivates expected watch time as an engagement signal; this entry adds "
            "the censoring correction that a plain regression target ignores."
        ),
        target_components=("training data targets", "censoring indicator", "auxiliary one-sided loss"),
        steps=(
            "Derive a watch-time target and a censoring indicator (play_time_ms at or above duration_ms) from training rows only.",
            "Apply squared error on uncensored rows and a one-sided penalty on censored rows for under-prediction only.",
            "Tune a single auxiliary weight and retain validation-primary checkpoint selection.",
        ),
        hyperparameters={"censoring_threshold_ratio": [0.95], "lambda_watch": [0.05, 0.1]},
        must_hold_constant=("primary label", "official split", "base features", "primary selection metric"),
        dependencies=(
            "PyTorch, for the gradients this needs. External open-source libraries are permitted "
            "(docs/coding-agent.md) and torch is installed and pinned in requirements.txt, so it may "
            "be imported directly. Pin it to one thread with torch.set_num_threads(1) to stay inside "
            "the single-core budget.",
        ),
        hardware="CPU-capable; GPU optional",
        runtime_impact="Approximately 1.5x-2x incumbent time",
        complexity="high",
        feasibility_notes=(
            "The censoring indicator must be derived from duration, not assumed; an incorrect threshold "
            "silently turns this back into plain regression."
        ),
        secondary_effects={
            "GAUC": "May increase if correcting censoring stops short completed videos from being read as weak engagement.",
            "nDCG@5": "May increase if completed short videos are promoted toward the top five.",
        },
        ablation="Compare the one-sided censored loss against plain squared error on the same target and weight.",
        failure_interpretation=(
            "If the censored loss does not beat plain regression, censoring is not what limited the watch-time "
            "signal, and the remaining explanation is the target itself rather than its loss."
        ),
        risks=(
            "The censoring threshold is a modelling choice that can be set wrong.",
            "A one-sided loss is easy to implement with the inequality reversed, which would silently invert the correction.",
        ),
    ),
    BacklogEntry(
        rank=12,
        key="WATCHTIME-RATIO",
        expected_wall_s=1500.0,
        minimum_remaining_iterations=1,
        title="Duration-normalized watch-ratio auxiliary objective",
        hypothesis=(
            "Use completion ratio -- play time divided by video duration -- as the auxiliary target, so "
            "engagement intensity is comparable across videos of very different lengths."
        ),
        mechanism=(
            "Raw watch time confounds engagement with duration: a long video accumulates more watch time at "
            "equal interest. Normalizing by duration removes that confound and sidesteps censoring entirely, "
            "since the ratio is naturally bounded at completion."
        ),
        citation_id="covington2016youtube",
        claim_id="ranking-objective-and-watch-time",
        evidence_application=(
            "The same watch-time engagement signal is used, with duration normalization substituted for raw "
            "magnitude; this isolates the target's scale rather than its presence."
        ),
        target_components=("training data targets", "auxiliary regression head", "joint objective"),
        steps=(
            "Derive a clipped completion ratio from play_time_ms and duration_ms on training rows only.",
            "Add a lightweight auxiliary head sharing the incumbent representation.",
            "Tune a single auxiliary weight and retain validation-primary checkpoint selection.",
        ),
        hyperparameters={"watch_time_transform": ["completion_ratio_clipped"], "lambda_watch": [0.05, 0.2]},
        must_hold_constant=("primary label", "official split", "base features", "primary selection metric"),
        dependencies=(
            "PyTorch, for the gradients this needs. External open-source libraries are permitted "
            "(docs/coding-agent.md) and torch is installed and pinned in requirements.txt, so it may "
            "be imported directly. Pin it to one thread with torch.set_num_threads(1) to stay inside "
            "the single-core budget.",
        ),
        hardware="CPU-capable; GPU optional",
        runtime_impact="Approximately 1.3x-1.8x incumbent time; the cheapest watch-time variant",
        complexity="medium",
        feasibility_notes=(
            "Structurally the simplest of the three watch-time attempts: a bounded target needing no "
            "censoring machinery, which makes it the one to keep under a tight budget."
        ),
        secondary_effects={
            "GAUC": "May increase if duration-normalized engagement separates candidates within a user.",
            "nDCG@5": "May increase if short fully-watched videos are ranked more highly.",
        },
        ablation="Compare completion ratio against the raw log1p-capped target at matched auxiliary weight.",
        failure_interpretation=(
            "If normalization does not beat the raw target, duration confounding is not what limited the "
            "watch-time signal."
        ),
        risks=(
            "Rows with missing or zero duration must be excluded rather than silently producing infinite ratios.",
            "Clipping at completion discards rewatch behavior.",
        ),
    ),
    # ------------------------------------------------------------------
    # THE TWO DIRECTIONS FROM solution/ideas.md THAT THE BACKLOG NEVER COVERED
    # (ideas.md "Unexplored directions" items 6 and 7). Ranked last only
    # because ranks 1-12 were already assigned; both are cheap relative to the
    # sequence and multi-task work above.
    #
    # DOMAIN JUDGMENT CALLS ARE FLAGGED IN docs/exploration-campaign.md -- the
    # time windows, the decay half-lives, the censoring threshold and the
    # diagnostic-only framing below are proposals for review, not settled
    # science.
    # ------------------------------------------------------------------
    BacklogEntry(
        rank=13,
        key="TIME-DRIFT",
        expected_wall_s=1200.0,
        minimum_remaining_iterations=1,
        title="Recency-weighted training with time-of-day interaction features",
        hypothesis=(
            "Down-weight older training impressions by recency and add a time-of-day by duration-bucket "
            "interaction, so the model reflects the period the validation split is drawn from rather than "
            "averaging uniformly over the whole training window."
        ),
        mechanism=(
            "Training treats every logged impression as equally representative, but the split is chronological "
            "and user interests and item popularity both move over the window. Recency weighting corrects the "
            "distribution the model is fit to, while a time-of-day cross supplies within-user temporal signal. "
            "The cross matters specifically because a term constant across one user's impressions cannot "
            "reorder them: crossing time against an item-side bucket is what makes the feature able to rank."
        ),
        citation_id="covington2016youtube",
        claim_id="example-age-freshness",
        evidence_application=(
            "The example-age treatment motivates representing when an impression occurred rather than "
            "averaging over the training window; here it is applied as sample weighting plus an explicit "
            "time-of-day interaction."
        ),
        target_components=("training sample weights", "time-derived interaction features"),
        steps=(
            "Derive each training row's age from its date relative to the latest training date.",
            "Weight the training loss by 0.5 ** (age_days / recency_half_life_days) -- a true half-life, "
            "so a row exactly one half-life old counts half as much; null keeps the no-weighting control.",
            "Add an hour-of-day bucket crossed with dur_bucket as an interaction feature, never as a user-side term alone.",
            "Select on validation primary with all other settings held at incumbent values.",
        ),
        hyperparameters={
            # Fitted to the MEASURED training window, not a guess: the official
            # train split (harness/data.py) declares 20220408-20220421 but the
            # data holds no 04-08 rows, so it spans 13 days (1,141,112 rows).
            # 3 days is roughly a quarter of that and retains ~19% of total
            # sample weight -- a genuinely aggressive end. 14 days is one window
            # length and retains ~67%. The previous [7, 21] pair was chosen
            # before the window was known and bracketed it poorly: at a 21-day
            # half-life the oldest row still keeps 67% of its weight, which is
            # so close to the uniform control that the cell measured almost
            # nothing.
            "recency_half_life_days": [3, 14, None],
            "hour_buckets": [8],
            "time_cross": ["hour_bucket_x_dur_bucket"],
        },
        must_hold_constant=("label", "official split", "model capacity", "optimizer", "selection metric"),
        dependencies=(),
        hardware="Existing CPU environment",
        runtime_impact="Near-neutral: one extra feature field and a per-row weight",
        complexity="low",
        feasibility_notes=(
            "Among the cheapest untried directions -- no new dependency and no architecture change. "
            "half_life=None is the control that separates the weighting from the added feature."
        ),
        secondary_effects={
            "GAUC": "May increase if recent behavior is more representative of the validation period.",
            "nDCG@5": "May increase if time-of-day interacts with which durations users complete.",
        },
        ablation=(
            "Three cells: weighting only, time cross only, and both, each against the unchanged incumbent "
            "on the same seeds."
        ),
        failure_interpretation=(
            "If neither recency weighting nor the time cross helps, the training window is temporally "
            "homogeneous enough that drift is not a limiting factor under the official split."
        ),
        risks=(
            "The training window is heavily front-loaded -- 63.6% of its rows fall in its first four days, "
            "which are also its OLDEST -- so recency weighting down-weights most of the data: a 3-day "
            "half-life retains only about 19% of total sample weight and can raise variance materially.",
            "A time feature used only as a user-side term contributes exactly zero to within-user ranking.",
            "Date handling must not let a validation-period row influence training weights.",
        ),
    ),
    BacklogEntry(
        rank=14,
        key="LOG-RANDOM-DIAGNOSTIC",
        expected_wall_s=900.0,
        minimum_remaining_iterations=1,
        title="Unbiased overfitting diagnostic on the randomized-exposure log",
        hypothesis=(
            "Report ranking quality on the randomized-exposure log alongside the ordinary validation metric, "
            "to establish whether incumbent gains reflect genuine ranking quality or fitting to the biased "
            "exposure policy that produced the training log."
        ),
        mechanism=(
            "Training and validation impressions are both drawn from the deployed recommender's exposure "
            "policy, so both share its bias; a model can improve on them by learning that policy rather than "
            "user preference. Impressions logged under randomized exposure do not share that bias, which makes "
            "them a check the ordinary split structurally cannot provide."
        ),
        citation_id="schnabel2016propensity",
        claim_id="biased-logging-biases-evaluation",
        evidence_application=(
            "The result that non-uniform exposure biases evaluation, and that randomized-exposure data gives "
            "an unbiased sample, is applied as a read-only diagnostic rather than as a change of objective."
        ),
        target_components=("evaluation reporting", "randomized-exposure diagnostic split"),
        steps=(
            "Load the randomized-exposure log strictly as an additional evaluation set, never as training data.",
            "Keep ONLY rows dated 20220422-20220428. The file spans 20220422-20220508, and 20220429 onward "
            "is the official held-out date range; those rows must never be read.",
            "Score the unchanged incumbent model on the retained rows and report GAUC and nDCG@5 as commentary.",
            "Leave the training objective, the checkpoint selection metric and the reported validation primary untouched.",
        ),
        hyperparameters={
            "diagnostic_split": ["log_random_4_22_to_5_08_pure"],
            # The file is 1,186,059 rows spanning 20220422-20220508, but 897,721
            # of them (75.5%) fall in the official test date range. Only the
            # 288,338 rows inside the validation window are in bounds -- see the
            # date_range below and the second implementation step.
            "date_range": [[20220422, 20220428]],
            "report_only": [True],
        },
        must_hold_constant=(
            "training data", "training objective", "model capacity",
            "official split", "validation primary as the sole selection metric",
            "the 20220422-20220428 date filter on the randomized-exposure log",
        ),
        dependencies=(),
        hardware="Existing CPU environment",
        runtime_impact="Additive scoring pass only; no change to training time",
        complexity="low",
        feasibility_notes=(
            "DATA BOUNDARY: reviewed and accepted. The profile's allowed_data_boundary names only "
            "public_metadata, train and validation, and the randomized-exposure log is not one of those "
            "categories -- but read-only diagnostic use of it is the dataset's own documented purpose for "
            "the file (solution/ideas.md, unexplored direction 7), and the entry is restricted to the "
            "20220422-20220428 validation window so it never reads a test-period row. See the assumptions "
            "list in agent/research/profiles/kuairand_pure.json and docs/exploration-campaign.md item 2."
        ),
        secondary_effects={
            "GAUC": "Unchanged by construction -- the model and its selection are not modified.",
            "nDCG@5": "Unchanged by construction.",
        },
        ablation=(
            "Compare the unbiased diagnostic against the ordinary validation metric across the accepted "
            "iterations to date; a widening gap indicates fitting to exposure bias."
        ),
        failure_interpretation=(
            "This entry cannot fail on validation primary because it changes no model. A close agreement "
            "between the two metrics means exposure bias is not distorting the incumbent's gains; a large "
            "gap means later ranking gains should be treated with suspicion."
        ),
        risks=(
            "MEASURES NOTHING ON THE PRIMARY METRIC BY DESIGN: expect a delta of approximately zero, which "
            "an Evaluator will read as REVERT. That verdict is about the absence of a modelling change, not "
            "about the direction being a dead end -- see the README note before letting it reach the ledger.",
            "The randomized log spans 20220422-20220508 and 75.5% of it lies in the official held-out "
            "date range; reading those rows would cross the split boundary, which is why the entry is "
            "restricted to 20220422-20220428, the official validation window.",
            "Its user and video coverage differs from the training log, so absolute numbers are not directly comparable.",
        ),
    ),
)


@dataclass
class OfflineResearchAgent:
    """History-aware deterministic implementation of ``ResearchAgent``."""

    citation_source: CitationSource = field(default_factory=JsonCitationCatalog)
    backlog: Sequence[BacklogEntry] = DEFAULT_BACKLOG
    convergence: ConvergenceConfig = DEFAULT_CONFIG.convergence
    # Cross-run Do/Don't ledger (agent/research/findings.py). Optional: None
    # preserves every existing caller/test, same convention as
    # Orchestrator.findings. Without it, select_proposal only dedupes against
    # THIS run's own history (via _validate_proposal_against_context below),
    # which is why two separate runs of the same backlog produced the
    # identical OFFLINE-HYBRID-BPR / OFFLINE-GAUC-WEIGHTED-BPR /
    # OFFLINE-DIN-SHORT-HISTORY sequence: each run's history started empty,
    # so nothing told it those exact backlog entries were already measured by
    # a PRIOR run. With it, a backlog entry whose id already appears as a
    # variant in the ledger is skipped the same way an in-run duplicate is.
    findings: Optional[FindingsLedger] = None

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

        known_ids = self._known_finding_ids()

        rejected_duplicates = 0
        for entry in viable:
            try:
                proposal = entry.build(context)
                validate_proposal_citations(proposal, self.citation_source)
                _validate_proposal_against_context(proposal, context, history)
                if proposal.hypothesis_id.casefold() in known_ids:
                    raise DuplicateHypothesisError(
                        f"proposal.hypothesis_id {proposal.hypothesis_id!r} already recorded "
                        "in the cross-run findings ledger by an earlier run"
                    )
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
            f"budget ({len(viable)} feasible, {rejected_duplicates} already attempted)"
        )

    def _known_finding_ids(self) -> set[str]:
        """Every proposal id any PRIOR run's ledger entry already covers.

        A finding's `variants` are exactly the proposal ids merged into that
        family (see findings.py's _merge) plus its own `direction` key, so
        this is the full set of backlog entries this run must not re-propose
        as if they were untried.
        """
        if self.findings is None:
            return set()
        ids: set[str] = set()
        for finding in self.findings.load():
            ids.add(finding.direction.casefold())
            ids.update(v.casefold() for v in finding.variants)
        return ids
