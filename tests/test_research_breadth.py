"""Focused tests for LLM Research breadth-then-depth selection."""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from agent.coding.llm import ScriptedClient
from agent.config import ConvergenceConfig
from agent.records import AggregateMetrics, Decision, ResourceUsage, RunRecord, Status
from agent.research.agent import LLMResearchAgent, ResearchOutputError
from agent.research.breadth import (
    BreadthCandidate,
    BreadthPlan,
    BreadthValidationError,
    StackStage,
    build_stack_coverage,
    classify_stack_stage,
    filter_breadth_candidates,
    infer_mechanism_signature,
    rank_breadth_candidates,
    validate_depth_alignment,
    validate_research_direction_safety,
)
from agent.research.citations import CitationClaim, CitationRecord, JsonCitationCatalog
from agent.research.context import build_research_context
from agent.research.schemas import ResearchProposal


def _record(
    iteration: int,
    hypothesis: str,
    decision: Decision,
    *,
    primary: float = 0.60,
    parent: int | None = 0,
    status: Status = Status.SUCCESS,
) -> RunRecord:
    return RunRecord(
        iteration=iteration,
        parent_iteration=None if iteration == 0 else parent,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        hypothesis=hypothesis,
        diff_path=f"solutions/{iteration}/config.json",
        status=status,
        seeds=[],
        aggregate=AggregateMetrics(
            primary_mean=primary,
            primary_std=0.001,
            gauc_mean=primary + 0.04,
            ndcg5_mean=primary - 0.04,
            n_seeds=1,
        ),
        delta_vs_current_best=primary - 0.60,
        decision=decision,
        events=[],
        resources=ResourceUsage(wall_s=10.0),
    )


def _candidate(
    candidate_id: str,
    stage: StackStage,
    *,
    title: str | None = None,
    primary_change: str | None = None,
    mechanism: str | None = None,
    metric_rationale: str = "Improve GAUC and nDCG@5 through a better ranking signal.",
    upside: str = "medium",
    risk: str = "medium",
    cost: str = "medium",
    evidence: list[dict] | None = None,
) -> dict:
    defaults = {
        StackStage.FEATURES: "Add content metadata features for each candidate item.",
        StackStage.ARCHITECTURE: "Add a compact MLP tower architecture.",
        StackStage.OBJECTIVE_SAMPLING: "Blend retained pointwise supervision with within-user pairwise ordering.",
        StackStage.OPTIMIZATION_REGULARIZATION: "Add dropout and learning-rate scheduling.",
        StackStage.INFERENCE_ENSEMBLE: "Use prediction averaging in a checkpoint ensemble.",
    }
    primary_defaults = {
        StackStage.FEATURES: "Add candidate metadata features.",
        StackStage.ARCHITECTURE: "Add a compact MLP tower architecture.",
        StackStage.OBJECTIVE_SAMPLING: "Add a weighted BPR pairwise objective.",
        StackStage.OPTIMIZATION_REGULARIZATION: "Apply embedding dropout regularization.",
        StackStage.INFERENCE_ENSEMBLE: "Use prediction averaging.",
    }
    return {
        "candidate_id": candidate_id,
        "title": title or f"{stage.value} direction",
        "stack_stage": stage.value,
        "primary_change": primary_change or primary_defaults[stage],
        "mechanism": mechanism or defaults[stage],
        "metric_rationale": metric_rationale,
        "expected_upside": upside,
        "implementation_risk": risk,
        "experiment_cost": cost,
        "evidence": evidence or [{
            "citation_id": "rendle2009bpr",
            "claim_id": "pairwise-ranking-objective",
        }],
    }


def _breadth(*candidates: dict) -> str:
    return json.dumps({"schema_version": 1, "candidates": list(candidates)})


def _complete_breadth(primary: dict) -> str:
    """Supply the three genuinely competing directions required by the contract."""
    fillers = [
        _candidate("B-FILLER-FEATURES", StackStage.FEATURES),
        _candidate("B-FILLER-ARCH", StackStage.ARCHITECTURE),
    ]
    used = {primary["stack_stage"]}
    selected = [primary]
    for filler in fillers:
        if filler["stack_stage"] not in used:
            selected.append(filler)
            used.add(filler["stack_stage"])
    if len(selected) < 3:
        selected.append(_candidate("B-FILLER-ENSEMBLE", StackStage.INFERENCE_ENSEMBLE))
    return _breadth(*selected)


def _objective_proposal(parent: int = 0) -> dict:
    return {
        "schema_version": 1,
        "hypothesis_id": "H-HYBRID",
        "parent_iteration": parent,
        "title": "Weighted BPR objective addition",
        "hypothesis": (
            "Blend retained pointwise supervision with a weighted within-user pairwise "
            "term to improve ordering."
        ),
        "rationale": {
            "mechanism": "Pairwise ordering pressure complements the incumbent pointwise signal.",
            "metric_alignment": ["GAUC", "nDCG@5"],
            "prior_results_used": [parent],
            "evidence": [{
                "citation_id": "rendle2009bpr",
                "claim_id": "pairwise-ranking-objective",
                "application": "Use within-user positive-negative ordering while retaining pointwise supervision.",
            }],
        },
        "implementation": {
            "target_components": ["training objective", "within-user sampler"],
            "steps": [
                "Retain pointwise supervision.",
                "Add a weighted within-user pairwise loss term.",
            ],
            "hyperparameters": {"pairwise_weight": [0.05, 0.1]},
            "must_hold_constant": ["features", "architecture", "evaluation"],
            "feasibility": {
                "dependencies": [],
                "hardware": "Existing CPU environment",
                "estimated_runtime_impact": "At most 1.5x incumbent time",
                "implementation_complexity": "medium",
                "notes": "Use vectorized pair construction.",
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
            "ablation": "Compare pairwise_weight=0 with nonzero values.",
            "failure_interpretation": "A loss indicates the pairwise addition is misaligned.",
        },
        "risks": ["Pair construction may increase runtime."],
    }


def _agent(tmp_path, responses, **kwargs):
    client = ScriptedClient(list(responses))
    agent = LLMResearchAgent(
        llm=client,
        citation_source=JsonCitationCatalog(),
        usage_log_path=tmp_path / "research_usage.jsonl",
        **kwargs,
    )
    return agent, client


def _baseline() -> RunRecord:
    return _record(
        0,
        "Baseline factorization machine with pointwise logloss.",
        Decision.ACCEPT,
        parent=None,
    )


class _Catalog:
    def __init__(self, records: list[CitationRecord]):
        self.records = {record.citation_id: record for record in records}

    def get(self, citation_id):
        return self.records.get(citation_id)

    def search(self, query, limit=10):
        return list(self.records.values())[:limit]


def _evidence_catalog() -> _Catalog:
    return _Catalog([
        CitationRecord(
            citation_id="paper-a",
            title="Pairwise BPR optimization",
            authors=("A",), year=2020, venue="Test", url="https://example/a",
            claims=tuple(
                CitationClaim(f"claim-{index}", "BPR uses a pairwise ranking objective.")
                for index in range(1, 4)
            ),
            tags=("pairwise-loss",),
        ),
        CitationRecord(
            citation_id="paper-b",
            title="Pairwise objectives for ranking",
            authors=("B",), year=2021, venue="Test", url="https://example/b",
            claims=(CitationClaim("claim-1", "A pairwise loss optimizes item ordering."),),
            tags=("pairwise-loss",),
        ),
    ])


def test_successful_propose_runs_breadth_then_depth_and_returns_idea(tmp_path):
    objective = _candidate(
        "B-OBJECTIVE",
        StackStage.OBJECTIVE_SAMPLING,
        evidence=[
            {"citation_id": "rendle2009bpr", "claim_id": "pairwise-ranking-objective"},
            {"citation_id": "covington2016youtube", "claim_id": "ranking-objective-and-watch-time"},
        ],
    )
    breadth = _complete_breadth(objective)
    agent, client = _agent(tmp_path, [breadth, json.dumps(_objective_proposal())])

    idea = agent.propose([_baseline()])

    assert idea.parent_iteration == 0
    assert idea.hypothesis.startswith("[RESEARCH_PROPOSAL v1]")
    assert [call[2] for call in client.calls] == ["research_breadth", "research_depth"]
    assert "Selected breadth direction (binding)" in client.calls[1][1]


def test_breadth_candidate_count_is_hard_bounded(tmp_path):
    stages = list(StackStage)
    candidates = [
        _candidate(f"B-{index}", stages[index % len(stages)])
        for index in range(6)
    ]
    agent, client = _agent(
        tmp_path,
        [_breadth(*candidates)],
        max_repair_attempts=0,
        breadth_candidate_count=5,
    )

    with pytest.raises(ResearchOutputError, match="exceeds maximum 5"):
        agent.propose([_baseline()])

    assert [call[2] for call in client.calls] == ["research_breadth"]


@pytest.mark.parametrize(
    ("hypothesis", "expected"),
    [
        ("Add content features from video metadata and tags.", StackStage.FEATURES),
        ("Use a DeepFM cross-feature interaction tower.", StackStage.ARCHITECTURE),
        ("Train with BPR pairwise loss and hard negative sampling.", StackStage.OBJECTIVE_SAMPLING),
        ("Tune dropout regularization and a learning rate scheduler.", StackStage.OPTIMIZATION_REGULARIZATION),
        ("Blend scores from an ensemble at inference.", StackStage.INFERENCE_ENSEMBLE),
    ],
)
def test_stack_stage_classification(hypothesis, expected):
    assert classify_stack_stage(hypothesis) == expected


def test_coverage_summary_tracks_objective_and_architecture_outcomes():
    history = [
        _baseline(),
        _record(1, "BPR pairwise loss experiment.", Decision.REVERT),
        _record(2, "DeepFM feature interaction tower.", Decision.ACCEPT),
        _record(3, "Wider DeepFM hidden layer architecture.", Decision.REVERT, parent=2),
    ]

    coverage = build_stack_coverage(history)

    objective = coverage.for_stage(StackStage.OBJECTIVE_SAMPLING)
    architecture = coverage.for_stage(StackStage.ARCHITECTURE)
    assert (objective.attempts, objective.reverted) == (1, 1)
    assert (architecture.attempts, architecture.accepted, architecture.reverted) == (2, 1, 1)
    assert coverage.for_stage(StackStage.FEATURES).attempts == 0


def test_soft_diversity_prefers_unexplored_stage_when_candidates_are_similar_strength():
    history = [
        _baseline(),
        _record(1, "DeepFM feature interaction architecture.", Decision.ACCEPT),
        _record(2, "Cross network architecture with hidden layers.", Decision.REVERT, parent=1),
    ]
    context = build_research_context(history)
    coverage = build_stack_coverage(history)
    architecture = BreadthCandidate.from_dict(
        _candidate("B-ARCH", StackStage.ARCHITECTURE), "candidate"
    )
    features = BreadthCandidate.from_dict(
        _candidate("B-FEATURE", StackStage.FEATURES), "candidate"
    )

    ranked = rank_breadth_candidates(
        [architecture, features], history=history, coverage=coverage, context=context,
        citation_source=JsonCitationCatalog(),
    )

    assert ranked[0].candidate is features
    assert ranked[0].coverage > ranked[1].coverage


def test_self_reported_upside_cannot_game_objective_ranking_signals():
    history = [
        _baseline(),
        _record(1, "DeepFM feature interaction architecture.", Decision.ACCEPT),
        _record(2, "Cross network architecture with hidden layers.", Decision.REVERT, parent=1),
    ]
    context = build_research_context(history)
    coverage = build_stack_coverage(history)
    strong_architecture = BreadthCandidate.from_dict(
        _candidate(
            "B-ARCH-STRONG",
            StackStage.ARCHITECTURE,
            upside="high",
            risk="low",
            cost="low",
            evidence=[
                {"citation_id": "guo2017deepfm", "claim_id": "joint-low-high-order-interactions"},
                {"citation_id": "zhou2018din", "claim_id": "candidate-conditioned-interest"},
            ],
        ),
        "candidate",
    )
    weak_features = BreadthCandidate.from_dict(
        _candidate(
            "B-FEATURE-WEAK",
            StackStage.FEATURES,
            upside="low",
            risk="high",
            cost="high",
        ),
        "candidate",
    )

    ranked = rank_breadth_candidates(
        [weak_features, strong_architecture],
        history=history,
        coverage=coverage,
        context=context,
        citation_source=JsonCitationCatalog(),
    )

    assert ranked[0].candidate is weak_features
    assert ranked[0].coverage > ranked[1].coverage


def test_filter_rejects_current_run_and_intra_batch_duplicates():
    duplicate = _candidate("B-1", StackStage.FEATURES)
    duplicate_text = "\n".join([
        duplicate["title"], duplicate["mechanism"], duplicate["metric_rationale"]
    ])
    history = [_baseline(), _record(1, duplicate_text, Decision.REVERT)]
    same_batch_a = _candidate("B-2", StackStage.ARCHITECTURE)
    same_batch_b = dict(same_batch_a, candidate_id="B-3")
    plan = BreadthPlan.from_json(
        _breadth(duplicate, same_batch_a, same_batch_b),
        max_candidates=5,
    )

    survivors, rejections = filter_breadth_candidates(
        plan,
        history=history,
        citation_source=JsonCitationCatalog(),
    )

    assert [candidate.candidate_id for candidate in survivors] == ["B-2"]
    reasons = {item.candidate_id: item.reason for item in rejections}
    assert reasons["B-1"] == "duplicates current-run history"
    assert reasons["B-3"] == "duplicates breadth batch candidate"


def test_filter_rejects_hidden_test_disallowed_and_invalid_citation_candidates():
    valid = _candidate("B-VALID", StackStage.FEATURES)
    leaking = _candidate(
        "B-LEAK",
        StackStage.FEATURES,
        mechanism="Select features using the private leaderboard result.",
    )
    disallowed = _candidate(
        "B-EVAL",
        StackStage.OPTIMIZATION_REGULARIZATION,
        mechanism="Modify the official evaluation metric to reward calibrated scores.",
    )
    invalid_citation = _candidate(
        "B-CITATION",
        StackStage.ARCHITECTURE,
        evidence=[{"citation_id": "invented", "claim_id": "unsupported"}],
    )
    plan = BreadthPlan.from_json(
        _breadth(valid, leaking, disallowed, invalid_citation),
        max_candidates=5,
    )

    survivors, rejections = filter_breadth_candidates(
        plan,
        history=[_baseline()],
        citation_source=JsonCitationCatalog(),
    )

    assert [candidate.candidate_id for candidate in survivors] == ["B-VALID"]
    assert {item.candidate_id for item in rejections} == {
        "B-LEAK", "B-EVAL", "B-CITATION",
    }


def test_all_filtered_breadth_candidates_receive_one_repair_before_depth(tmp_path):
    duplicate = _candidate("B-DUPLICATE", StackStage.FEATURES)
    duplicate_text = "\n".join([
        duplicate["title"], duplicate["mechanism"], duplicate["metric_rationale"]
    ])
    history = [_baseline(), _record(1, duplicate_text, Decision.REVERT)]
    repaired = _candidate(
        "B-OBJECTIVE",
        StackStage.OBJECTIVE_SAMPLING,
        evidence=[
            {"citation_id": "rendle2009bpr", "claim_id": "pairwise-ranking-objective"},
            {"citation_id": "covington2016youtube", "claim_id": "ranking-objective-and-watch-time"},
        ],
    )
    duplicate_batch = [
        dict(duplicate, candidate_id=f"B-DUPLICATE-{index}") for index in range(3)
    ]
    repaired_batch = _breadth(
        repaired,
        _candidate("B-REPAIRED-ARCH", StackStage.ARCHITECTURE),
        _candidate("B-REPAIRED-ENSEMBLE", StackStage.INFERENCE_ENSEMBLE),
    )
    agent, client = _agent(tmp_path, [
        _breadth(*duplicate_batch),
        repaired_batch,
        json.dumps(_objective_proposal()),
    ])

    idea = agent.propose(history)

    assert idea.parent_iteration == 0
    assert [call[2] for call in client.calls] == [
        "research_breadth",
        "research_breadth_repair",
        "research_depth",
    ]
    assert "post-filter survivors" in client.calls[1][1]


def test_depth_alignment_rejects_switch_to_unrelated_stage():
    selected = BreadthCandidate.from_dict(
        _candidate(
            "B-FEATURE",
            StackStage.FEATURES,
            title="Candidate metadata features",
            mechanism="Add candidate video metadata and content features.",
        ),
        "candidate",
    )
    unrelated = ResearchProposal.from_dict(_objective_proposal())

    with pytest.raises(BreadthValidationError, match="changed the selected stack stage"):
        validate_depth_alignment(unrelated, selected)


def test_malformed_breadth_has_one_bounded_repair_and_never_reaches_depth(tmp_path):
    agent, client = _agent(tmp_path, ["not json", "still not json"])

    with pytest.raises(ResearchOutputError, match="breadth phase failed after 2 call"):
        agent.propose([_baseline()])

    assert [call[2] for call in client.calls] == [
        "research_breadth",
        "research_breadth_repair",
    ]


def test_real_run_pattern_prefers_strong_unexplored_features_direction():
    history = [
        _baseline(),
        _record(1, "Replace logloss with BPR pairwise objective.", Decision.REVERT, primary=0.3956),
        _record(2, "Add a DeepFM feature interaction architecture.", Decision.ACCEPT, primary=0.6035),
        _record(3, "Use wider and deeper DeepFM hidden layers.", Decision.REVERT, primary=0.6035, parent=2),
    ]
    coverage = build_stack_coverage(history)
    context = build_research_context(history, ConvergenceConfig(max_iterations=10))
    architecture = BreadthCandidate.from_dict(
        _candidate(
            "B-ARCH-FOLLOWUP",
            StackStage.ARCHITECTURE,
            mechanism="Add a residual cross network interaction tower.",
        ),
        "candidate",
    )
    features = BreadthCandidate.from_dict(
        _candidate(
            "B-CONTENT-FEATURES",
            StackStage.FEATURES,
            mechanism="Add candidate video metadata and content feature embeddings.",
        ),
        "candidate",
    )

    ranked = rank_breadth_candidates(
        [architecture, features], history=history, coverage=coverage, context=context,
        citation_source=JsonCitationCatalog(),
    )

    assert coverage.for_stage(StackStage.ARCHITECTURE).attempts == 2
    assert coverage.for_stage(StackStage.OBJECTIVE_SAMPLING).attempts == 1
    assert coverage.for_stage(StackStage.FEATURES).attempts == 0
    assert ranked[0].candidate is features


@pytest.mark.parametrize("count", [0, 1, 2])
def test_breadth_plan_rejects_too_few_competing_directions(count):
    candidates = [
        _candidate(f"B-{index}", list(StackStage)[index % len(StackStage)])
        for index in range(count)
    ]
    with pytest.raises(BreadthValidationError, match="at least 3"):
        BreadthPlan.from_json(_breadth(*candidates), max_candidates=8)


@pytest.mark.parametrize("count", [3, 5, 8])
def test_breadth_plan_accepts_supported_candidate_boundaries(count):
    candidates = [
        _candidate(f"B-{index}", list(StackStage)[index % len(StackStage)])
        for index in range(count)
    ]
    assert len(BreadthPlan.from_json(
        _breadth(*candidates), max_candidates=8
    ).candidates) == count


def test_absolute_breadth_maximum_cannot_be_bypassed_by_caller():
    candidates = [
        _candidate(f"B-{index}", list(StackStage)[index % len(StackStage)])
        for index in range(9)
    ]
    with pytest.raises(BreadthValidationError, match="absolute maximum 8"):
        BreadthPlan.from_json(_breadth(*candidates), max_candidates=99)


def test_agent_configuration_also_enforces_real_breadth(tmp_path):
    with pytest.raises(ValueError, match="between 3 and 8"):
        _agent(tmp_path, [], breadth_candidate_count=1)


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"schema_version": True, "candidates": []}, "schema_version"),
        ({
            "schema_version": 1,
            "candidates": [
                _candidate("B-SAME", StackStage.FEATURES),
                _candidate("b-same", StackStage.ARCHITECTURE),
                _candidate("B-THIRD", StackStage.OBJECTIVE_SAMPLING),
            ],
        }, "duplicate candidate IDs"),
    ],
)
def test_breadth_schema_rejects_boolean_version_and_duplicate_ids(payload, match):
    with pytest.raises(BreadthValidationError, match=match):
        BreadthPlan.from_json(json.dumps(payload), max_candidates=8)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("candidate_id", "B" * 65, "candidate_id exceeds"),
        ("title", "T" * 201, "title exceeds"),
        ("primary_change", "P" * 401, "primary_change exceeds"),
        ("mechanism", "M" * 1201, "mechanism exceeds"),
        ("metric_rationale", "R" * 801, "metric_rationale exceeds"),
        ("stack_stage", "not-a-stage", "stack_stage must be one of"),
        ("expected_upside", 3, "non-empty string"),
    ],
)
def test_breadth_schema_rejects_oversized_or_malformed_fields(field, value, match):
    candidate = _candidate("B-BAD", StackStage.FEATURES)
    candidate[field] = value
    with pytest.raises(BreadthValidationError, match=match):
        BreadthPlan.from_json(_breadth(
            candidate,
            _candidate("B-ARCH", StackStage.ARCHITECTURE),
            _candidate("B-OBJECTIVE", StackStage.OBJECTIVE_SAMPLING),
        ), max_candidates=8)


def test_breadth_schema_rejects_duplicate_evidence_pairs():
    candidate = _candidate("B-BAD", StackStage.FEATURES)
    candidate["evidence"] = [candidate["evidence"][0], deepcopy(candidate["evidence"][0])]
    with pytest.raises(BreadthValidationError, match="evidence contains duplicates"):
        BreadthPlan.from_json(_breadth(
            candidate,
            _candidate("B-ARCH", StackStage.ARCHITECTURE),
            _candidate("B-OBJECTIVE", StackStage.OBJECTIVE_SAMPLING),
        ), max_candidates=8)


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "Train on a test label with temporal activity features.",
        "Train on test labels with temporal activity features.",
        "Tune using final hold-out results for temporal activity features.",
        "Optimize based on hidden test outcomes.",
        "Use feedback from the public leaderboard.",
        "Private leaderboard tuning for temporal features.",
        "Rewrite the official scorer.",
        "Alter the scoring script.",
        "Optimize using the final holdout with temporal activity features.",
        "Use public leaderboard feedback to select temporal activity features.",
        "Tune the scoring script while adding temporal activity features.",
        "Modify evaluate.py to reward temporal activity features.",
        "Choose hyperparameters from test performance for temporal activity features.",
        "Select parameters based on test performance.",
    ],
)
def test_breadth_filter_rejects_forbidden_development_strategies(unsafe_text):
    unsafe = _candidate(
        "B-UNSAFE", StackStage.FEATURES, mechanism=unsafe_text
    )
    plan = BreadthPlan.from_json(_breadth(
        unsafe,
        _candidate("B-ARCH", StackStage.ARCHITECTURE),
        _candidate("B-OBJECTIVE", StackStage.OBJECTIVE_SAMPLING),
    ), max_candidates=8)
    survivors, rejections = filter_breadth_candidates(
        plan, history=[_baseline()], citation_source=JsonCitationCatalog()
    )
    assert "B-UNSAFE" not in {candidate.candidate_id for candidate in survivors}
    assert "unsafe Research direction" in {
        item.candidate_id: item.reason for item in rejections
    }["B-UNSAFE"]


@pytest.mark.parametrize(
    "safe_text",
    [
        "Add temporal activity features and evaluate on validation only.",
        "Add temporal activity features; never use test labels.",
        "Do not tune on the final holdout.",
        "Add temporal activity features and keep the official scoring script unchanged.",
        "Preserve evaluate.py exactly.",
        "Use validation-only evaluation.",
        "Add temporal activity features using a train/validation split.",
        "Reproduce the official metric locally on validation data for temporal features.",
    ],
)
def test_safety_boundary_allows_validation_only_and_negated_language(safe_text):
    validate_research_direction_safety({"mechanism": safe_text}, origin="test")


def test_safety_boundary_cannot_be_bypassed_by_splitting_phrase_across_fields():
    with pytest.raises(BreadthValidationError, match="unsafe Research direction"):
        validate_research_direction_safety(
            {"title": "Train on", "mechanism": "test labels for temporal features"},
            origin="test",
        )


def test_negation_does_not_neutralize_a_later_forbidden_action():
    with pytest.raises(BreadthValidationError, match="unsafe Research direction"):
        validate_research_direction_safety(
            {"mechanism": "Never use test labels but tune with leaderboard feedback."},
            origin="test",
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Temporal activity features and user history statistics", StackStage.FEATURES),
        ("Video duration interactions and feature crosses", StackStage.FEATURES),
        ("DeepFM with temporal wording", StackStage.ARCHITECTURE),
        ("DCN deep cross network", StackStage.ARCHITECTURE),
        ("BPR with hard negative sampling", StackStage.OBJECTIVE_SAMPLING),
        ("A listwise objective", StackStage.OBJECTIVE_SAMPLING),
        ("Embedding dropout and weight decay", StackStage.OPTIMIZATION_REGULARIZATION),
        ("AdamW with gradient clipping", StackStage.OPTIMIZATION_REGULARIZATION),
        ("Prediction averaging over a checkpoint ensemble", StackStage.INFERENCE_ENSEMBLE),
        ("Embedding", None),
        ("DeepFM with BPR", None),
        (
            "Use DeepFM architecture with temporal features, feature crosses, "
            "pairwise BPR loss, and embedding dropout.",
            None,
        ),
    ],
)
def test_token_phrase_stage_classifier_adversarial_cases(text, expected):
    assert classify_stack_stage(text) == expected


def test_declared_stage_must_match_confident_mechanism_inference():
    candidates = [
        _candidate(
            "B-MISLABEL-DEEPFM", StackStage.FEATURES,
            primary_change="Add a DeepFM interaction architecture.",
            mechanism="Add a DeepFM interaction tower architecture.",
        ),
        _candidate(
            "B-MISLABEL-BPR", StackStage.ARCHITECTURE,
            primary_change="Train a BPR pairwise objective.",
            mechanism="Train a BPR pairwise loss.",
        ),
        _candidate(
            "B-REG", StackStage.OPTIMIZATION_REGULARIZATION,
            mechanism="Apply embedding dropout and weight decay.",
        ),
        _candidate(
            "B-AMBIGUOUS", StackStage.ARCHITECTURE,
            primary_change="Change the recommender representation.",
            mechanism="Change the recommender representation.",
        ),
    ]
    plan = BreadthPlan.from_json(_breadth(*candidates), max_candidates=8)
    survivors, rejections = filter_breadth_candidates(
        plan, history=[_baseline()], citation_source=JsonCitationCatalog()
    )
    assert {item.candidate_id for item in survivors} == {"B-REG"}
    reasons = {item.candidate_id: item.reason for item in rejections}
    assert "conflicts" in reasons["B-MISLABEL-DEEPFM"]
    assert "conflicts" in reasons["B-MISLABEL-BPR"]
    assert "unambiguous" in reasons["B-AMBIGUOUS"]


def _proposal_for_method(
    *, title: str, hypothesis: str, mechanism: str, components: list[str], steps: list[str]
) -> ResearchProposal:
    proposal = _objective_proposal()
    proposal["title"] = title
    proposal["hypothesis"] = hypothesis
    proposal["rationale"]["mechanism"] = mechanism
    proposal["implementation"]["target_components"] = components
    proposal["implementation"]["steps"] = steps
    return ResearchProposal.from_dict(proposal)


def test_depth_cannot_switch_temporal_features_to_deepfm_with_superficial_words():
    selected = BreadthCandidate.from_dict(_candidate(
        "B-TEMPORAL", StackStage.FEATURES,
        title="Temporal user activity features",
        mechanism="Add temporal activity features and user history statistics.",
    ), "candidate")
    proposal = _proposal_for_method(
        title="DeepFM interaction architecture",
        hypothesis="Add DeepFM while retaining temporal user activity inputs.",
        mechanism="A DeepFM interaction tower learns nonlinear feature interactions.",
        components=["DeepFM architecture"],
        steps=["Build a DeepFM interaction tower with the temporal inputs."],
    )
    with pytest.raises(BreadthValidationError, match="changed the selected stack stage"):
        validate_depth_alignment(proposal, selected)


def test_depth_accepts_embedding_regularization_elaboration():
    selected = BreadthCandidate.from_dict(_candidate(
        "B-REG", StackStage.OPTIMIZATION_REGULARIZATION,
        title="Embedding regularization",
        mechanism="Regularize embedding layers.",
    ), "candidate")
    proposal = _proposal_for_method(
        title="Embedding dropout and weight decay",
        hypothesis="Apply dropout and weight decay to embedding layers.",
        mechanism="Embedding dropout plus weight decay controls co-adaptation.",
        components=["embedding layers"],
        steps=["Apply embedding dropout and weight decay during training."],
    )
    validate_depth_alignment(proposal, selected)


def test_citation_overlap_alone_does_not_prove_depth_alignment():
    selected = BreadthCandidate.from_dict(_candidate(
        "B-BPR", StackStage.OBJECTIVE_SAMPLING,
        title="Pairwise hard-negative BPR",
        mechanism="Use BPR pairwise loss with hard negative sampling.",
    ), "candidate")
    proposal = _proposal_for_method(
        title="Listwise LambdaRank objective",
        hypothesis="Replace training with a listwise objective.",
        mechanism="LambdaRank supplies listwise gradients.",
        components=["listwise objective"],
        steps=["Train a listwise loss function."],
    )
    proposal_data = proposal.to_dict()
    proposal_data["implementation"]["hyperparameters"] = {"listwise_temperature": [1.0]}
    proposal = ResearchProposal.from_dict(proposal_data)
    with pytest.raises(BreadthValidationError, match="primary mechanism family"):
        validate_depth_alignment(proposal, selected)


def test_distinct_sources_not_claim_pairs_determine_evidence_score():
    one_source = BreadthCandidate.from_dict(_candidate(
        "B-ONE", StackStage.OBJECTIVE_SAMPLING,
        evidence=[
            {"citation_id": "paper-a", "claim_id": "claim-1"},
            {"citation_id": "paper-a", "claim_id": "claim-2"},
            {"citation_id": "paper-a", "claim_id": "claim-3"},
        ],
    ), "candidate")
    two_sources = BreadthCandidate.from_dict(_candidate(
        "B-TWO", StackStage.OBJECTIVE_SAMPLING,
        evidence=[
            {"citation_id": "paper-a", "claim_id": "claim-1"},
            {"citation_id": "paper-b", "claim_id": "claim-1"},
        ],
    ), "candidate")
    history = [_baseline()]
    scores = {
        score.candidate.candidate_id: score
        for score in rank_breadth_candidates(
            [one_source, two_sources],
            history=history,
            coverage=build_stack_coverage(history),
            context=build_research_context(history),
            citation_source=_evidence_catalog(),
        )
    }
    assert scores["B-ONE"].evidence == 2.25
    assert scores["B-TWO"].evidence == 4.5


def test_duplicate_survivor_uses_quality_not_id_or_input_order():
    weak = _candidate(
        "B-A-WEAK", StackStage.ARCHITECTURE,
        title="DeepFM interaction architecture",
        primary_change="Add a DeepFM interaction architecture.",
        mechanism="Add a DeepFM interaction tower architecture.",
    )
    strong = _candidate(
        "B-Z-STRONG", StackStage.ARCHITECTURE,
        title="DeepFM interaction architecture",
        primary_change="Add a DeepFM interaction architecture.",
        mechanism="Add a DeepFM interaction tower architecture.",
        evidence=[
            {"citation_id": "guo2017deepfm", "claim_id": "joint-low-high-order-interactions"},
            {"citation_id": "zhou2018din", "claim_id": "candidate-conditioned-interest"},
        ],
    )
    third = _candidate("B-THIRD", StackStage.FEATURES)
    semantic_sets = []
    for batch in ((weak, strong, third), (third, strong, weak)):
        plan = BreadthPlan.from_json(_breadth(*batch), max_candidates=8)
        survivors, _ = filter_breadth_candidates(
            plan, history=[_baseline()], citation_source=JsonCitationCatalog()
        )
        semantic_sets.append({item.candidate_id for item in survivors})
    assert semantic_sets[0] == semantic_sets[1]
    assert "B-Z-STRONG" in semantic_sets[0]
    assert "B-A-WEAK" not in semantic_sets[0]


def test_generic_recommender_words_do_not_collapse_distinct_methods():
    plan = BreadthPlan.from_json(_breadth(
        _candidate(
            "B-BPR", StackStage.OBJECTIVE_SAMPLING,
            mechanism="Use a BPR pairwise loss with hard negative sampling for ranking users.",
        ),
        _candidate(
            "B-DEEPFM", StackStage.ARCHITECTURE,
            primary_change="Add a DeepFM interaction architecture.",
            mechanism="Use a DeepFM interaction tower architecture for ranking users.",
        ),
        _candidate(
            "B-TEMPORAL", StackStage.FEATURES,
            primary_change="Add temporal activity features.",
            mechanism="Add temporal activity features for ranking users.",
        ),
    ), max_candidates=8)
    survivors, _ = filter_breadth_candidates(
        plan, history=[_baseline()], citation_source=JsonCitationCatalog()
    )
    assert {item.candidate_id for item in survivors} == {
        "B-BPR", "B-DEEPFM", "B-TEMPORAL",
    }


def _strong_objective_breadth() -> str:
    return _complete_breadth(_candidate(
        "B-OBJECTIVE",
        StackStage.OBJECTIVE_SAMPLING,
        evidence=[
            {"citation_id": "rendle2009bpr", "claim_id": "pairwise-ranking-objective"},
            {"citation_id": "covington2016youtube", "claim_id": "ranking-objective-and-watch-time"},
        ],
    ))


def test_final_depth_proposal_is_safety_scanned(tmp_path):
    unsafe = _objective_proposal()
    unsafe["hypothesis"] = (
        "Use public leaderboard feedback to choose the BPR pairwise weight."
    )
    agent, client = _agent(
        tmp_path,
        [_strong_objective_breadth(), json.dumps(unsafe)],
        max_repair_attempts=0,
    )
    with pytest.raises(ResearchOutputError, match="unsafe Research direction"):
        agent.propose([_baseline()])
    assert [call[2] for call in client.calls] == ["research_breadth", "research_depth"]


def test_unsafe_final_depth_can_be_repaired_once_through_same_boundary(tmp_path):
    unsafe = _objective_proposal()
    unsafe["implementation"]["steps"].append(
        "Choose hyperparameters from test performance."
    )
    agent, client = _agent(tmp_path, [
        _strong_objective_breadth(),
        json.dumps(unsafe),
        json.dumps(_objective_proposal()),
    ])
    idea = agent.propose([_baseline()])
    assert idea.parent_iteration == 0
    assert [call[2] for call in client.calls] == [
        "research_breadth", "research_depth", "research_depth_repair",
    ]


def test_too_small_breadth_uses_one_bounded_repair(tmp_path):
    too_small = _breadth(_candidate("B-ONLY", StackStage.OBJECTIVE_SAMPLING))
    agent, client = _agent(tmp_path, [
        too_small,
        _strong_objective_breadth(),
        json.dumps(_objective_proposal()),
    ])
    idea = agent.propose([_baseline()])
    assert idea.parent_iteration == 0
    assert [call[2] for call in client.calls] == [
        "research_breadth", "research_breadth_repair", "research_depth",
    ]


def test_too_small_breadth_repair_failure_stays_bounded(tmp_path):
    too_small = _breadth(_candidate("B-ONLY", StackStage.OBJECTIVE_SAMPLING))
    agent, client = _agent(tmp_path, [too_small, too_small])
    with pytest.raises(ResearchOutputError, match="breadth phase failed after 2 call"):
        agent.propose([_baseline()])
    assert [call[2] for call in client.calls] == [
        "research_breadth", "research_breadth_repair",
    ]


def test_misaligned_depth_uses_one_bounded_repair(tmp_path):
    switched = _proposal_for_method(
        title="Temporal activity features",
        hypothesis="Add temporal activity features and user history statistics.",
        mechanism="Temporal features expose recent activity patterns.",
        components=["temporal feature builder"],
        steps=["Add temporal activity features."],
    ).to_dict()
    agent, client = _agent(tmp_path, [
        _strong_objective_breadth(),
        json.dumps(switched),
        json.dumps(_objective_proposal()),
    ])
    idea = agent.propose([_baseline()])
    assert idea.parent_iteration == 0
    assert [call[2] for call in client.calls] == [
        "research_breadth", "research_depth", "research_depth_repair",
    ]


def test_coverage_ignores_bootstrap_and_transient_retry_but_counts_outcomes():
    baseline = _record(
        0, "DeepFM interaction tower architecture.", Decision.ACCEPT, parent=None
    )
    transient = _record(
        1, "BPR pairwise loss.", Decision.REVERT, status=Status.FAILED
    )
    # A true retry has no terminal decision; reconstruct it explicitly.
    transient = RunRecord(**{
        **transient.__dict__,
        "decision": None,
    })
    accepted = _record(2, "BPR pairwise loss.", Decision.ACCEPT)
    abandoned = _record(
        3, "Embedding dropout and weight decay.", Decision.ABANDON,
        status=Status.ABANDONED,
    )
    coverage = build_stack_coverage([baseline, transient, accepted, abandoned])
    assert coverage.for_stage(StackStage.ARCHITECTURE).attempts == 0
    assert coverage.for_stage(StackStage.OBJECTIVE_SAMPLING).attempts == 1
    optimization = coverage.for_stage(StackStage.OPTIMIZATION_REGULARIZATION)
    assert (optimization.attempts, optimization.failed) == (1, 1)


def test_genuinely_stronger_repeated_stage_can_still_win_soft_coverage_ranking():
    history = [
        _baseline(),
        _record(1, "DeepFM interaction tower.", Decision.ACCEPT),
    ]
    catalog_record = JsonCitationCatalog().get("zhou2018din")
    assert catalog_record is not None
    attention_followup = CitationRecord(
        citation_id="attention-followup",
        title="Candidate-conditioned attention networks",
        authors=("Researcher",), year=2022, venue="Test",
        url="https://example/attention",
        claims=(CitationClaim(
            "attention-mechanism",
            "An attention architecture conditions user interest on the candidate.",
        ),),
        tags=("attention", "candidate-conditioning"),
    )
    catalog = _Catalog([catalog_record, attention_followup])
    repeated = BreadthCandidate.from_dict(_candidate(
        "B-ATTENTION", StackStage.ARCHITECTURE,
        title="Candidate-conditioned attention architecture",
        primary_change="Use a candidate-conditioned attention architecture.",
        mechanism=(
            "Use an attention architecture with gated interest vectors and shared "
            "feature embeddings."
        ),
        upside="high", risk="low", cost="low",
        evidence=[
            {"citation_id": "zhou2018din", "claim_id": "candidate-conditioned-interest"},
            {"citation_id": "attention-followup", "claim_id": "attention-mechanism"},
        ],
    ), "candidate")
    weak_unexplored = BreadthCandidate.from_dict(_candidate(
        "B-WEAK-FEATURE", StackStage.FEATURES,
        mechanism="Add metadata features.",
        upside="low", risk="high", cost="high",
    ), "candidate")
    ranked = rank_breadth_candidates(
        [weak_unexplored, repeated],
        history=history,
        coverage=build_stack_coverage(history),
        context=build_research_context(history),
        citation_source=catalog,
    )
    assert ranked[0].candidate is repeated
    assert ranked[0].coverage < ranked[1].coverage


@pytest.mark.parametrize(
    "mechanism",
    [
        "Retain pointwise supervision and add a weighted within-user pairwise loss term.",
        "Retain pointwise supervision and add a weighted within-user pairwise loss term with coefficient 0.10.",
    ],
)
def test_same_historical_method_renamed_or_reworded_is_filtered(mechanism):
    prior = ResearchProposal.from_dict(_objective_proposal()).to_handoff_text()
    candidate = _candidate(
        "B-RENAMED", StackStage.OBJECTIVE_SAMPLING,
        title="Fresh ordering experiment",
        mechanism=mechanism,
    )
    plan = BreadthPlan.from_json(_breadth(
        candidate,
        _candidate("B-ARCH", StackStage.ARCHITECTURE),
        _candidate("B-FEATURE", StackStage.FEATURES),
    ), max_candidates=8)
    _, rejections = filter_breadth_candidates(
        plan,
        history=[_baseline(), _record(1, prior, Decision.REVERT)],
        citation_source=JsonCitationCatalog(),
    )
    assert {
        item.candidate_id: item.reason for item in rejections
    }["B-RENAMED"] == "duplicates current-run history"


def _two_survivor_breadth() -> str:
    unsafe = [
        _candidate(
            f"B-UNSAFE-{index}", StackStage.FEATURES,
            mechanism=f"Train on a test label for candidate variant {index}.",
        )
        for index in range(3)
    ]
    return _breadth(
        _candidate("B-VALID-OBJECTIVE", StackStage.OBJECTIVE_SAMPLING),
        _candidate("B-VALID-ARCH", StackStage.ARCHITECTURE),
        *unsafe,
    )


def test_two_post_filter_survivors_trigger_one_breadth_repair(tmp_path):
    agent, client = _agent(tmp_path, [
        _two_survivor_breadth(),
        _strong_objective_breadth(),
        json.dumps(_objective_proposal()),
    ])
    idea = agent.propose([_baseline()])
    assert idea.parent_iteration == 0
    assert [call[2] for call in client.calls] == [
        "research_breadth", "research_breadth_repair", "research_depth",
    ]


def test_two_post_filter_survivors_after_repair_raise_without_depth(tmp_path):
    agent, client = _agent(tmp_path, [
        _two_survivor_breadth(),
        _two_survivor_breadth(),
    ])
    with pytest.raises(ResearchOutputError, match="post-filter survivors; got 2"):
        agent.propose([_baseline()])
    assert [call[2] for call in client.calls] == [
        "research_breadth", "research_breadth_repair",
    ]


def test_breadth_and_depth_repairs_keep_maximum_call_count_at_four(tmp_path):
    switched = _proposal_for_method(
        title="Temporal activity features",
        hypothesis="Add temporal activity features and user history statistics.",
        mechanism="Temporal features expose recent activity patterns.",
        components=["temporal feature builder"],
        steps=["Add temporal activity features."],
    ).to_dict()
    agent, client = _agent(tmp_path, [
        _two_survivor_breadth(),
        _strong_objective_breadth(),
        json.dumps(switched),
        json.dumps(_objective_proposal()),
    ])
    idea = agent.propose([_baseline()])
    assert idea.parent_iteration == 0
    assert [call[2] for call in client.calls] == [
        "research_breadth",
        "research_breadth_repair",
        "research_depth",
        "research_depth_repair",
    ]


def test_phrase_stuffed_candidate_is_rejected_as_ambiguous():
    stuffed = _candidate(
        "B-STUFFED", StackStage.FEATURES,
        primary_change="Add temporal activity features.",
        mechanism=(
            "Use DeepFM architecture with temporal features, feature crosses, "
            "pairwise BPR loss, and embedding dropout."
        ),
    )
    plan = BreadthPlan.from_json(_breadth(
        stuffed,
        _candidate("B-ARCH", StackStage.ARCHITECTURE),
        _candidate("B-OBJECTIVE", StackStage.OBJECTIVE_SAMPLING),
    ), max_candidates=8)
    survivors, rejections = filter_breadth_candidates(
        plan, history=[_baseline()], citation_source=JsonCitationCatalog()
    )
    assert "B-STUFFED" not in {item.candidate_id for item in survivors}
    assert "conflicts" in {
        item.candidate_id: item.reason for item in rejections
    }["B-STUFFED"]


def test_embedding_does_not_create_a_din_or_attention_signature():
    signature = infer_mechanism_signature("Apply embedding dropout regularization.")
    assert signature.stack_stage == StackStage.OPTIMIZATION_REGULARIZATION
    assert signature.primary_family == "regularization"


def test_secondary_hard_negatives_cannot_rescue_pairwise_to_listwise_switch():
    selected = BreadthCandidate.from_dict(_candidate(
        "B-BPR-HARD", StackStage.OBJECTIVE_SAMPLING,
        primary_change="Use a BPR pairwise objective.",
        mechanism="Use BPR pairwise loss with hard negative sampling.",
    ), "candidate")
    proposal = _proposal_for_method(
        title="Listwise hard-negative objective",
        hypothesis="Use a listwise objective with hard negative sampling.",
        mechanism="LambdaRank listwise gradients use the same hard negatives.",
        components=["listwise objective", "hard negative sampler"],
        steps=["Train a listwise loss with hard negative sampling."],
    )
    data = proposal.to_dict()
    data["implementation"]["hyperparameters"] = {"listwise_temperature": [1.0]}
    with pytest.raises(BreadthValidationError, match="primary mechanism family"):
        validate_depth_alignment(ResearchProposal.from_dict(data), selected)


def test_pairwise_depth_can_refine_its_hard_negative_sampler():
    selected = BreadthCandidate.from_dict(_candidate(
        "B-BPR-HARD", StackStage.OBJECTIVE_SAMPLING,
        primary_change="Use a BPR pairwise objective.",
        mechanism="Use BPR pairwise loss with hard negative sampling.",
    ), "candidate")
    proposal = _proposal_for_method(
        title="Detailed BPR pairwise objective",
        hypothesis="Train a BPR pairwise objective with an improved hard-negative sampler.",
        mechanism="BPR orders positives above hard negatives within each user.",
        components=["BPR pairwise loss", "hard negative sampler"],
        steps=["Train the BPR pairwise loss with refined hard negative sampling."],
    )
    validate_depth_alignment(proposal, selected)


def test_valid_but_irrelevant_citation_receives_no_ranking_credit():
    relevant = BreadthCandidate.from_dict(_candidate(
        "B-RELEVANT", StackStage.OBJECTIVE_SAMPLING,
        evidence=[{"citation_id": "rendle2009bpr", "claim_id": "pairwise-ranking-objective"}],
    ), "candidate")
    irrelevant = BreadthCandidate.from_dict(_candidate(
        "B-IRRELEVANT", StackStage.OBJECTIVE_SAMPLING,
        evidence=[{"citation_id": "guo2017deepfm", "claim_id": "joint-low-high-order-interactions"}],
    ), "candidate")
    history = [_baseline()]
    scores = {
        score.candidate.candidate_id: score
        for score in rank_breadth_candidates(
            [relevant, irrelevant],
            history=history,
            coverage=build_stack_coverage(history),
            context=build_research_context(history),
            citation_source=JsonCitationCatalog(),
        )
    }
    assert scores["B-RELEVANT"].evidence == 2.25
    assert scores["B-IRRELEVANT"].evidence == 0.0


def test_lexically_novel_deepfm_rewrite_has_low_structural_novelty():
    history = [
        _baseline(),
        _record(1, "Use a DeepFM interaction architecture.", Decision.REVERT),
    ]
    rewrite = BreadthCandidate.from_dict(_candidate(
        "B-REWRITE", StackStage.ARCHITECTURE,
        title="Nonlinear latent conjunction learner",
        primary_change="Add a DeepFM architecture.",
        mechanism=(
            "Deploy DeepFM as a differently worded latent conjunction learner with "
            "ornate terminology."
        ),
    ), "candidate")
    score = rank_breadth_candidates(
        [rewrite],
        history=history,
        coverage=build_stack_coverage(history),
        context=build_research_context(history),
        citation_source=JsonCitationCatalog(),
    )[0]
    assert score.novelty < 1.0


def test_self_report_combined_advantage_is_smaller_than_one_relevant_source():
    gamer = BreadthCandidate.from_dict(_candidate(
        "B-GAMER", StackStage.OBJECTIVE_SAMPLING,
        upside="high", risk="low", cost="low",
        evidence=[{"citation_id": "guo2017deepfm", "claim_id": "joint-low-high-order-interactions"}],
    ), "candidate")
    supported = BreadthCandidate.from_dict(_candidate(
        "B-SUPPORTED", StackStage.OBJECTIVE_SAMPLING,
        upside="medium", risk="medium", cost="medium",
        evidence=[{"citation_id": "rendle2009bpr", "claim_id": "pairwise-ranking-objective"}],
    ), "candidate")
    history = [_baseline()]
    scores = {
        score.candidate.candidate_id: score
        for score in rank_breadth_candidates(
            [gamer, supported], history=history,
            coverage=build_stack_coverage(history),
            context=build_research_context(history),
            citation_source=JsonCitationCatalog(),
        )
    }
    self_report_advantage = (
        scores["B-GAMER"].upside + scores["B-GAMER"].feasibility
        - scores["B-SUPPORTED"].upside - scores["B-SUPPORTED"].feasibility
    )
    assert self_report_advantage < 2.25
    assert scores["B-SUPPORTED"].total > scores["B-GAMER"].total


def test_duplicate_semantic_survivor_is_independent_of_ids_and_permutation():
    def batch(first_id, second_id, reverse=False):
        first = _candidate(
            first_id, StackStage.ARCHITECTURE,
            title="DeepFM interaction architecture",
            primary_change="Add a DeepFM architecture.",
            mechanism="Add a DeepFM interaction architecture.",
            evidence=[{"citation_id": "guo2017deepfm", "claim_id": "joint-low-high-order-interactions"}],
        )
        second = dict(first, candidate_id=second_id)
        candidates = [first, second, _candidate("B-FEATURE", StackStage.FEATURES)]
        if reverse:
            candidates.reverse()
        plan = BreadthPlan.from_json(_breadth(*candidates), max_candidates=8)
        survivors, _ = filter_breadth_candidates(
            plan, history=[_baseline()], citation_source=JsonCitationCatalog()
        )
        return {
            (candidate.stack_stage.value, candidate.primary_change, candidate.mechanism)
            for candidate in survivors
        }

    assert batch("B-Z", "B-A") == batch("B-NEW-1", "B-NEW-2", reverse=True)


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "Choose model settings using challenge ranking feedback.",
        "Calibrate from final competition results.",
        "Choose architecture from competition rank.",
        "Refine hyperparameters using challenge scores.",
        "Pick the best model according to competition ranking.",
        "Select checkpoints from final evaluation performance.",
        "Choose the model based on hidden evaluation results.",
        "Keep the scorer unchanged, then use leaderboard feedback to choose the model.",
        "Do not change evaluate.py, but calibrate using final competition results.",
    ],
)
def test_evaluation_feedback_euphemisms_are_rejected_clause_locally(unsafe_text):
    with pytest.raises(BreadthValidationError, match="unsafe Research direction"):
        validate_research_direction_safety({"mechanism": unsafe_text}, origin="test")


def test_negated_competition_feedback_prohibition_is_allowed():
    validate_research_direction_safety(
        {"mechanism": "Do not use final competition results for model selection."},
        origin="test",
    )


@pytest.mark.parametrize(
    ("text", "stage", "family", "secondary"),
    [
        (
            "Train a BPR pairwise objective with AdamW and dropout.",
            StackStage.OBJECTIVE_SAMPLING,
            "pairwise",
            {"adamw", "dropout"},
        ),
        (
            "Use a DeepFM architecture with embedding dropout and weight decay.",
            StackStage.ARCHITECTURE,
            "deepfm",
            {"dropout", "weight_decay"},
        ),
        (
            "Apply dropout and weight decay to embedding vectors.",
            StackStage.OPTIMIZATION_REGULARIZATION,
            "regularization",
            {"dropout", "weight_decay"},
        ),
    ],
)
def test_primary_signature_separates_ancillary_training_details(
    text, stage, family, secondary
):
    signature = infer_mechanism_signature(text)
    assert (signature.stack_stage, signature.primary_family) == (stage, family)
    assert secondary.issubset(signature.secondary_tags)


@pytest.mark.parametrize(
    "text",
    [
        "Use BPR BPR BPR but make LambdaRank listwise the new objective.",
        "Use LambdaRank LambdaRank LambdaRank but make pairwise BPR the new objective.",
    ],
)
def test_repetition_cannot_resolve_same_stage_primary_family_conflicts(text):
    signature = infer_mechanism_signature(text)
    assert signature.stack_stage == StackStage.OBJECTIVE_SAMPLING
    assert signature.primary_family is None
    assert signature.ambiguous


def test_depth_rejects_listwise_primary_despite_repeated_bpr_background():
    selected = BreadthCandidate.from_dict(_candidate(
        "B-BPR", StackStage.OBJECTIVE_SAMPLING,
        primary_change="Use a BPR pairwise objective.",
        mechanism="Use BPR with hard-negative sampling.",
    ), "candidate")
    proposal = _proposal_for_method(
        title="LambdaRank listwise objective",
        hypothesis="Replace the training loss with a LambdaRank listwise objective.",
        mechanism="BPR BPR BPR is prior background; the intervention is listwise LambdaRank.",
        components=["listwise objective"],
        steps=["Train the LambdaRank listwise objective with hard negatives."],
    )
    with pytest.raises(BreadthValidationError, match="primary mechanism family"):
        validate_depth_alignment(proposal, selected)


def test_depth_rejects_pairwise_primary_despite_repeated_lambdarank_background():
    selected = BreadthCandidate.from_dict(_candidate(
        "B-LISTWISE", StackStage.OBJECTIVE_SAMPLING,
        primary_change="Use a LambdaRank listwise objective.",
        mechanism="Optimize the listwise ranking objective.",
    ), "candidate")
    proposal = _proposal_for_method(
        title="BPR pairwise objective",
        hypothesis="Replace the training loss with a BPR pairwise objective.",
        mechanism="LambdaRank LambdaRank LambdaRank is background; BPR is the intervention.",
        components=["pairwise objective"],
        steps=["Train the BPR pairwise objective."],
    )
    with pytest.raises(BreadthValidationError, match="primary mechanism family"):
        validate_depth_alignment(proposal, selected)


def test_depth_accepts_bpr_with_ancillary_adamw_dropout_and_sampler():
    selected = BreadthCandidate.from_dict(_candidate(
        "B-BPR", StackStage.OBJECTIVE_SAMPLING,
        primary_change="Use a BPR pairwise objective.",
        mechanism="Use BPR pairwise ordering.",
    ), "candidate")
    proposal = _proposal_for_method(
        title="BPR with bounded training details",
        hypothesis="Train a BPR pairwise objective with AdamW and dropout.",
        mechanism="BPR orders positives over hard negatives within each user.",
        components=["BPR objective", "hard-negative sampler"],
        steps=["Train BPR with AdamW, dropout, and hard-negative sampling."],
    )
    validate_depth_alignment(proposal, selected)


def test_depth_accepts_deepfm_with_ancillary_dropout_and_weight_decay():
    selected = BreadthCandidate.from_dict(_candidate(
        "B-DEEPFM", StackStage.ARCHITECTURE,
        primary_change="Use a DeepFM architecture.",
        mechanism="Use a DeepFM interaction architecture.",
    ), "candidate")
    proposal = _proposal_for_method(
        title="DeepFM with regularized embeddings",
        hypothesis="Use a DeepFM architecture with embedding dropout and weight decay.",
        mechanism="DeepFM learns low- and high-order feature interactions.",
        components=["DeepFM architecture"],
        steps=["Train DeepFM with embedding dropout and weight decay."],
    )
    validate_depth_alignment(proposal, selected)


def test_historical_coverage_ignores_ancillary_optimizer_and_regularizer_details():
    history = [
        _baseline(),
        _record(1, "DeepFM with ordinary embedding dropout.", Decision.REVERT),
        _record(2, "BPR trained with AdamW.", Decision.ACCEPT),
    ]
    coverage = build_stack_coverage(history)
    assert coverage.for_stage(StackStage.ARCHITECTURE).attempts == 1
    assert coverage.for_stage(StackStage.OBJECTIVE_SAMPLING).attempts == 1
    assert coverage.for_stage(StackStage.OPTIMIZATION_REGULARIZATION).attempts == 0


def test_structured_history_extracts_primary_method_before_ancillary_details():
    deepfm = _proposal_for_method(
        title="DeepFM with ordinary dropout",
        hypothesis="Use a DeepFM architecture with embedding dropout.",
        mechanism="DeepFM learns feature interactions.",
        components=["DeepFM architecture"],
        steps=["Train DeepFM with ordinary embedding dropout."],
    ).to_handoff_text()
    bpr = _proposal_for_method(
        title="BPR with AdamW",
        hypothesis="Use a BPR pairwise objective trained with AdamW.",
        mechanism="BPR provides within-user ordering pressure.",
        components=["BPR objective"],
        steps=["Train BPR with AdamW."],
    ).to_handoff_text()
    history = [
        _baseline(),
        _record(1, deepfm, Decision.REVERT),
        _record(2, bpr, Decision.ACCEPT),
    ]
    coverage = build_stack_coverage(history)
    assert coverage.for_stage(StackStage.ARCHITECTURE).attempts == 1
    assert coverage.for_stage(StackStage.OBJECTIVE_SAMPLING).attempts == 1
    assert coverage.for_stage(StackStage.OPTIMIZATION_REGULARIZATION).attempts == 0


def test_primary_change_remains_authoritative_with_ancillary_candidate_details():
    plan = BreadthPlan.from_json(_breadth(
        _candidate(
            "B-DEEPFM", StackStage.ARCHITECTURE,
            primary_change="Use a DeepFM architecture.",
            mechanism="Train DeepFM with AdamW, embedding dropout, and weight decay.",
            evidence=[{"citation_id": "guo2017deepfm", "claim_id": "joint-low-high-order-interactions"}],
        ),
        _candidate("B-BPR", StackStage.OBJECTIVE_SAMPLING),
        _candidate("B-FEATURE", StackStage.FEATURES),
    ), max_candidates=8)
    survivors, _ = filter_breadth_candidates(
        plan, history=[_baseline()], citation_source=JsonCitationCatalog()
    )
    assert "B-DEEPFM" in {candidate.candidate_id for candidate in survivors}


def test_renamed_deepfm_is_a_structural_history_duplicate():
    plan = BreadthPlan.from_json(_breadth(
        _candidate(
            "B-RENAMED", StackStage.ARCHITECTURE,
            title="DeepFM latent conjunction estimator",
            primary_change="Use a DeepFM latent conjunction estimator.",
            mechanism="Estimate latent conjunctions with DeepFM.",
            evidence=[{"citation_id": "guo2017deepfm", "claim_id": "joint-low-high-order-interactions"}],
        ),
        _candidate("B-FEATURE", StackStage.FEATURES),
        _candidate("B-OBJECTIVE", StackStage.OBJECTIVE_SAMPLING),
    ), max_candidates=8)
    survivors, rejections = filter_breadth_candidates(
        plan,
        history=[_baseline(), _record(1, "DeepFM interaction architecture.", Decision.REVERT)],
        citation_source=JsonCitationCatalog(),
    )
    assert "B-RENAMED" not in {candidate.candidate_id for candidate in survivors}
    assert {item.candidate_id: item.reason for item in rejections}["B-RENAMED"] == (
        "duplicates current-run history"
    )


def test_genuinely_different_feature_method_has_more_novelty_than_renamed_deepfm():
    history = [
        _baseline(),
        _record(1, "Use a DeepFM interaction architecture.", Decision.REVERT),
    ]
    renamed = BreadthCandidate.from_dict(_candidate(
        "B-RENAMED", StackStage.ARCHITECTURE,
        primary_change="Use a DeepFM latent conjunction estimator.",
        mechanism="Estimate latent conjunctions with DeepFM.",
        evidence=[{"citation_id": "guo2017deepfm", "claim_id": "joint-low-high-order-interactions"}],
    ), "candidate")
    temporal = BreadthCandidate.from_dict(_candidate(
        "B-TEMPORAL", StackStage.FEATURES,
        primary_change="Add temporal activity features.",
        mechanism="Add temporal activity and recency features.",
    ), "candidate")
    scores = {
        score.candidate.candidate_id: score.novelty
        for score in rank_breadth_candidates(
            [renamed, temporal],
            history=history,
            coverage=build_stack_coverage(history),
            context=build_research_context(history),
            citation_source=JsonCitationCatalog(),
        )
    }
    assert scores["B-RENAMED"] < 1.0
    assert scores["B-TEMPORAL"] > scores["B-RENAMED"]


def test_material_same_family_primary_variant_remains_eligible():
    plan = BreadthPlan.from_json(_breadth(
        _candidate(
            "B-BPR-HARD", StackStage.OBJECTIVE_SAMPLING,
            primary_change="Use a BPR pairwise objective with hard-negative sampling.",
            mechanism="Make hard-negative sampling the new BPR variant.",
        ),
        _candidate("B-FEATURE", StackStage.FEATURES),
        _candidate("B-ARCH", StackStage.ARCHITECTURE),
    ), max_candidates=8)
    survivors, _ = filter_breadth_candidates(
        plan,
        history=[_baseline(), _record(1, "Use a plain BPR pairwise objective.", Decision.REVERT)],
        citation_source=JsonCitationCatalog(),
    )
    assert "B-BPR-HARD" in {candidate.candidate_id for candidate in survivors}


def test_renamed_duplicate_survivor_is_stable_across_ids_and_order():
    def surviving_title(first_id, second_id, reverse):
        candidates = [
            _candidate(
                first_id, StackStage.ARCHITECTURE,
                title="DeepFM interaction architecture",
                primary_change="Use a DeepFM architecture.",
                mechanism="Use DeepFM for feature interactions.",
                evidence=[{"citation_id": "guo2017deepfm", "claim_id": "joint-low-high-order-interactions"}],
            ),
            _candidate(
                second_id, StackStage.ARCHITECTURE,
                title="DeepFM latent conjunction estimator",
                primary_change="Use a DeepFM latent conjunction estimator.",
                mechanism="Estimate latent conjunctions with DeepFM.",
                evidence=[{"citation_id": "guo2017deepfm", "claim_id": "joint-low-high-order-interactions"}],
            ),
            _candidate("B-FEATURE", StackStage.FEATURES),
        ]
        if reverse:
            candidates.reverse()
        plan = BreadthPlan.from_json(_breadth(*candidates), max_candidates=8)
        survivors, _ = filter_breadth_candidates(
            plan, history=[_baseline()], citation_source=JsonCitationCatalog()
        )
        return [candidate.title for candidate in survivors if candidate.stack_stage == StackStage.ARCHITECTURE]

    assert surviving_title("B-Z", "B-A", False) == surviving_title(
        "B-NEW-1", "B-NEW-2", True
    )


def test_two_copied_technical_tokens_do_not_create_full_citation_credit():
    copied = CitationRecord(
        citation_id="copied-terms",
        title="Unrelated empirical note",
        authors=("A",), year=2025, venue="Test", url="https://example/copied",
        claims=(CitationClaim("copied", "Embedding ranking observations are reported."),),
        tags=("embedding", "ranking"),
    )
    deepfm = JsonCitationCatalog().get("guo2017deepfm")
    assert deepfm is not None
    catalog = _Catalog([copied, deepfm])
    copied_candidate = BreadthCandidate.from_dict(_candidate(
        "B-COPIED", StackStage.ARCHITECTURE,
        primary_change="Use a DeepFM architecture.",
        mechanism="Use a DeepFM embedding ranking architecture.",
        evidence=[{"citation_id": "copied-terms", "claim_id": "copied"}],
    ), "candidate")
    relevant_candidate = BreadthCandidate.from_dict(_candidate(
        "B-RELEVANT", StackStage.ARCHITECTURE,
        primary_change="Use a DeepFM architecture.",
        mechanism="Use a DeepFM architecture.",
        evidence=[{"citation_id": "guo2017deepfm", "claim_id": "joint-low-high-order-interactions"}],
    ), "candidate")
    scores = {
        score.candidate.candidate_id: score.evidence
        for score in rank_breadth_candidates(
            [copied_candidate, relevant_candidate],
            history=[_baseline()],
            coverage=build_stack_coverage([_baseline()]),
            context=build_research_context([_baseline()]),
            citation_source=catalog,
        )
    }
    assert scores == {"B-COPIED": 0.0, "B-RELEVANT": 2.25}


def test_tight_budget_does_not_double_reward_self_reported_low_cost():
    low = BreadthCandidate.from_dict(_candidate(
        "B-LOW", StackStage.OBJECTIVE_SAMPLING, cost="low"
    ), "candidate")
    high = BreadthCandidate.from_dict(_candidate(
        "B-HIGH", StackStage.OBJECTIVE_SAMPLING, cost="high"
    ), "candidate")
    history = [_baseline()]
    scores = {
        score.candidate.candidate_id: score
        for score in rank_breadth_candidates(
            [low, high],
            history=history,
            coverage=build_stack_coverage(history),
            context=build_research_context(
                history, ConvergenceConfig(max_iterations=2, max_wall_s=3600.0)
            ),
            citation_source=JsonCitationCatalog(),
        )
    }
    assert scores["B-LOW"].budget == 0.0
    assert scores["B-HIGH"].budget == 0.0


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "Leverage leaderboard feedback for model selection.",
        "Adapt the model in light of final competition results.",
        "Inform architecture choice with leaderboard position.",
        "Use leaderboard position to guide model selection.",
    ],
)
def test_remaining_evaluation_feedback_paraphrases_are_rejected(unsafe_text):
    with pytest.raises(BreadthValidationError, match="unsafe Research direction"):
        validate_research_direction_safety({"mechanism": unsafe_text}, origin="test")


def test_preservation_is_target_local_for_mechanism_inference():
    signature = infer_mechanism_signature(
        "Keep the architecture constant and add temporal features."
    )
    assert signature.stack_stage == StackStage.FEATURES
    assert signature.primary_family == "temporal_features"


def test_depth_fails_closed_on_ambiguous_implementation_signature():
    selected = BreadthCandidate.from_dict(_candidate(
        "B-BPR", StackStage.OBJECTIVE_SAMPLING,
        primary_change="Use a BPR pairwise objective.",
        mechanism="Use BPR pairwise ordering with hard negatives.",
    ), "candidate")
    proposal = _proposal_for_method(
        title="BPR pairwise objective",
        hypothesis="Use a BPR pairwise objective with hard-negative sampling.",
        mechanism="BPR BPR BPR provides background ordering terminology.",
        components=["training objective", "hard-negative sampler"],
        steps=[
            "Implement LambdaRank listwise gradients and BPR pairwise loss with hard negatives."
        ],
    )
    with pytest.raises(BreadthValidationError, match="ambiguous|conflict|family"):
        validate_depth_alignment(proposal, selected)


def test_structured_history_falls_back_after_unknown_hypothesis():
    proposal = _proposal_for_method(
        title="DeepFM interaction architecture",
        hypothesis="Try a stronger ranking representation.",
        mechanism="DeepFM learns low- and high-order feature interactions.",
        components=["DeepFM architecture"],
        steps=["Train DeepFM with ordinary embedding dropout."],
    ).to_handoff_text()
    coverage = build_stack_coverage([
        _baseline(),
        _record(1, proposal, Decision.REVERT),
    ])
    assert coverage.for_stage(StackStage.ARCHITECTURE).attempts == 1
    assert coverage.unclassified_attempts == 0


def test_bayesian_personalized_pairwise_ranking_alias_is_a_bpr_duplicate():
    plan = BreadthPlan.from_json(_breadth(
        _candidate(
            "B-BPR-ALIAS", StackStage.OBJECTIVE_SAMPLING,
            primary_change="Use a Bayesian personalized pairwise ranking loss.",
            mechanism="Optimize a Bayesian personalized pairwise ranking loss.",
        ),
        _candidate("B-FEATURE", StackStage.FEATURES),
        _candidate("B-ARCH", StackStage.ARCHITECTURE),
    ), max_candidates=8)
    survivors, rejections = filter_breadth_candidates(
        plan,
        history=[_baseline(), _record(1, "Use a BPR pairwise objective.", Decision.REVERT)],
        citation_source=JsonCitationCatalog(),
    )
    assert "B-BPR-ALIAS" not in {candidate.candidate_id for candidate in survivors}
    assert {item.candidate_id: item.reason for item in rejections}["B-BPR-ALIAS"] == (
        "duplicates current-run history"
    )
