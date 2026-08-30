"""Internal breadth-stage models, stack coverage, filtering, and ranking.

The Orchestrator and Coding Agent never see these objects. They exist only to
help ``LLMResearchAgent`` compare several shallow directions before asking for
one full ``ResearchProposal``.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from enum import Enum
from difflib import SequenceMatcher
from typing import Any, Mapping, Optional, Sequence

from agent.config import BOOTSTRAP_ITERATION, FORBIDDEN_PAYLOAD_KEYS
from agent.records import Decision, RunRecord, Status
from agent.research.citations import CitationSource
from agent.research.context import ResearchContext
from agent.research.retrieval.safety import ResearchSafetyError, ResearchSafetyScanner
from agent.research.schemas import ProposalValidationError, ResearchProposal


class BreadthValidationError(ProposalValidationError):
    """A breadth response or selected direction is invalid."""


class StackStage(str, Enum):
    FEATURES = "features"
    ARCHITECTURE = "architecture"
    OBJECTIVE_SAMPLING = "objective_sampling"
    OPTIMIZATION_REGULARIZATION = "optimization_regularization"
    INFERENCE_ENSEMBLE = "inference_ensemble"


STACK_STAGES = tuple(StackStage)
MIN_BREADTH_CANDIDATES = 3
MIN_BREADTH_SURVIVORS = 3
MAX_BREADTH_CANDIDATES = 8
MAX_BREADTH_EVIDENCE = 3
NEAR_DUPLICATE_THRESHOLD = 0.72
MAX_CANDIDATE_ID_CHARS = 64
MAX_TITLE_CHARS = 200
MAX_PRIMARY_CHANGE_CHARS = 400
MAX_MECHANISM_CHARS = 1_200
MAX_METRIC_RATIONALE_CHARS = 800
MAX_EVIDENCE_ID_CHARS = 128

_IDENTIFIER = re.compile(rf"[A-Za-z0-9][A-Za-z0-9._:-]{{0,{MAX_CANDIDATE_ID_CHARS - 1}}}")
_LEVELS = frozenset({"low", "medium", "high"})

# Whole-token primary mechanism anchors. Secondary sampling details are kept
# separate so they cannot rescue an objective-family switch during alignment.
_PRIMARY_FAMILY_PHRASES: dict[StackStage, dict[str, tuple[tuple[str, ...], ...]]] = {
    StackStage.FEATURES: {
        "temporal_features": (("temporal", "features"), ("temporal", "activity", "features"), ("recency", "features")),
        "aggregate_statistics": (("aggregate", "statistics"), ("user", "history", "statistics"), ("activity", "statistics")),
        "interaction_features": (("interaction", "features"), ("video", "duration", "interactions")),
        "feature_crosses": (("feature", "crosses"), ("cross", "features")),
        "metadata_features": (("metadata", "features"), ("content", "features"), ("context", "features")),
    },
    StackStage.ARCHITECTURE: {
        "deepfm": (("deepfm",),),
        "dcn": (("dcn",), ("deep", "cross", "network")),
        "fm": (("factorization", "machine"), ("fm",)),
        "mlp": (("mlp",), ("mlp", "tower"), ("multilayer", "perceptron")),
        "attention": (("attention", "architecture"), ("attention", "network"), ("din",)),
        "transformer": (("transformer",),),
    },
    StackStage.OBJECTIVE_SAMPLING: {
        "pairwise": (
            ("bpr",),
            ("bayesian", "personalized", "ranking"),
            ("bayesian", "personalized", "pairwise", "ranking"),
            ("personalized", "pairwise", "ranking"),
            ("pairwise", "loss"),
            ("pairwise", "objective"),
            ("pairwise", "ordering"),
            ("pairwise", "term"),
        ),
        "listwise": (("listwise",), ("listwise", "objective"), ("listwise", "loss"), ("lambdarank",), ("listnet",)),
        "pointwise": (("pointwise",), ("pointwise", "loss"), ("logloss",), ("binary", "cross", "entropy")),
    },
    StackStage.OPTIMIZATION_REGULARIZATION: {
        "regularization": (("embedding", "regularization"), ("embedding", "dropout"), ("dropout",), ("weight", "decay"), ("label", "smoothing")),
        "optimizer_change": (("adamw",), ("optimizer", "change")),
        "lr_schedule": (("learning", "rate", "schedule"), ("learning", "rate", "scheduler"), ("lr", "schedule")),
        "gradient_clipping": (("gradient", "clipping"), ("clip", "gradients")),
        "early_stopping": (("early", "stopping"),),
    },
    StackStage.INFERENCE_ENSEMBLE: {
        "prediction_averaging": (("prediction", "averaging"), ("rank", "averaging")),
        "checkpoint_ensemble": (("checkpoint", "ensemble"),),
        "blending": (("score", "blending"), ("blend", "scores"), ("blending",), ("ensemble",)),
        "calibration": (("inference", "time", "calibration"), ("score", "calibration")),
    },
}

_SECONDARY_TAG_PHRASES: dict[str, tuple[tuple[str, ...], ...]] = {
    "hard_negative_sampling": (("hard", "negative", "sampling"), ("hard", "negatives")),
    "within_user_sampling": (("within", "user", "sampling"), ("same", "user", "pairs")),
    "dropout": (("dropout",), ("embedding", "dropout")),
    "weight_decay": (("weight", "decay"),),
    "candidate_conditioning": (("candidate", "conditioned"), ("candidate", "conditioning")),
    "adamw": (("adamw",),),
    "gradient_clipping": (("gradient", "clipping"), ("clip", "gradients")),
    "lr_schedule": (("learning", "rate", "schedule"), ("learning", "rate", "scheduler"), ("lr", "schedule")),
}
_ANCILLARY_TRAINING_TAGS = frozenset({
    "adamw", "dropout", "gradient_clipping", "lr_schedule", "weight_decay",
})

_PRESERVATION_TOKENS = frozenset({
    "constant", "keep", "keeping", "kept", "preserve", "preserved", "preserving",
    "retain", "retained", "retaining", "unchanged",
})
_CONTEXTUAL_OLD_TOKENS = frozenset({"baseline", "current", "existing", "incumbent"})
_METHOD_ANCHOR_TOKENS = frozenset({
    "attention", "bpr", "dcn", "deepfm", "din", "fm", "lambdarank", "listnet",
    "listwise", "pairwise", "pointwise", "transformer",
})
_ALIGNMENT_STOPWORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "if", "in", "into", "is", "it",
    "of", "on", "or", "so", "that", "the", "their", "this", "to", "using", "with", "without",
    "add", "change", "improve", "model", "ranking", "validation", "training", "user",
    "video", "feature", "features", "metric", "data",
})

_SIMILARITY_STOPWORDS = _ALIGNMENT_STOPWORDS | {
    "candidate", "experiment", "expected", "gauc", "ndcg", "primary", "score",
}

_NEGATIONS = frozenset({"avoid", "avoids", "avoiding", "never", "no", "not", "without"})
_DATA_ACTIONS = frozenset({
    "adapt", "adjust", "base", "calibrate", "choose", "condition", "decide", "exploit",
    "fit", "guide", "inform", "leverage", "optimize", "pick", "rank", "read", "refine",
    "select", "set", "train", "tune", "use",
})
_SCORER_ACTIONS = frozenset({
    "adjust", "alter", "calibrate", "change", "modify", "optimize", "patch",
    "replace", "rewrite", "tune",
})
_ACTION_CANONICAL = {
    "adjusted": "adjust", "adjusting": "adjust", "calibrated": "calibrate", "calibrating": "calibrate",
    "choosing": "choose", "chosen": "choose", "exploiting": "exploit", "fitted": "fit", "fitting": "fit",
    "optimized": "optimize", "optimise": "optimize", "optimised": "optimize", "optimising": "optimize",
    "optimizing": "optimize", "reading": "read", "selected": "select", "selecting": "select",
    "trained": "train", "training": "train", "tuned": "tune", "tuning": "tune", "used": "use", "using": "use",
    "altered": "alter", "altering": "alter", "changed": "change", "changing": "change", "modified": "modify",
    "modifying": "modify", "patched": "patch", "patching": "patch", "replaced": "replace", "replacing": "replace",
    "rewritten": "rewrite", "rewriting": "rewrite",
    "decided": "decide", "deciding": "decide", "guided": "guide", "guiding": "guide",
    "picked": "pick", "picking": "pick", "ranked": "rank", "ranking": "rank",
    "refined": "refine", "refining": "refine", "sets": "set", "setting": "set",
    "adjusts": "adjust", "calibrates": "calibrate", "chooses": "choose", "decides": "decide",
    "guides": "guide", "optimizes": "optimize", "picks": "pick", "refines": "refine",
    "selects": "select", "trains": "train", "tunes": "tune", "uses": "use",
    "alters": "alter", "changes": "change", "modifies": "modify", "patches": "patch",
    "replaces": "replace", "rewrites": "rewrite",
    "adapted": "adapt", "adapting": "adapt", "adapts": "adapt",
    "based": "base", "bases": "base", "basing": "base",
    "conditioned": "condition", "conditioning": "condition", "conditions": "condition",
    "informed": "inform", "informing": "inform", "informs": "inform",
    "leveraged": "leverage", "leverages": "leverage", "leveraging": "leverage",
}
_PROPOSAL_SAFETY_SCANNER = ResearchSafetyScanner(max_text_chars=32_000, max_total_chars=64_000)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BreadthValidationError(f"{path} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise BreadthValidationError(f"{path}: " + "; ".join(details))


def _text(value: Any, path: str, *, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BreadthValidationError(f"{path} must be a non-empty string")
    result = value.strip()
    if len(result) > max_chars:
        raise BreadthValidationError(f"{path} exceeds {max_chars} characters")
    return result


def _level(value: Any, path: str) -> str:
    level = _text(value, path, max_chars=16).casefold()
    if level not in _LEVELS:
        raise BreadthValidationError(f"{path} must be low, medium, or high")
    return level


def _json_safe(value: Any, path: str = "breadth") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise BreadthValidationError(f"{path} contains a non-string key")
            if key.casefold() in FORBIDDEN_PAYLOAD_KEYS:
                raise BreadthValidationError(f"{path} contains forbidden key {key!r}")
            _json_safe(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _json_safe(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise BreadthValidationError(f"{path} must be finite")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise BreadthValidationError(f"{path} is not JSON-serialisable")


@dataclass(frozen=True)
class BreadthEvidence:
    citation_id: str
    claim_id: str

    @classmethod
    def from_dict(cls, value: Any, path: str) -> "BreadthEvidence":
        data = _mapping(value, path)
        _exact_keys(data, {"citation_id", "claim_id"}, path)
        return cls(
            citation_id=_text(
                data["citation_id"], f"{path}.citation_id", max_chars=MAX_EVIDENCE_ID_CHARS
            ),
            claim_id=_text(
                data["claim_id"], f"{path}.claim_id", max_chars=MAX_EVIDENCE_ID_CHARS
            ),
        )


@dataclass(frozen=True)
class BreadthCandidate:
    candidate_id: str
    title: str
    stack_stage: StackStage
    primary_change: str
    mechanism: str
    metric_rationale: str
    expected_upside: str
    implementation_risk: str
    experiment_cost: str
    evidence: tuple[BreadthEvidence, ...]

    @classmethod
    def from_dict(cls, value: Any, path: str) -> "BreadthCandidate":
        data = _mapping(value, path)
        _exact_keys(data, {
            "candidate_id", "title", "stack_stage", "primary_change", "mechanism", "metric_rationale",
            "expected_upside", "implementation_risk", "experiment_cost", "evidence",
        }, path)
        candidate_id = _text(
            data["candidate_id"],
            f"{path}.candidate_id",
            max_chars=MAX_CANDIDATE_ID_CHARS,
        )
        if _IDENTIFIER.fullmatch(candidate_id) is None:
            raise BreadthValidationError(f"{path}.candidate_id is not a valid identifier")
        try:
            stack_stage = StackStage(
                _text(data["stack_stage"], f"{path}.stack_stage", max_chars=64)
            )
        except ValueError as exc:
            raise BreadthValidationError(
                f"{path}.stack_stage must be one of {[stage.value for stage in STACK_STAGES]}"
            ) from exc
        evidence_raw = data["evidence"]
        if not isinstance(evidence_raw, Sequence) or isinstance(evidence_raw, (str, bytes)):
            raise BreadthValidationError(f"{path}.evidence must be an array")
        evidence = tuple(
            BreadthEvidence.from_dict(item, f"{path}.evidence[{index}]")
            for index, item in enumerate(evidence_raw)
        )
        if not evidence:
            raise BreadthValidationError(f"{path}.evidence must contain at least one citation")
        if len(evidence) > MAX_BREADTH_EVIDENCE:
            raise BreadthValidationError(
                f"{path}.evidence exceeds the maximum of {MAX_BREADTH_EVIDENCE} references"
            )
        evidence_keys = {
            (item.citation_id.casefold(), item.claim_id.casefold()) for item in evidence
        }
        if len(evidence_keys) != len(evidence):
            raise BreadthValidationError(f"{path}.evidence contains duplicates")
        return cls(
            candidate_id=candidate_id,
            title=_text(data["title"], f"{path}.title", max_chars=MAX_TITLE_CHARS),
            stack_stage=stack_stage,
            primary_change=_text(
                data["primary_change"],
                f"{path}.primary_change",
                max_chars=MAX_PRIMARY_CHANGE_CHARS,
            ),
            mechanism=_text(
                data["mechanism"], f"{path}.mechanism", max_chars=MAX_MECHANISM_CHARS
            ),
            metric_rationale=_text(
                data["metric_rationale"],
                f"{path}.metric_rationale",
                max_chars=MAX_METRIC_RATIONALE_CHARS,
            ),
            expected_upside=_level(data["expected_upside"], f"{path}.expected_upside"),
            implementation_risk=_level(data["implementation_risk"], f"{path}.implementation_risk"),
            experiment_cost=_level(data["experiment_cost"], f"{path}.experiment_cost"),
            evidence=evidence,
        )

    def selection_text(self) -> str:
        return (
            f"{self.title}\n{self.primary_change}\n"
            f"{self.mechanism}\n{self.metric_rationale}"
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["stack_stage"] = self.stack_stage.value
        return result


@dataclass(frozen=True)
class BreadthRejection:
    candidate_id: str
    reason: str


@dataclass(frozen=True)
class BreadthPlan:
    schema_version: int
    candidates: tuple[BreadthCandidate, ...]

    @classmethod
    def from_json(cls, text: str, *, max_candidates: int) -> "BreadthPlan":
        return cls.from_json_with_bounds(
            text,
            min_candidates=MIN_BREADTH_CANDIDATES,
            max_candidates=max_candidates,
        )

    @classmethod
    def from_json_with_bounds(
        cls,
        text: str,
        *,
        min_candidates: int,
        max_candidates: int,
        exact_candidates: Optional[int] = None,
    ) -> "BreadthPlan":
        """Parse a normal batch or a smaller incremental replacement batch."""
        if not isinstance(max_candidates, int) or isinstance(max_candidates, bool):
            raise BreadthValidationError("max_candidates must be an integer")
        if not isinstance(min_candidates, int) or isinstance(min_candidates, bool):
            raise BreadthValidationError("min_candidates must be an integer")
        if min_candidates < 1:
            raise BreadthValidationError("min_candidates must be positive")
        if max_candidates < min_candidates:
            raise BreadthValidationError(
                "max_candidates must be at least min_candidates"
            )
        effective_maximum = min(max_candidates, MAX_BREADTH_CANDIDATES)
        if exact_candidates is not None:
            if (
                not isinstance(exact_candidates, int)
                or isinstance(exact_candidates, bool)
                or not min_candidates <= exact_candidates <= effective_maximum
            ):
                raise BreadthValidationError(
                    "exact_candidates must be within the configured candidate bounds"
                )
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BreadthValidationError(f"breadth response is not valid JSON: {exc}") from exc
        data = _mapping(value, "breadth")
        _json_safe(data)
        _exact_keys(data, {"schema_version", "candidates"}, "breadth")
        if (
            not isinstance(data["schema_version"], int)
            or isinstance(data["schema_version"], bool)
            or data["schema_version"] != 1
        ):
            raise BreadthValidationError("breadth.schema_version must be 1")
        raw_candidates = data["candidates"]
        if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
            raise BreadthValidationError("breadth.candidates must be an array")
        if len(raw_candidates) < min_candidates:
            raise BreadthValidationError(
                "breadth.candidates must contain at least "
                f"{min_candidates} competing direction(s)"
            )
        if len(raw_candidates) > effective_maximum:
            raise BreadthValidationError(
                "breadth.candidates exceeds maximum "
                f"{effective_maximum} (absolute maximum {MAX_BREADTH_CANDIDATES})"
            )
        if exact_candidates is not None and len(raw_candidates) != exact_candidates:
            raise BreadthValidationError(
                "breadth.candidates must contain exactly "
                f"{exact_candidates} candidate(s); got {len(raw_candidates)}"
            )
        candidates = tuple(
            BreadthCandidate.from_dict(item, f"breadth.candidates[{index}]")
            for index, item in enumerate(raw_candidates)
        )
        identifiers = [candidate.candidate_id.casefold() for candidate in candidates]
        if len(identifiers) != len(set(identifiers)):
            raise BreadthValidationError("breadth.candidates contains duplicate candidate IDs")
        return cls(schema_version=1, candidates=candidates)


@dataclass(frozen=True)
class BreadthParseResult:
    """Valid candidates plus candidate-local failures from one JSON envelope."""

    plan: BreadthPlan
    rejections: tuple[BreadthRejection, ...]
    reserved_candidate_ids: frozenset[str]
    raw_candidate_count: int


def _usable_raw_candidate_id(value: Any) -> Optional[str]:
    if not isinstance(value, Mapping):
        return None
    candidate_id = value.get("candidate_id")
    if not isinstance(candidate_id, str):
        return None
    candidate_id = candidate_id.strip()
    if (
        not candidate_id
        or len(candidate_id) > MAX_CANDIDATE_ID_CHARS
        or _IDENTIFIER.fullmatch(candidate_id) is None
    ):
        return None
    return candidate_id


def parse_breadth_candidates_individually(
    text: str,
    *,
    min_candidates: int,
    max_candidates: int,
) -> BreadthParseResult:
    """Parse a valid breadth envelope without discarding valid siblings.

    Envelope shape, version, and raw candidate-count bounds remain atomic. Each
    array item then receives the unchanged strict ``BreadthCandidate`` schema.
    Usable raw IDs are reserved before detailed validation so a malformed item
    cannot free its identity for a replacement.
    """
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool):
        raise BreadthValidationError("max_candidates must be an integer")
    if not isinstance(min_candidates, int) or isinstance(min_candidates, bool):
        raise BreadthValidationError("min_candidates must be an integer")
    if min_candidates < 1:
        raise BreadthValidationError("min_candidates must be positive")
    if max_candidates < min_candidates:
        raise BreadthValidationError("max_candidates must be at least min_candidates")
    effective_maximum = min(max_candidates, MAX_BREADTH_CANDIDATES)

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BreadthValidationError(f"breadth response is not valid JSON: {exc}") from exc
    data = _mapping(value, "breadth")
    _exact_keys(data, {"schema_version", "candidates"}, "breadth")
    _json_safe(data["schema_version"], "breadth.schema_version")
    if (
        not isinstance(data["schema_version"], int)
        or isinstance(data["schema_version"], bool)
        or data["schema_version"] != 1
    ):
        raise BreadthValidationError("breadth.schema_version must be 1")
    raw_candidates = data["candidates"]
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        raise BreadthValidationError("breadth.candidates must be an array")
    if len(raw_candidates) < min_candidates:
        raise BreadthValidationError(
            "breadth.candidates must contain at least "
            f"{min_candidates} competing direction(s)"
        )
    if len(raw_candidates) > effective_maximum:
        raise BreadthValidationError(
            "breadth.candidates exceeds maximum "
            f"{effective_maximum} (absolute maximum {MAX_BREADTH_CANDIDATES})"
        )

    raw_ids = tuple(_usable_raw_candidate_id(item) for item in raw_candidates)
    reserved_ids = frozenset(
        candidate_id.casefold() for candidate_id in raw_ids if candidate_id is not None
    )
    id_counts: dict[str, int] = {}
    for candidate_id in raw_ids:
        if candidate_id is not None:
            key = candidate_id.casefold()
            id_counts[key] = id_counts.get(key, 0) + 1

    candidates: list[BreadthCandidate] = []
    rejections: list[BreadthRejection] = []
    for index, item in enumerate(raw_candidates):
        path = f"breadth.candidates[{index}]"
        diagnostic_id = raw_ids[index] or f"candidate[{index}]"
        try:
            _json_safe(item, path)
            candidate = BreadthCandidate.from_dict(item, path)
            if id_counts.get(candidate.candidate_id.casefold(), 0) > 1:
                raise BreadthValidationError(
                    f"{path}.candidate_id duplicates another original candidate ID"
                )
            candidates.append(candidate)
        except BreadthValidationError as exc:
            rejections.append(BreadthRejection(diagnostic_id, str(exc)))

    return BreadthParseResult(
        plan=BreadthPlan(schema_version=1, candidates=tuple(candidates)),
        rejections=tuple(rejections),
        reserved_candidate_ids=reserved_ids,
        raw_candidate_count=len(raw_candidates),
    )


@dataclass(frozen=True)
class StageCoverage:
    attempts: int = 0
    accepted: int = 0
    reverted: int = 0
    failed: int = 0
    most_recent_iteration: Optional[int] = None


@dataclass(frozen=True)
class StackCoverageSummary:
    stages: dict[StackStage, StageCoverage]
    unclassified_attempts: int = 0

    def for_stage(self, stage: StackStage) -> StageCoverage:
        return self.stages[stage]

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            stage.value: asdict(self.stages[stage])
            for stage in STACK_STAGES
        } | {"unclassified_attempts": self.unclassified_attempts}


def normalize_research_text(text: str) -> str:
    """Canonical text normalization shared with full-proposal duplicate checks."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"evaluate\s*\.\s*py", "evaluate_py", normalized)
    return " ".join(re.findall(r"[a-z0-9_]+", normalized))


def _section(text: str, heading: str) -> Optional[str]:
    match = re.search(
        rf"(?ms)^{re.escape(heading)}:\s*\n(.*?)(?=\n[A-Z][A-Z ]+:\s*\n|\Z)",
        text,
    )
    return match.group(1).strip() if match else None


def _method_relevant_text(text: str) -> str:
    """Extract intervention-bearing fields from a structured Research handoff.

    Evidence, risks, constants, hyperparameters, metric criteria, and ablation
    prose are intentionally excluded.  They may mention technical methods as
    context, but they do not state what the experiment primarily changed.
    """
    if "[RESEARCH_PROPOSAL" not in text:
        return text
    title = re.search(r"(?m)^TITLE:\s*([^\r\n]+?)\s*$", text)
    sections = [title.group(1) if title else ""]
    sections.extend(
        section
        for heading in (
            "HYPOTHESIS",
            "WHY THIS SHOULD HELP",
            "IMPLEMENTATION",
            "TARGET COMPONENTS",
        )
        if (section := _section(text, heading))
    )
    return "\n".join(sections)


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_research_text(_method_relevant_text(text)).split()
        if len(token) >= 3 and token not in _SIMILARITY_STOPWORDS
    }


def research_text_similarity(left: str, right: str) -> float:
    """Mechanism-level lexical similarity robust to renamed structured handoffs."""
    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    containment = len(intersection) / min(len(left_tokens), len(right_tokens))
    union = left_tokens | right_tokens
    jaccard = len(intersection) / len(union)
    left_normalized = " ".join(sorted(left_tokens))
    right_normalized = " ".join(sorted(right_tokens))
    sequence = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    token_similarity = 0.75 * containment + 0.25 * jaccard
    return max(sequence, token_similarity)


def historical_hypothesis(record: RunRecord) -> str:
    return _method_relevant_text(record.hypothesis)


def historical_mechanism_signature(record: RunRecord) -> MechanismSignature:
    """Infer history from intervention-bearing fields, not handoff boilerplate."""
    text = record.hypothesis
    if text.lstrip().startswith("[RESEARCH_PROPOSAL"):
        hypothesis = _section(text, "HYPOTHESIS")
        title = re.search(r"(?m)^TITLE:\s*([^\r\n]+?)\s*$", text)
        implementation = "\n".join(filter(None, (
            _section(text, "IMPLEMENTATION"),
            _section(text, "TARGET COMPONENTS"),
        )))
        fields = (
            hypothesis or "",
            title.group(1) if title else "",
            implementation,
        )
        signatures = [infer_mechanism_signature(field) for field in fields if field.strip()]
        confident = [item for item in signatures if item.confidence == "confident"]
        if not confident:
            secondary = tuple(sorted({
                tag for item in signatures for tag in item.secondary_tags
            }))
            return MechanismSignature(
                None,
                None,
                (),
                secondary,
                "ambiguous" if any(item.ambiguous for item in signatures) else "unknown",
                any(item.ambiguous for item in signatures),
            )

        chosen = confident[0]
        for item in confident[1:]:
            if (
                item.stack_stage != chosen.stack_stage
                or not _compatible_primary_families(
                    chosen.stack_stage,
                    item.primary_family,
                    chosen.primary_family,
                )
            ):
                return MechanismSignature(
                    None,
                    None,
                    (),
                    tuple(sorted({
                        tag for signature in signatures
                        for tag in signature.secondary_tags
                    })),
                    "ambiguous",
                    True,
                )
        return MechanismSignature(
            chosen.stack_stage,
            chosen.primary_family,
            tuple(sorted({
                tag for item in confident for tag in item.primary_tags
            })),
            tuple(sorted({
                tag for item in signatures for tag in item.secondary_tags
            })),
            "confident",
            False,
        )
    return infer_mechanism_signature(text)


def was_attempted(record: RunRecord) -> bool:
    return record.decision in {Decision.ACCEPT, Decision.REVERT, Decision.ABANDON} or (
        record.status == Status.ABANDONED
    )


def _contains_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    width = len(phrase)
    return any(tuple(tokens[index:index + width]) == tuple(phrase) for index in range(len(tokens) - width + 1))


def _phrase_positions(tokens: Sequence[str], phrase: Sequence[str]) -> tuple[int, ...]:
    width = len(phrase)
    return tuple(
        index
        for index in range(len(tokens) - width + 1)
        if tuple(tokens[index:index + width]) == tuple(phrase)
    )


_CLAUSE_BOUNDARY = re.compile(
    r"(?:[\r\n.;!?]+|,\s*(?=(?:then|but|however|yet|while|although)\b)|"
    r"\b(?:then|but|however|yet|while|although)\b|"
    r"\band\s+(?=(?:add|adapt|adjust|apply|calibrate|change|choose|introduce|"
    r"modify|optimize|refine|replace|select|switch|train|tune|use)\b))",
    re.IGNORECASE,
)


def _clause_texts(text: str) -> tuple[str, ...]:
    protected = re.sub(r"evaluate\s*\.\s*py", "evaluate_py", text, flags=re.IGNORECASE)
    return tuple(part.strip(" ,") for part in _CLAUSE_BOUNDARY.split(protected) if part.strip(" ,"))


def _is_preserved(tokens: Sequence[str], index: int) -> bool:
    """Return whether an anchor is contextual/control text in this clause only."""
    prefix = tokens[max(0, index - 7):index]
    if any(token in _PRESERVATION_TOKENS or token in _NEGATIONS for token in prefix):
        return True
    if (
        ("rather" in prefix and "than" in prefix)
        or ("instead" in prefix and "of" in prefix)
    ):
        return True
    context_positions = [
        position for position, token in enumerate(tokens[:index])
        if token in _CONTEXTUAL_OLD_TOKENS
    ]
    if context_positions:
        context_position = context_positions[-1]
        if any(token in _METHOD_ANCHOR_TOKENS for token in tokens[:context_position]):
            return True

    # In "replace X with Y" and "switch from X to Y", X describes the old
    # mechanism.  It must not conflict with the new primary family Y.
    before = tokens[:index]
    after = tokens[index + 1:]
    transition = {"change", "move", "replace", "switch"}
    if "from" in prefix and "to" in after:
        return True
    if any(token in transition for token in before[-5:]) and any(
        token in {"to", "with"} for token in after[:5]
    ):
        return True
    return False


@dataclass(frozen=True)
class MechanismSignature:
    stack_stage: Optional[StackStage]
    primary_family: Optional[str]
    primary_tags: tuple[str, ...]
    secondary_tags: tuple[str, ...]
    confidence: str
    ambiguous: bool


@dataclass(frozen=True)
class StructuralFingerprint:
    """Canonical intervention identity, independent of IDs and prose wording."""

    stack_stage: Optional[StackStage]
    primary_family: Optional[str]
    primary_tags: tuple[str, ...]
    secondary_tags: tuple[str, ...]


def _matched_primary_families(text: str) -> dict[StackStage, set[str]]:
    matched: dict[StackStage, set[str]] = {stage: set() for stage in STACK_STAGES}
    for clause in _clause_texts(_method_relevant_text(text)):
        tokens = normalize_research_text(clause).split()
        for stage, families in _PRIMARY_FAMILY_PHRASES.items():
            for family, phrases in families.items():
                if any(
                    not _is_preserved(tokens, position)
                    for phrase in phrases
                    for position in _phrase_positions(tokens, phrase)
                ):
                    matched[stage].add(family)
    # DeepFM explicitly contains an FM component; that compositional detail is
    # not an incompatible second primary architecture.
    architecture = matched[StackStage.ARCHITECTURE]
    if "deepfm" in architecture:
        architecture.discard("fm")
    inference = matched[StackStage.INFERENCE_ENSEMBLE]
    if "checkpoint_ensemble" in inference:
        inference.discard("prediction_averaging")
        inference.discard("blending")
    return {stage: families for stage, families in matched.items() if families}


def _tags_in_text(text: str) -> tuple[str, ...]:
    tokens = normalize_research_text(_method_relevant_text(text)).split()
    return tuple(sorted(
        tag
        for tag, phrases in _SECONDARY_TAG_PHRASES.items()
        if any(_contains_phrase(tokens, phrase) for phrase in phrases)
    ))


def infer_mechanism_signature(
    text: str,
    *,
    secondary_text: str = "",
) -> MechanismSignature:
    """Infer one primary intervention using whole-token technical anchors.

    Primary anchors are presence-based: repetition never votes one conflicting
    family above another.  Optimizer/regularizer details are ancillary when a
    feature, architecture, objective, or inference intervention is clear, but
    remain primary when optimization/regularization is itself the intervention.
    ``secondary_text`` contributes tags only and cannot override ``text``.
    """
    family_matches = _matched_primary_families(text)
    primary_tags = _tags_in_text(text)
    secondary_tags = tuple(sorted(set(primary_tags) | set(_tags_in_text(secondary_text))))

    non_optimization = {
        stage: families
        for stage, families in family_matches.items()
        if stage != StackStage.OPTIMIZATION_REGULARIZATION
    }
    # AdamW/dropout/weight decay commonly describe how another intervention is
    # trained.  They do not compete with a clear core intervention.
    active = non_optimization or family_matches
    if not active:
        return MechanismSignature(None, None, primary_tags, secondary_tags, "unknown", False)
    if len(active) != 1:
        return MechanismSignature(None, None, primary_tags, secondary_tags, "ambiguous", True)

    stage, families = next(iter(active.items()))
    if len(families) != 1:
        return MechanismSignature(stage, None, primary_tags, secondary_tags, "ambiguous", True)
    family = next(iter(families))
    if stage != StackStage.OPTIMIZATION_REGULARIZATION:
        primary_tags = tuple(
            tag for tag in primary_tags if tag not in _ANCILLARY_TRAINING_TAGS
        )
    return MechanismSignature(stage, family, primary_tags, secondary_tags, "confident", False)


def structural_fingerprint(signature: MechanismSignature) -> StructuralFingerprint:
    return StructuralFingerprint(
        stack_stage=signature.stack_stage,
        primary_family=signature.primary_family,
        primary_tags=signature.primary_tags,
        secondary_tags=signature.secondary_tags,
    )


def classify_stack_stage(text: str) -> Optional[StackStage]:
    """Return only a confident, single-stage mechanism classification."""
    return infer_mechanism_signature(text).stack_stage


def _is_negated(tokens: Sequence[str], action_index: int) -> bool:
    window = list(tokens[max(0, action_index - 5):action_index])
    boundaries = {"although", "but", "however", "while", "yet"}
    boundary = max(
        (index for index, token in enumerate(window) if token in boundaries),
        default=-1,
    )
    return any(token in _NEGATIONS for token in window[boundary + 1:])


def _policy_tokens(text: str) -> list[str]:
    tokens = normalize_research_text(text).split()
    canonical = []
    for token in tokens:
        token = _ACTION_CANONICAL.get(token, token)
        token = {
            "labels": "label", "outcomes": "outcome", "result": "outcome", "results": "outcome",
            "performance": "outcome", "score": "outcome", "scores": "outcome", "metrics": "outcome",
            "rankings": "ranking", "ranks": "rank",
            "parameters": "parameter", "hyperparameters": "parameter",
        }.get(token, token)
        canonical.append(token)
    return canonical


def _near(tokens: Sequence[str], index: int, choices: set[str], radius: int = 4) -> bool:
    return any(
        token in choices
        for token in tokens[max(0, index - radius):index + radius + 1]
    )


def _data_target_positions(tokens: Sequence[str]) -> set[int]:
    positions: set[int] = set()
    for index, token in enumerate(tokens):
        nearby = tokens[index:index + 5]
        if token == "test" and any(item in {"label", "outcome"} for item in nearby[1:]):
            positions.add(index)
        if token == "hidden" and any(item in {"label", "test", "evaluation"} for item in nearby[1:]):
            positions.add(index)
        if token == "final" and (
            "holdout" in nearby[1:]
            or ("hold" in nearby[1:] and "out" in nearby[1:])
            or ("evaluation" in nearby[1:] and "outcome" in tokens[index:index + 7])
        ):
            positions.add(index)
        if token == "leaderboard":
            positions.add(index)
        if token in {"challenge", "competition"} and _near(
            tokens, index, {"feedback", "outcome", "rank", "ranking"}, 6
        ):
            positions.add(index)
        if token == "test" and _near(tokens, index, {"outcome"}, 4):
            positions.add(index)
        if token == "final" and _near(
            tokens, index, {"competition", "evaluation", "holdout", "outcome"}, 6
        ) and _near(tokens, index, {"outcome", "rank", "ranking"}, 6):
            positions.add(index)
        if token == "feedback" and (
            _near(tokens, index, {"challenge", "competition", "leaderboard"}, 5)
            or (_near(tokens, index, {"hidden"}, 4) and _near(tokens, index, {"evaluation"}, 4))
        ):
            positions.add(index)
    return positions


def _scorer_target_positions(tokens: Sequence[str]) -> set[int]:
    positions: set[int] = set()
    for index, token in enumerate(tokens):
        suffix = tokens[index + 1:index + 6]
        preservation_bridges = {
            "be", "code", "implementation", "is", "must", "remain", "remains",
            "script", "stay", "stays", "weighting",
        }
        preserved = any(
            item in _PRESERVATION_TOKENS
            and all(prefix in preservation_bridges for prefix in suffix[:offset])
            for offset, item in enumerate(suffix)
        )
        if preserved:
            continue
        if token in {"scorer", "evaluator", "evaluate_py"}:
            positions.add(index)
        if token in {"script", "implementation", "code"} and _near(
            tokens,
            index,
            {
                "benchmark", "evaluate_py", "evaluation", "evaluator", "metric",
                "official", "scoring",
            },
            4,
        ):
            positions.add(index)
        if token == "metric" and _near(tokens, index, {"official", "benchmark"}, 3):
            positions.add(index)
    return positions


_MODEL_SIDE_TARGETS = frozenset({
    "architecture", "deepfm", "dropout", "embedding", "feature", "loss", "model",
    "objective", "optimizer", "parameter", "regularization", "sampler", "training",
})


def _action_targets(
    tokens: Sequence[str],
    actions: frozenset[str],
    targets: set[int],
    *,
    window: int = 12,
) -> bool:
    if not targets:
        return False
    for index, token in enumerate(tokens):
        if token not in actions or _is_negated(tokens, index):
            continue
        if any(abs(index - target) <= window for target in targets):
            return True
    return False


def _scorer_action_targets(tokens: Sequence[str]) -> bool:
    """Match scorer actions without treating a model-side object as the scorer.

    A phrase such as "tune model regularization to improve the official metric"
    contains scorer vocabulary but directs ``tune`` at regularization.  A
    nearer explicit model target protects only that action-target pair.  It
    does not protect coordinated targets ("model and official metric") or a
    later independent action ("tune model, then modify evaluate.py").
    """
    targets = _scorer_target_positions(tokens)
    for action_index, token in enumerate(tokens):
        if token not in _SCORER_ACTIONS or _is_negated(tokens, action_index):
            continue
        for target in sorted(targets, key=lambda item: abs(item - action_index)):
            if abs(action_index - target) > 12:
                continue
            lower, upper = sorted((action_index, target))
            between = tokens[lower + 1:upper]
            nearby_model_positions = [
                index for index, item in enumerate(tokens)
                if item in _MODEL_SIDE_TARGETS and abs(index - action_index) <= 12
            ]
            scorer_implementation_is_explicit = any(
                item in {"code", "implementation", "script", "scoring", "weighting"}
                for item in tokens[max(0, target - 3):target + 4]
            )
            model_before_scorer = [
                index for index in nearby_model_positions
                if action_index < index < target
            ]
            direct_model_before_scorer = (
                bool(model_before_scorer)
                and min(model_before_scorer) - action_index <= 4
                and "and" not in between
            )
            purpose_marker = action_index < target and any(
                item in {"for", "toward", "towards"}
                for item in tokens[action_index + 1:target]
            )
            model_after_purpose = [
                index for index in nearby_model_positions if target < index
            ]
            reordered_model_object = (
                purpose_marker
                and bool(model_after_purpose)
                and min(model_after_purpose) - target <= 6
                and "rather" not in between
                and "than" not in between
            )
            model_before_action = [
                index for index in nearby_model_positions if index < action_index
            ]
            trailing_metric_purpose = (
                bool(model_before_action)
                and action_index - max(model_before_action) <= 4
                and purpose_marker
            )
            model_target_is_direct = (
                direct_model_before_scorer
                or reordered_model_object
                or trailing_metric_purpose
            )
            if scorer_implementation_is_explicit or not model_target_is_direct:
                return True
    return False


def _semantic_policy_violation(text: str) -> Optional[str]:
    """Match normalized ACTION + TARGET concepts in a bounded local window."""
    for clause in _clause_texts(text):
        tokens = _policy_tokens(clause)
        if not tokens:
            continue
        if _action_targets(tokens, _DATA_ACTIONS, _data_target_positions(tokens)):
            return "uses forbidden evaluation data or feedback for development"
        if _scorer_action_targets(tokens):
            return "modifies or tunes the official scorer/metric implementation"
    return None


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result: list[str] = []
        for child in value.values():
            result.extend(_string_values(child))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = []
        for child in value:
            result.extend(_string_values(child))
        return result
    return []


def validate_research_direction_safety(value: Any, *, origin: str) -> None:
    """One reusable boundary for generated breadth and depth content."""
    try:
        _PROPOSAL_SAFETY_SCANNER.scan_value(value, origin=origin)
    except ResearchSafetyError as exc:
        raise BreadthValidationError(f"unsafe Research direction: {exc}") from exc
    strings = _string_values(value)
    # Scan fields independently and once as a flattened payload so an unsafe
    # phrase cannot be evaded by splitting its action and target across fields.
    violation = next(
        (
            found
            for text in (*strings, " ".join(strings))
            if (found := _semantic_policy_violation(text)) is not None
        ),
        None,
    )
    if violation is not None:
        raise BreadthValidationError(f"unsafe Research direction: {violation}")


def build_stack_coverage(history: Sequence[RunRecord]) -> StackCoverageSummary:
    mutable = {
        stage: {"attempts": 0, "accepted": 0, "reverted": 0, "failed": 0, "recent": None}
        for stage in STACK_STAGES
    }
    unclassified = 0
    for record in history:
        if record.iteration == BOOTSTRAP_ITERATION or not was_attempted(record):
            continue
        stage = historical_mechanism_signature(record).stack_stage
        if stage is None:
            unclassified += 1
            continue
        item = mutable[stage]
        item["attempts"] += 1
        item["recent"] = max(item["recent"] or record.iteration, record.iteration)
        if record.decision == Decision.ACCEPT:
            item["accepted"] += 1
        elif record.decision == Decision.REVERT:
            item["reverted"] += 1
        else:
            item["failed"] += 1
    return StackCoverageSummary(
        stages={
            stage: StageCoverage(
                attempts=item["attempts"],
                accepted=item["accepted"],
                reverted=item["reverted"],
                failed=item["failed"],
                most_recent_iteration=item["recent"],
            )
            for stage, item in mutable.items()
        },
        unclassified_attempts=unclassified,
    )


def _candidate_has_valid_evidence(candidate: BreadthCandidate, source: CitationSource) -> bool:
    for evidence in candidate.evidence:
        record = source.get(evidence.citation_id)
        if record is None or record.claim(evidence.claim_id) is None:
            return False
    return True


def _candidate_safety_error(candidate: BreadthCandidate) -> Optional[str]:
    try:
        validate_research_direction_safety(
            candidate.to_prompt_dict(),
            origin=f"breadth candidate {candidate.candidate_id}",
        )
    except BreadthValidationError as exc:
        return str(exc)
    return None


def _candidate_signature(candidate: BreadthCandidate) -> MechanismSignature:
    # ``primary_change`` is the authoritative intervention.  The verbose body
    # can enrich secondary tags but cannot vote for a different primary family.
    return infer_mechanism_signature(
        candidate.primary_change,
        secondary_text=f"{candidate.title}\n{candidate.mechanism}",
    )


def _candidate_relevant_source_ids(
    candidate: BreadthCandidate,
    source: CitationSource,
) -> set[str]:
    candidate_signature = _candidate_signature(candidate)
    relevant: set[str] = set()
    for evidence in candidate.evidence:
        record = source.get(evidence.citation_id)
        claim = record.claim(evidence.claim_id) if record is not None else None
        if record is None or claim is None:
            continue
        trusted_text = " ".join((record.title, claim.text, *record.tags))
        evidence_signature = infer_mechanism_signature(trusted_text)
        family_match = (
            candidate_signature.primary_family is not None
            and evidence_signature.primary_family == candidate_signature.primary_family
            and evidence_signature.stack_stage == candidate_signature.stack_stage
        )
        # Provenance validity and ranking relevance are deliberately separate.
        # Unknown or merely token-overlapping evidence remains allowed but does
        # not receive deterministic mechanism-relevance credit.
        if family_match:
            relevant.add(evidence.citation_id.casefold())
    return relevant


def _candidate_semantic_key(candidate: BreadthCandidate) -> tuple[str, ...]:
    signature = _candidate_signature(candidate)
    fingerprint = structural_fingerprint(signature)
    return (
        fingerprint.stack_stage.value if fingerprint.stack_stage else "unknown",
        fingerprint.primary_family or "unknown",
        ",".join(fingerprint.primary_tags),
        ",".join(fingerprint.secondary_tags),
        normalize_research_text(candidate.primary_change),
        normalize_research_text(candidate.mechanism),
        normalize_research_text(candidate.title),
    )


def _structurally_duplicates(
    candidate: MechanismSignature,
    prior: MechanismSignature,
) -> bool:
    """Conservative same-method decision with room for material variants."""
    if candidate.confidence != "confident" or prior.confidence != "confident":
        return False
    candidate_fp = structural_fingerprint(candidate)
    prior_fp = structural_fingerprint(prior)
    if (
        candidate_fp.stack_stage != prior_fp.stack_stage
        or candidate_fp.primary_family != prior_fp.primary_family
    ):
        return False
    # A new tag counts only when it is asserted in primary intervention text;
    # ancillary mechanism prose cannot manufacture novelty.
    return set(candidate_fp.primary_tags).issubset(prior_fp.primary_tags)


def _duplicates_text(
    candidate: BreadthCandidate,
    prior_text: str,
    prior_signature: Optional[MechanismSignature] = None,
) -> bool:
    candidate_signature = _candidate_signature(candidate)
    prior_signature = prior_signature or infer_mechanism_signature(prior_text)
    if (
        candidate_signature.confidence == "confident"
        and prior_signature.confidence == "confident"
    ):
        return _structurally_duplicates(candidate_signature, prior_signature)
    return research_text_similarity(candidate.selection_text(), prior_text) >= NEAR_DUPLICATE_THRESHOLD


def _duplicates_candidate(
    candidate: BreadthCandidate,
    prior: BreadthCandidate,
) -> bool:
    candidate_signature = _candidate_signature(candidate)
    prior_signature = _candidate_signature(prior)
    if (
        candidate_signature.confidence == "confident"
        and prior_signature.confidence == "confident"
    ):
        return _structurally_duplicates(candidate_signature, prior_signature)
    return research_text_similarity(
        candidate.selection_text(), prior.selection_text()
    ) >= NEAR_DUPLICATE_THRESHOLD


def _candidate_quality_key(
    candidate: BreadthCandidate,
    source: CitationSource,
) -> tuple[Any, ...]:
    """Quality-first duplicate survivor order, independent of candidate ID."""
    return (
        -len(_candidate_relevant_source_ids(candidate, source)),
        _candidate_semantic_key(candidate),
    )


def _compatible_primary_families(
    stage: Optional[StackStage],
    left: Optional[str],
    right: Optional[str],
) -> bool:
    if left == right:
        return True
    if stage == StackStage.INFERENCE_ENSEMBLE:
        return {left, right} == {"checkpoint_ensemble", "prediction_averaging"}
    return False


def filter_breadth_candidates(
    plan: BreadthPlan,
    *,
    history: Sequence[RunRecord],
    citation_source: CitationSource,
    protected_candidate_ids: frozenset[str] = frozenset(),
) -> tuple[tuple[BreadthCandidate, ...], tuple[BreadthRejection, ...]]:
    """Apply deterministic safety, evidence, and duplicate filters.

    ``protected_candidate_ids`` is used only by incremental repair.  Those
    candidates already survived this exact filter once, so they are processed
    first to prevent a replacement duplicate from displacing valid retained
    work.  They are nevertheless revalidated through every filter.
    """
    attempted = [
        (historical_hypothesis(record), historical_mechanism_signature(record))
        for record in history if was_attempted(record)
    ]
    eligible: list[BreadthCandidate] = []
    rejected: list[BreadthRejection] = []
    identifiers = {item.candidate_id.casefold() for item in plan.candidates}
    missing_protected = {
        item.casefold() for item in protected_candidate_ids
    } - identifiers
    if missing_protected:
        raise BreadthValidationError(
            "protected breadth candidates are absent from the combined repair pool: "
            + ", ".join(sorted(missing_protected))
        )
    ordered_candidates = sorted(
        plan.candidates,
        key=lambda item: (
            0 if item.candidate_id.casefold() in {
                value.casefold() for value in protected_candidate_ids
            } else 1,
            _candidate_quality_key(item, citation_source),
        ),
    )
    for candidate in ordered_candidates:
        safety_error = _candidate_safety_error(candidate)
        if safety_error is not None:
            rejected.append(BreadthRejection(candidate.candidate_id, safety_error))
            continue
        if not _candidate_has_valid_evidence(candidate, citation_source):
            rejected.append(BreadthRejection(candidate.candidate_id, "invalid citation evidence"))
            continue
        primary_signature = _candidate_signature(candidate)
        described_signature = infer_mechanism_signature(candidate.mechanism)
        if primary_signature.confidence != "confident":
            rejected.append(BreadthRejection(
                candidate.candidate_id,
                "primary_change does not identify one unambiguous mechanism",
            ))
            continue
        if primary_signature.stack_stage != candidate.stack_stage:
            rejected.append(BreadthRejection(
                candidate.candidate_id,
                "declared stack_stage conflicts with primary_change: declared "
                f"{candidate.stack_stage.value}, inferred "
                f"{primary_signature.stack_stage.value}",
            ))
            continue
        described_families = _matched_primary_families(candidate.mechanism)
        described_core_stages = {
            stage for stage in described_families
            if stage != StackStage.OPTIMIZATION_REGULARIZATION
        }
        foreign_core_stage = any(
            stage != primary_signature.stack_stage for stage in described_core_stages
        )
        confident_conflict = (
            described_signature.confidence == "confident"
            and (
                described_signature.stack_stage != primary_signature.stack_stage
                or not _compatible_primary_families(
                    primary_signature.stack_stage,
                    described_signature.primary_family,
                    primary_signature.primary_family,
                )
            )
        )
        if described_signature.ambiguous or foreign_core_stage or confident_conflict:
            rejected.append(BreadthRejection(
                candidate.candidate_id,
                "mechanism conflicts with the declared primary_change",
            ))
            continue
        if any(
            _duplicates_text(candidate, prior_text, prior_signature)
            for prior_text, prior_signature in attempted
        ):
            rejected.append(BreadthRejection(candidate.candidate_id, "duplicates current-run history"))
            continue
        if any(_duplicates_candidate(candidate, prior) for prior in eligible):
            rejected.append(BreadthRejection(candidate.candidate_id, "duplicates breadth batch candidate"))
            continue
        eligible.append(candidate)
    # Do not leak model output order into selection/ranking behavior.
    survivors = sorted(eligible, key=_candidate_semantic_key)
    return tuple(survivors), tuple(sorted(
        rejected, key=lambda item: item.candidate_id.casefold()
    ))


@dataclass(frozen=True)
class BreadthScore:
    candidate: BreadthCandidate
    total: float
    evidence: float
    novelty: float
    coverage: float
    upside: float
    feasibility: float
    budget: float


def mechanism_signature_similarity(
    left: MechanismSignature,
    right: MechanismSignature,
) -> float:
    """Structural similarity; vocabulary changes contribute nothing here."""
    left_fp = structural_fingerprint(left)
    right_fp = structural_fingerprint(right)
    if left_fp.stack_stage is None or right_fp.stack_stage is None:
        return 0.0
    left_primary_tags = set(left_fp.primary_tags)
    right_primary_tags = set(right_fp.primary_tags)
    tag_union = left_primary_tags | right_primary_tags
    tag_similarity = (
        len(left_primary_tags & right_primary_tags) / len(tag_union)
        if tag_union else 1.0
    )
    if (
        left_fp.stack_stage == right_fp.stack_stage
        and left_fp.primary_family == right_fp.primary_family
    ):
        # Renaming an unchanged family remains almost identical.  A genuinely
        # new primary intervention tag creates bounded within-family novelty.
        if left_primary_tags - right_primary_tags:
            return 0.65 + 0.15 * tag_similarity
        return 0.95 + 0.05 * tag_similarity
    if left_fp.stack_stage == right_fp.stack_stage:
        return 0.4 + 0.1 * tag_similarity
    return 0.0


# Centralized, intentionally coarse scoring constants. These create a soft
# preference, not a stage ban; upside/evidence/feasibility can outweigh
# coverage when a repeated-stage idea is materially stronger.
_UPSIDE_SCORE = {"low": 0.0, "medium": 0.15, "high": 0.3}
_RISK_SCORE = {"low": 0.15, "medium": 0.0, "high": -0.15}
_COST_SCORE = {"low": 0.15, "medium": 0.0, "high": -0.15}
_EVIDENCE_PER_SOURCE = 2.25
_STRUCTURAL_NOVELTY_WEIGHT = 3.5
_LEXICAL_NOVELTY_WEIGHT = 0.5
_UNEXPLORED_STAGE_BONUS = 2.5
_COVERAGE_PENALTY_PER_ATTEMPT = 0.75
_MAX_COVERAGE_PENALTY = 2.25
_MOST_RECENT_STAGE_PENALTY = 0.5


def rank_breadth_candidates(
    candidates: Sequence[BreadthCandidate],
    *,
    history: Sequence[RunRecord],
    coverage: StackCoverageSummary,
    context: ResearchContext,
    citation_source: CitationSource,
) -> tuple[BreadthScore, ...]:
    attempted_records = [record for record in history if was_attempted(record)]
    attempted = [historical_hypothesis(record) for record in attempted_records]
    attempted_signatures = [
        historical_mechanism_signature(record) for record in attempted_records
    ]
    latest_iteration = max(
        (
            item.most_recent_iteration
            for item in coverage.stages.values()
            if item.most_recent_iteration is not None
        ),
        default=None,
    )
    scored = []
    for candidate in candidates:
        lexical_similarity = max(
            (research_text_similarity(candidate.selection_text(), prior) for prior in attempted),
            default=0.0,
        )
        candidate_signature = _candidate_signature(candidate)
        structural_similarity = max(
            (
                mechanism_signature_similarity(
                    candidate_signature,
                    prior_signature,
                )
                for prior_signature in attempted_signatures
            ),
            default=0.0,
        )
        novelty = (
            _STRUCTURAL_NOVELTY_WEIGHT * (1.0 - structural_similarity)
            + _LEXICAL_NOVELTY_WEIGHT * (1.0 - lexical_similarity)
        )
        stage_coverage = coverage.for_stage(candidate.stack_stage)
        if stage_coverage.attempts == 0:
            coverage_score = _UNEXPLORED_STAGE_BONUS
        else:
            coverage_score = -min(
                _MAX_COVERAGE_PENALTY,
                stage_coverage.attempts * _COVERAGE_PENALTY_PER_ATTEMPT,
            )
            if stage_coverage.most_recent_iteration == latest_iteration:
                coverage_score -= _MOST_RECENT_STAGE_PENALTY
        evidence_score = (
            min(len(_candidate_relevant_source_ids(candidate, citation_source)), 2)
            * _EVIDENCE_PER_SOURCE
        )
        upside_score = _UPSIDE_SCORE[candidate.expected_upside]
        feasibility_score = (
            _RISK_SCORE[candidate.implementation_risk]
            + _COST_SCORE[candidate.experiment_cost]
        )
        # Self-reported experiment cost influences ranking once, through the
        # weak normal cost prior above.  Remaining budget does not create a
        # second reward/penalty from the same unverified declaration.
        budget_score = 0.0
        total = sum((
            evidence_score,
            novelty,
            coverage_score,
            upside_score,
            feasibility_score,
            budget_score,
        ))
        scored.append(BreadthScore(
            candidate=candidate,
            total=round(total, 6),
            evidence=evidence_score,
            novelty=round(novelty, 6),
            coverage=coverage_score,
            upside=upside_score,
            feasibility=feasibility_score,
            budget=budget_score,
        ))
    return tuple(sorted(
        scored,
        key=lambda item: (
            -item.total,
            item.candidate.stack_stage.value,
            _candidate_semantic_key(item.candidate),
        ),
    ))


def _alignment_tokens(text: str) -> set[str]:
    return {
        token for token in normalize_research_text(text).split()
        if len(token) >= 3 and token not in _ALIGNMENT_STOPWORDS
    }


def _depth_mechanism_signature(proposal: ResearchProposal) -> tuple[MechanismSignature, str]:
    """Fail closed while combining independently inferred change fields.

    Title, hypothesis, target components, and individual implementation steps
    can each state the intervention.  Controls, evidence, risks,
    hyperparameters, feasibility, and evaluation prose are excluded.  An
    ambiguous field or a confident core-family conflict invalidates the whole
    depth signature; repetition and ancillary tags cannot outvote it.
    """
    fields = (
        proposal.title,
        proposal.hypothesis,
        *proposal.implementation.target_components,
        *proposal.implementation.steps,
    )
    signatures = tuple(infer_mechanism_signature(field) for field in fields)
    secondary_tags = tuple(sorted({
        tag for signature in signatures for tag in signature.secondary_tags
    }))
    if any(signature.ambiguous for signature in signatures):
        return MechanismSignature(
            None, None, (), secondary_tags, "ambiguous", True
        ), "\n".join(fields)

    confident = [
        signature for signature in signatures
        if signature.confidence == "confident"
    ]
    core = [
        signature for signature in confident
        if signature.stack_stage != StackStage.OPTIMIZATION_REGULARIZATION
    ]
    candidates = core or confident
    if not candidates:
        return MechanismSignature(
            None, None, (), secondary_tags, "unknown", False
        ), "\n".join(fields)

    chosen = candidates[0]
    for signature in candidates[1:]:
        if (
            signature.stack_stage != chosen.stack_stage
            or not _compatible_primary_families(
                chosen.stack_stage,
                signature.primary_family,
                chosen.primary_family,
            )
        ):
            return MechanismSignature(
                None, None, (), secondary_tags, "ambiguous", True
            ), "\n".join(fields)

    return MechanismSignature(
        chosen.stack_stage,
        chosen.primary_family,
        tuple(sorted({tag for signature in candidates for tag in signature.primary_tags})),
        secondary_tags,
        "confident",
        False,
    ), "\n".join(fields)


def validate_depth_alignment(
    proposal: ResearchProposal,
    selected: BreadthCandidate,
) -> None:
    """Require the depth proposal to preserve the selected stage/mechanism."""
    selected_signature = _candidate_signature(selected)
    actual_signature, method_text = _depth_mechanism_signature(proposal)
    if actual_signature.confidence != "confident":
        raise BreadthValidationError(
            "depth proposal has an unknown or ambiguous primary mechanism"
        )
    if actual_signature.stack_stage != selected_signature.stack_stage:
        raise BreadthValidationError(
            "depth proposal changed the selected stack stage: expected "
            f"{selected.stack_stage.value}, classified "
            f"{actual_signature.stack_stage.value if actual_signature.stack_stage else 'unknown'}"
        )
    if actual_signature.primary_family != selected_signature.primary_family:
        raise BreadthValidationError(
            "depth proposal changed the selected primary mechanism family: expected "
            f"{selected_signature.primary_family}, classified "
            f"{actual_signature.primary_family}"
        )
    selected_text = (
        f"{selected.title} {selected.primary_change} {selected.mechanism}"
    )
    selected_tokens = _alignment_tokens(selected_text)
    proposal_tokens = _alignment_tokens(method_text)
    token_overlap = selected_tokens & proposal_tokens
    required_overlap = 1 if len(selected_tokens) <= 3 else 2
    if len(token_overlap) < required_overlap:
        raise BreadthValidationError(
            "depth proposal is not meaningfully aligned with the selected breadth mechanism"
        )
    selected_evidence = {
        (item.citation_id.casefold(), item.claim_id.casefold())
        for item in selected.evidence
    }
    proposal_evidence = {
        (item.citation_id.casefold(), item.claim_id.casefold())
        for item in proposal.rationale.evidence
    }
    if selected_evidence.isdisjoint(proposal_evidence):
        raise BreadthValidationError(
            "depth proposal did not retain any evidence supporting the selected breadth direction"
        )
