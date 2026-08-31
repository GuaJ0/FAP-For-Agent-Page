"""Cross-run Do/Don't ledger: what this pipeline has already measured.

WHY THIS EXISTS
---------------
build_research_context() reads only the CURRENT run's RunRecord history, and
deliberately never reads solution/ideas.md or docs/results.md because those
hold test-split numbers. Correct -- but it means a run whose logs/runs.jsonl
starts empty has no institutional memory at all. In our graded practice run the
Research Agent's first proposal was a pairwise ranking loss we had already
measured as a loser; the iteration regressed to 0.3956 from a 0.6016 incumbent
and helped trip the 3-stall convergence rule, ending the run with 47 of 50
iterations and 5h43m of the budget unused.

SYSTEM_PROMPT already says "Do not repeat a measured dead end unless the
proposal states the material mechanism change that makes the new experiment
different". On a fresh run that rule pointed at nothing. This file is what it
points at.

HOW ENTRIES ARE PRODUCED
------------------------
Deterministically, from data the pipeline already produced. Every field is
copied out of a RunRecord or the Evaluator's own commentary Event -- there are
no LLM calls in the write path, and nothing here is hand-curated. A judge
asking "how did the agent know not to retry that?" can be shown the commit in
which the agent logged its own finding.

SAFETY
------
Structurally validation-only, the same boundary as RunRecord. The numeric
fields are copied from RunRecord's validation aggregates, which have no
test-split field to copy from. `why` is Evaluator commentary, and the Evaluator
only ever sees a RunRecord, so it cannot quote a number it was never shown.
Belt and braces on top of that:

  - every entry is scanned on write (see _assert_validation_only), and a
    finding that fails the scan is dropped rather than written;
  - findings travel inside ResearchContext, so the existing fail-closed
    _assert_validation_only_context() in agent/research/agent.py re-checks them
    at prompt-construction time.

DIRECTION FAMILIES AND CONFIDENCE
---------------------------------
A structurally complex direction cannot be settled by one attempt. "DIN-style
sequences don't help" is a claim about the mechanism; a single Coding Agent
generation at one window length is evidence about one implementation of it. A
ledger that records both the same way manufactures false Don'ts, and a false
Don't is worse than no entry at all -- it actively steers Research away from
something nobody has actually tested.

So entries roll up by FAMILY, not by individual proposal id. Every attempt at
a family merges into that family's single entry, accumulating:

  - attempts: how many distinct real attempts were measured;
  - variants: which proposal ids / sweep points those were;
  - deltas: each attempt's measured delta, so the spread is visible;
  - coverage: the hyperparameter range those attempts actually spanned;
  - confidence: derived from attempts, and the field that stops a one-shot
    result from reading as a settled Don't.

A `dont` at confidence "inconclusive" means "one attempt, it lost, nobody has
ruled the mechanism out". A `dont` at "well_tested" means the pipeline spent
three real generations across a stated range and every one lost. Research is
told the difference in prompts.py rather than being left to infer it.

STORAGE
-------
agent/research/findings.jsonl, committed to git. It has to outlive a reset, and
a reset here means archiving logs/ wholesale -- so logs/ is the one place this
must not live. Being in version control is also the audit trail: the history
shows which run learned what, and when.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from agent.config import FORBIDDEN_PAYLOAD_KEYS
from agent.records import Decision, RunRecord

DEFAULT_FINDINGS_PATH = Path(__file__).resolve().parent / "findings.jsonl"

# Bounded on purpose: this is prompt context, not an audit log (runs.jsonl is
# the audit log). At the cap the weakest evidence is evicted first, so a
# well-measured dead end outranks a marginal one.
DEFAULT_MAX_FINDINGS = 40

VERDICT_DO = "do"
VERDICT_DONT = "dont"

# Confidence tiers, derived from attempt count alone -- deliberately not from
# the size of the deltas. How firmly a direction was measured is a question
# about how much of it we tried, not about how badly the tries lost.
CONFIDENCE_INCONCLUSIVE = "inconclusive"   # 1 attempt: not enough to rule anything out
CONFIDENCE_TESTED = "tested"               # 2 attempts
CONFIDENCE_WELL_TESTED = "well_tested"     # 3+ attempts across a stated range

_WELL_TESTED_ATTEMPTS = 3


def confidence_for(attempts: int) -> str:
    if attempts >= _WELL_TESTED_ATTEMPTS:
        return CONFIDENCE_WELL_TESTED
    if attempts >= 2:
        return CONFIDENCE_TESTED
    return CONFIDENCE_INCONCLUSIVE


# Which backlog directions belong to the same research family.
#
# Declared here rather than imported from agent/research/offline.py because
# that module imports this one (via agent.py) -- and because a family is a
# claim about the RESEARCH direction, which outlives whichever backlog entries
# currently implement it. A live-Research proposal with an id nobody declared
# still gets an entry; it simply forms its own single-member family.
#
# tests/test_research_offline.py asserts every DEFAULT_BACKLOG key resolves to
# a family declared here, so a new backlog entry cannot silently go ungrouped.
DIRECTION_FAMILIES: dict[str, tuple[str, ...]] = {
    "RANKING-LOSS": ("HYBRID-BPR", "GAUC-WEIGHTED-BPR"),
    "DIN-SEQUENCE": ("DIN-SHORT-HISTORY", "DIN-LONG-HISTORY", "DIN-MEAN-POOL"),
    "MULTITASK": ("MULTITASK-ENGAGEMENT", "MULTITASK-ALL-ENGAGEMENT", "MULTITASK-CLICK-HEAVY"),
    "WATCHTIME": ("WATCHTIME-AUXILIARY", "WATCHTIME-CENSORED", "WATCHTIME-RATIO"),
    "ARCHITECTURE": ("DEEPFM",),
    "TIME-DRIFT": ("TIME-DRIFT",),
    "UNBIASED-VALIDATION": ("LOG-RANDOM-DIAGNOSTIC",),
}

_FAMILY_BY_MEMBER: dict[str, str] = {
    member: family for family, members in DIRECTION_FAMILIES.items() for member in members
}


def resolve_family(direction: str) -> str:
    """The family a proposal id belongs to, or the id itself if it declares none.

    Tolerates the ``OFFLINE-`` prefix OfflineResearchAgent stamps onto every
    hypothesis_id, so the ledger groups by the backlog key rather than by the
    agent that happened to produce it.
    """
    key = (direction or "").strip()
    bare = key[len("OFFLINE-"):] if key.upper().startswith("OFFLINE-") else key
    return _FAMILY_BY_MEMBER.get(bare.upper(), bare or key)

_TITLE_LIMIT = 120
_WHY_LIMIT = 280

# Same shape as agent/research/agent.py's prompt-time guard. Duplicated rather
# than imported because that module imports this one; the authoritative check
# still runs there, this is the earlier of the two.
_UNSAFE_TEXT_PATTERNS = (
    re.compile(r"\btest[\s_-]*(split|set|primary|gauc|ndcg)\b", re.IGNORECASE),
    re.compile(r"\bhidden[\s_-]*test\b", re.IGNORECASE),
    re.compile(r"\bTEST_METRICS\b"),
)


@dataclass(frozen=True)
class Finding:
    """One measured outcome for one research direction.

    Deliberately a fraction of an IterationSummary: a direction key, what
    happened, by how much, and why -- not a second copy of the run history.
    """

    direction: str                  # the FAMILY key -- the merge key, stable across runs
    title: str                      # short human-readable description of the direction
    verdict: str                    # VERDICT_DO | VERDICT_DONT
    decision: str                   # the Evaluator's own verdict on the decisive attempt
    delta_vs_incumbent: Optional[float]   # the BEST delta any attempt achieved
    validation_primary: Optional[float]   # validation primary of that best attempt
    why: str                        # Evaluator commentary, truncated
    iteration: int

    # --- how much was actually tested ---------------------------------------
    # Defaulted so every field is optional on read: findings.jsonl lines
    # written before these existed load unchanged, as a single-attempt entry,
    # which is exactly what they were.
    attempts: int = 1
    variants: tuple[str, ...] = ()      # proposal ids / sweep points behind those attempts
    deltas: tuple[float, ...] = ()      # one per attempt, so the spread is visible
    coverage: str = ""                  # hyperparameter range the attempts spanned
    confidence: str = CONFIDENCE_INCONCLUSIVE

    @property
    def evidence_strength(self) -> float:
        """How firmly this direction was measured. Used only for eviction.

        Attempt count leads, magnitude breaks ties: at the cap we would rather
        evict one big loss nobody replicated than three consistent attempts
        that together actually close a direction.
        """
        magnitude = abs(self.delta_vs_incumbent) if self.delta_vs_incumbent is not None else 0.0
        return float(self.attempts) + min(magnitude, 0.999)

    @property
    def is_conclusive(self) -> bool:
        """False for a one-attempt result, whatever its verdict."""
        return self.confidence != CONFIDENCE_INCONCLUSIVE

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Finding":
        """Rebuild from one JSONL line.

        Fields absent from the line fall back to the dataclass default rather
        than to None, so a pre-rollup entry reads as the single-attempt,
        inconclusive finding it always was instead of arriving with attempts
        set to None and blowing up the first comparison that touches it.
        """
        fields = cls.__dataclass_fields__
        kwargs = {k: d[k] for k in fields if k in d}
        for key in ("variants", "deltas"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)


def _assert_validation_only(payload: Any) -> None:
    """Raise if a finding could carry hidden-test material."""
    text = json.dumps(payload, sort_keys=True, default=str)
    lowered = text.lower()
    for key in FORBIDDEN_PAYLOAD_KEYS:
        if key in lowered:
            raise ValueError(f"finding contains forbidden key {key!r}")
    for pattern in _UNSAFE_TEXT_PATTERNS:
        if pattern.search(text):
            raise ValueError("finding contains hidden-test material")


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _evaluator_commentary(record: RunRecord) -> str:
    for event in record.events:
        if event.type == "evaluator_commentary" and event.detail:
            return event.detail
    return ""


_HYPERPARAM_LIMIT = 200


def _hyperparameters_section(hypothesis: str) -> str:
    """The HYPERPARAMETERS block of a structured handoff, as one line.

    This is what `coverage` is built from: the settings the attempt actually
    declared. Absent for a legacy or free-text hypothesis, which correctly
    yields empty coverage rather than an invented range.
    """
    match = re.search(
        r"(?ms)^HYPERPARAMETERS:\s*\n(.*?)(?=\n[A-Z][A-Z ]+:\s*\n|\Z)", hypothesis or "",
    )
    if not match:
        return ""
    items = [line.strip(" -") for line in match.group(1).strip().splitlines() if line.strip(" -")]
    return _truncate("; ".join(items), _HYPERPARAM_LIMIT)


# A GAUC below this is anti-correlated with the label -- worse than random
# ordering. Kept in sync with agent/coding/agent.py's SUB_RANDOM_GAUC, and
# duplicated rather than imported: findings.py is imported by the Research
# agent, which has no business importing the Coding agent.
SUB_RANDOM_GAUC = 0.5


def _is_sub_random(record: RunRecord) -> bool:
    """Whether this result ranks worse than chance within users."""
    agg = record.aggregate
    if agg is None or agg.gauc_mean is None:
        return False
    return agg.gauc_mean < SUB_RANDOM_GAUC


def build_finding(
    record: RunRecord,
    *,
    direction: str,
    title: str,
    family: Optional[str] = None,
) -> Optional[Finding]:
    """A Finding for one Evaluator-judged record, or None if it isn't one.

    Only Evaluator-judged outcomes qualify. A technical abandon (the executor
    crashed, the smoke run timed out) carries no delta and no judgment: it means
    the Coding Agent could not build the idea, which is not evidence the
    direction is scientifically dead. Recording those as "don't" would steer
    Research away from directions nobody has actually tested.
    """
    if record.decision is None:
        return None

    # A sub-random result is not evidence about the research direction.
    # Within-user GAUC below 0.5 means the model ordered positives BELOW
    # negatives, which a weak-but-correct implementation does not do -- it is
    # an inverted comparison (flipped gradient sign, swapped pair, scrambled
    # ids). Recording it as a "don't" would attribute a one-character bug to
    # the mechanism and steer every future run away from a direction nobody
    # actually tested.
    #
    # This really happened: an inverted BPR gradient scored 0.3937 primary /
    # 0.3704 GAUC against a 0.6016 incumbent, and adding the missing minus sign
    # to dL/ds took the same code to 0.5864 / 0.6457. Without this guard the
    # ledger would have closed the ranking-loss direction -- the one the task
    # notes rank as most likely to pay off -- on that result.
    #
    # Returning None rather than a "do": nothing was measured either way.
    if _is_sub_random(record):
        return None

    verdict = VERDICT_DO if record.decision == Decision.ACCEPT else VERDICT_DONT
    delta = record.delta_vs_current_best
    return Finding(
        # The merge key is the family, so three DIN variants accumulate into
        # one DIN-SEQUENCE entry instead of three unrelated one-shot Don'ts.
        direction=family or resolve_family(direction),
        title=_truncate(title, _TITLE_LIMIT),
        verdict=verdict,
        decision=record.decision.value,
        delta_vs_incumbent=delta,
        validation_primary=record.aggregate.primary_mean if record.aggregate else None,
        why=_truncate(_evaluator_commentary(record), _WHY_LIMIT),
        iteration=record.iteration,
        # One attempt as constructed. FindingsLedger.record() merges this into
        # whatever the family already holds and recomputes the rollup.
        attempts=1,
        variants=(direction,),
        deltas=() if delta is None else (delta,),
        coverage=_hyperparameters_section(record.hypothesis),
        confidence=CONFIDENCE_INCONCLUSIVE,
    )


def _merge_coverage(prior: str, new: str) -> str:
    """Union of two coverage strings, order-preserving and deduplicated."""
    seen: list[str] = []
    for part in [p.strip() for p in (prior or "").split(";")] + \
                [p.strip() for p in (new or "").split(";")]:
        if part and part not in seen:
            seen.append(part)
    return _truncate("; ".join(seen), _HYPERPARAM_LIMIT)


def _merge(prior: Optional[Finding], new: Finding) -> Finding:
    """Fold one fresh attempt into a family's existing entry.

    Re-running the SAME variant does not inflate the attempt count -- variants
    are deduplicated by name, so a resumed or repeated run cannot manufacture
    confidence the pipeline never earned. A config sweep gets around this
    honestly by labelling each point as its own variant (see agent/sweep.py),
    because a different config point genuinely is a different measurement.

    The rolled-up verdict is the best outcome any attempt achieved: one ACCEPT
    among three attempts makes the family a "do", because the direction demonstrably
    can work even if two implementations of it did not.
    """
    if prior is None:
        return replace(new, confidence=confidence_for(new.attempts))

    variants = list(prior.variants) or [prior.direction]
    is_new_variant = not set(new.variants) <= set(variants)
    for name in new.variants:
        if name not in variants:
            variants.append(name)

    attempts = prior.attempts + (new.attempts if is_new_variant else 0)
    deltas = tuple(prior.deltas) + (tuple(new.deltas) if is_new_variant else ())

    # "Best" means highest delta -- the least-bad attempt if all of them lost.
    prior_best = prior.delta_vs_incumbent
    new_best = new.delta_vs_incumbent
    new_wins = prior_best is None or (new_best is not None and new_best > prior_best)
    winner = new if new_wins else prior

    return Finding(
        direction=prior.direction,
        title=winner.title or prior.title,
        # An ACCEPT anywhere in the family makes the family a "do".
        verdict=VERDICT_DO if VERDICT_DO in (prior.verdict, new.verdict) else VERDICT_DONT,
        decision=winner.decision,
        delta_vs_incumbent=winner.delta_vs_incumbent,
        validation_primary=winner.validation_primary,
        why=winner.why or prior.why,
        iteration=max(prior.iteration, new.iteration),
        attempts=attempts,
        variants=tuple(variants),
        deltas=deltas,
        coverage=_merge_coverage(prior.coverage, new.coverage),
        confidence=confidence_for(attempts),
    )


@dataclass
class FindingsLedger:
    """Bounded, deduplicated, cross-run store of measured directions."""

    path: Path = DEFAULT_FINDINGS_PATH
    max_findings: int = DEFAULT_MAX_FINDINGS

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def load(self) -> tuple[Finding, ...]:
        """Every stored finding. A malformed line is skipped, not fatal.

        Parsed line-by-line rather than through runlog.emit.read_lines: that
        helper is a generator that decodes as it iterates, so a single corrupt
        line aborts the whole sweep and would discard every good finding with
        it. This is advisory prompt context -- degrading to "some findings" is
        right, losing all of them (or the run) is not.
        """
        if not self.path.exists():
            return ()
        out: list[Finding] = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return ()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                finding = Finding.from_json(json.loads(line))
                _assert_validation_only(finding.to_json())
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            out.append(finding)
        return tuple(out)

    def record(self, finding: Optional[Finding]) -> Optional[Finding]:
        """Merge one attempt into its family's entry, then re-cap.

        Returns the entry as stored -- the merged rollup, not the single
        attempt that was passed in, so a caller can see the accumulated
        attempts/confidence rather than what it already knew.

        One entry per FAMILY: a second attempt at the same direction deepens
        that entry instead of appending a competing one, which is both what
        keeps a much-retried dead end from crowding out the list and what makes
        "tested at three config points" expressible at all.

        Never raises -- a ledger problem must not take down a live run.
        """
        if finding is None:
            return None
        try:
            _assert_validation_only(finding.to_json())
        except ValueError:
            return None

        try:
            existing = self.load()
            prior = next((f for f in existing if f.direction == finding.direction), None)
            merged = _merge(prior, finding)
            kept = [f for f in existing if f.direction != finding.direction]
            kept.append(merged)
            self._write(self._cap(kept))
        except OSError:
            return None
        return merged

    def _cap(self, findings: list[Finding]) -> list[Finding]:
        if len(findings) <= self.max_findings:
            return findings
        # Strongest evidence survives; ties broken by recency.
        ranked = sorted(findings, key=lambda f: (f.evidence_strength, f.iteration), reverse=True)
        return ranked[: self.max_findings]

    def _write(self, findings: Sequence[Finding]) -> None:
        """Atomic rewrite -- temp + os.replace, as registry.py and state.py do.

        Not runlog.emit.append_line: that is append-only by design, and this
        file is deduplicated and capped, so an entry has to be replaceable.
        Reading still goes through read_lines.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(
            json.dumps(f.to_json(), sort_keys=True, separators=(",", ":")) + "\n"
            for f in sorted(findings, key=lambda f: (f.verdict, f.direction))
        )
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, self.path)


def findings_for_prompt(findings: Iterable[Finding]) -> tuple[dict[str, Any], ...]:
    """Render findings for ResearchContext, don'ts first, best-evidenced first.

    Don'ts lead because they are the constraint: the prompt rule they serve is
    about not repeating a measured dead end. Within the don'ts, conclusive ones
    lead over inconclusive ones -- a direction closed by three attempts is a
    firmer constraint than one that lost once, and if the prompt gets truncated
    the firm constraints are the ones worth keeping.
    """
    ordered = sorted(
        findings,
        key=lambda f: (
            f.verdict != VERDICT_DONT,
            not f.is_conclusive,
            -f.evidence_strength,
            f.direction,
        ),
    )
    return tuple(f.to_json() for f in ordered)
