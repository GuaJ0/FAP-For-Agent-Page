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
from dataclasses import asdict, dataclass
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

    direction: str                  # ResearchProposal.hypothesis_id -- stable across runs
    title: str                      # short human-readable description of the direction
    verdict: str                    # VERDICT_DO | VERDICT_DONT
    decision: str                   # the Evaluator's own verdict
    delta_vs_incumbent: Optional[float]
    validation_primary: Optional[float]
    why: str                        # Evaluator commentary, truncated
    iteration: int

    @property
    def evidence_strength(self) -> float:
        """How firmly this direction was measured. Used only for eviction."""
        return abs(self.delta_vs_incumbent) if self.delta_vs_incumbent is not None else 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Finding":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


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


def build_finding(
    record: RunRecord,
    *,
    direction: str,
    title: str,
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

    verdict = VERDICT_DO if record.decision == Decision.ACCEPT else VERDICT_DONT
    return Finding(
        direction=direction,
        title=_truncate(title, _TITLE_LIMIT),
        verdict=verdict,
        decision=record.decision.value,
        delta_vs_incumbent=record.delta_vs_current_best,
        validation_primary=record.aggregate.primary_mean if record.aggregate else None,
        why=_truncate(_evaluator_commentary(record), _WHY_LIMIT),
        iteration=record.iteration,
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
        """Insert or update one direction, then re-cap. Returns what was stored.

        One entry per direction: re-testing a direction updates its entry rather
        than appending, so a much-retried dead end cannot crowd out the list.
        Never raises -- a ledger problem must not take down a live run.
        """
        if finding is None:
            return None
        try:
            _assert_validation_only(finding.to_json())
        except ValueError:
            return None

        try:
            kept = [f for f in self.load() if f.direction != finding.direction]
            kept.append(finding)
            kept = self._cap(kept)
            self._write(kept)
        except OSError:
            return None
        return finding

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
    """Render findings for ResearchContext, don'ts first.

    Don'ts lead because they are the constraint: the prompt rule they serve is
    about not repeating a measured dead end.
    """
    ordered = sorted(
        findings,
        key=lambda f: (f.verdict != VERDICT_DONT, -f.evidence_strength, f.direction),
    )
    return tuple(f.to_json() for f in ordered)
