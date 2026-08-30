"""Cross-run Do/Don't ledger (agent/research/findings.py).

The gap this closes: build_research_context reads only the current run's
RunRecord history, so a run whose logs/runs.jsonl starts empty has no memory of
directions the pipeline already measured. In the graded practice run that cost
an iteration re-proposing a ranking loss we had already measured as a loser.
"""
import json
from pathlib import Path

import pytest

from agent.records import (
    AggregateMetrics,
    Decision,
    Event,
    ResourceUsage,
    RunRecord,
    Status,
)
from agent.research.context import build_research_context
from agent.research.findings import (
    DEFAULT_FINDINGS_PATH,
    VERDICT_DO,
    VERDICT_DONT,
    Finding,
    FindingsLedger,
    build_finding,
    findings_for_prompt,
)

HANDOFF = """[RESEARCH_PROPOSAL v1]
ID: {hid}
TITLE: {title}
PARENT ITERATION: 0

HYPOTHESIS:
{hypothesis}

WHY THIS SHOULD HELP:
mechanism
"""


def _record(iteration=1, hid="RP-001", decision=Decision.REVERT, primary=0.5989,
            delta=-0.0027, commentary="Lost to the incumbent on validation primary.",
            hypothesis="Replace pointwise logloss with a pairwise BPR ranking loss."):
    events = []
    if commentary:
        events.append(Event(type="evaluator_commentary", detail=commentary,
                            agent_action="evaluator"))
    return RunRecord(
        iteration=iteration, parent_iteration=0, timestamp="2026-01-01T00:00:00+00:00",
        hypothesis=HANDOFF.format(hid=hid, title="a title", hypothesis=hypothesis),
        diff_path="/c.json", status=Status.SUCCESS, seeds=[],
        aggregate=AggregateMetrics(primary, 0.0, primary, primary, 1) if primary else None,
        delta_vs_current_best=delta, decision=decision, events=events,
        resources=ResourceUsage(wall_s=1.0),
    )


def _finding(direction="RP-001", verdict=VERDICT_DONT, delta=-0.0027, iteration=1):
    return Finding(
        direction=direction, title="a direction", verdict=verdict,
        decision="revert" if verdict == VERDICT_DONT else "accept",
        delta_vs_incumbent=delta, validation_primary=0.5989,
        why="why it lost", iteration=iteration,
    )


# ---------------------------------------------------------------------------
# Building an entry from a judged record -- no LLM, all extraction.
# ---------------------------------------------------------------------------

def test_a_reverted_iteration_becomes_a_dont_with_its_delta_and_reason():
    f = build_finding(_record(), direction="RP-001", title="pairwise BPR ranking loss")

    assert f.verdict == VERDICT_DONT
    assert f.decision == "revert"
    assert f.delta_vs_incumbent == pytest.approx(-0.0027)
    assert f.validation_primary == pytest.approx(0.5989)
    assert "Lost to the incumbent" in f.why       # the Evaluator's own words
    assert f.iteration == 1


def test_an_accepted_iteration_becomes_a_do():
    f = build_finding(_record(decision=Decision.ACCEPT, delta=0.0185),
                      direction="RP-002", title="t")
    assert f.verdict == VERDICT_DO


def test_an_abandoned_iteration_becomes_a_dont():
    f = build_finding(_record(decision=Decision.ABANDON), direction="RP-003", title="t")
    assert f.verdict == VERDICT_DONT


def test_an_unjudged_record_produces_nothing():
    """Technical abandons have no Evaluator judgment. A crash means the Coding
    agent couldn't build the idea -- not evidence the direction is dead."""
    assert build_finding(_record(decision=None), direction="RP-004", title="t") is None


def test_missing_commentary_is_tolerated():
    f = build_finding(_record(commentary=""), direction="RP-005", title="t")
    assert f.why == ""


def test_long_text_is_truncated_so_entries_stay_small():
    f = build_finding(_record(commentary="x" * 5000), direction="RP-006", title="y" * 5000)
    assert len(f.why) <= 280
    assert len(f.title) <= 120


# ---------------------------------------------------------------------------
# Persistence, dedup, cap.
# ---------------------------------------------------------------------------

def test_a_finding_survives_a_new_ledger_over_the_same_file(tmp_path):
    """The whole point: it outlives the process, and the run's logs/."""
    path = tmp_path / "findings.jsonl"
    FindingsLedger(path).record(_finding())

    assert [f.direction for f in FindingsLedger(path).load()] == ["RP-001"]


def test_one_entry_per_direction_updates_rather_than_appends(tmp_path):
    path = tmp_path / "findings.jsonl"
    ledger = FindingsLedger(path)
    ledger.record(_finding(delta=-0.001, iteration=1))
    ledger.record(_finding(delta=-0.004, iteration=7))    # same direction, retested

    stored = ledger.load()
    assert len(stored) == 1
    assert stored[0].delta_vs_incumbent == pytest.approx(-0.004)
    assert stored[0].iteration == 7


def test_distinct_directions_coexist(tmp_path):
    ledger = FindingsLedger(tmp_path / "f.jsonl")
    ledger.record(_finding(direction="RP-001"))
    ledger.record(_finding(direction="RP-002", verdict=VERDICT_DO, delta=0.02))

    assert {f.direction for f in ledger.load()} == {"RP-001", "RP-002"}


def test_the_cap_keeps_the_strongest_evidence(tmp_path):
    ledger = FindingsLedger(tmp_path / "f.jsonl", max_findings=3)
    for i, delta in enumerate([-0.0001, -0.05, -0.0002, -0.04, -0.03]):
        ledger.record(_finding(direction=f"RP-{i}", delta=delta, iteration=i))

    kept = {f.direction for f in ledger.load()}
    assert len(kept) == 3
    # The three largest |delta| survive; the two marginal ones are evicted.
    assert kept == {"RP-1", "RP-3", "RP-4"}


def test_a_corrupt_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "f.jsonl"
    FindingsLedger(path).record(_finding())
    with path.open("a") as fh:
        fh.write("{not json\n")

    assert len(FindingsLedger(path).load()) == 1


def test_a_missing_file_loads_as_empty(tmp_path):
    assert FindingsLedger(tmp_path / "nope.jsonl").load() == ()


def test_writes_are_atomic_leaving_no_tmp_file(tmp_path):
    path = tmp_path / "f.jsonl"
    FindingsLedger(path).record(_finding())
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# Safety: structurally incapable of carrying hidden-test material.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("poison", [
    "test_primary was 0.61",
    "the hidden test split shows improvement",
    "TEST_METRICS: {...}",
])
def test_a_poisoned_finding_is_refused_on_write(tmp_path, poison):
    """An Evaluator only ever sees a RunRecord, so it cannot quote a test
    number -- but the ledger refuses one anyway rather than trusting that."""
    path = tmp_path / "f.jsonl"
    ledger = FindingsLedger(path)

    stored = ledger.record(Finding(
        direction="RP-X", title="t", verdict=VERDICT_DONT, decision="revert",
        delta_vs_incumbent=-0.01, validation_primary=0.59, why=poison, iteration=1,
    ))

    assert stored is None
    assert ledger.load() == ()


def test_a_poisoned_line_already_on_disk_is_not_loaded(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text(json.dumps({
        "direction": "RP-X", "title": "t", "verdict": "dont", "decision": "revert",
        "delta_vs_incumbent": -0.01, "validation_primary": 0.59,
        "why": "test_gauc was 0.66", "iteration": 1,
    }) + "\n")

    assert FindingsLedger(path).load() == ()


def test_findings_pass_the_research_agents_fail_closed_check(tmp_path):
    """Findings ride inside ResearchContext, so the existing prompt-time guard
    re-checks them. This proves the guard actually sees them."""
    from agent.research.agent import ResearchInputError, _assert_validation_only_context

    clean = build_research_context([], prior_findings=findings_for_prompt([_finding()]))
    _assert_validation_only_context(clean)          # must not raise

    poisoned = build_research_context([], prior_findings=(
        {"direction": "RP-X", "why": "hidden test primary 0.61"},
    ))
    with pytest.raises(ResearchInputError):
        _assert_validation_only_context(poisoned)


# ---------------------------------------------------------------------------
# Ordering into the prompt.
# ---------------------------------------------------------------------------

def test_donts_lead_and_strongest_evidence_first():
    rendered = findings_for_prompt([
        _finding(direction="D-weak", verdict=VERDICT_DONT, delta=-0.001),
        _finding(direction="A", verdict=VERDICT_DO, delta=0.02),
        _finding(direction="D-strong", verdict=VERDICT_DONT, delta=-0.05),
    ])

    assert [f["direction"] for f in rendered] == ["D-strong", "D-weak", "A"]


def test_context_carries_findings_into_the_prompt_payload():
    ctx = build_research_context([], prior_findings=findings_for_prompt([_finding()]))
    payload = ctx.to_prompt_dict()

    assert payload["prior_findings"][0]["direction"] == "RP-001"
    assert payload["prior_findings"][0]["verdict"] == VERDICT_DONT


def test_context_without_findings_is_unchanged():
    """Existing callers pass nothing and must be unaffected."""
    assert build_research_context([]).prior_findings == ()


def test_the_shipped_default_path_is_outside_logs():
    """It must survive a reset, and a reset archives logs/ wholesale."""
    assert "logs" not in DEFAULT_FINDINGS_PATH.parts
    assert DEFAULT_FINDINGS_PATH.name == "findings.jsonl"
