"""runlog/report.py's human-readable summary.

report.py had no test coverage. It is debug/reporting only -- nothing under
agent/ imports it -- but it is the thing a human reads to find out what a run
did, so it silently drifting out of step with RunRecord is worth catching.
"""
from pathlib import Path

import pytest

from agent.records import (
    AggregateMetrics,
    ResourceUsage,
    RunLog,
    RunRecord,
    Status,
)
from runlog.report import count_manual_interventions, summarize


def _record(iteration, primary=None, patch_path=None, status=None, manual_intervention=False):
    agg = None
    if primary is not None:
        agg = AggregateMetrics(primary, 0.0, primary, primary, 1)
    return RunRecord(
        iteration=iteration, parent_iteration=None,
        timestamp="2026-01-01T00:00:00+00:00", hypothesis=f"idea {iteration}",
        diff_path=f"/runs/{iteration}/config.json",
        status=status or (Status.SUCCESS if agg else Status.FAILED),
        seeds=[], aggregate=agg, delta_vs_current_best=None, decision=None,
        events=[], resources=ResourceUsage(wall_s=1.0), patch_path=patch_path,
        manual_intervention=manual_intervention,
    )


def _summary(tmp_path, records):
    log = RunLog(tmp_path / "runs.jsonl")
    for r in records:
        log.append(r)
    return summarize(tmp_path / "runs.jsonl")


def test_empty_log(tmp_path):
    assert summarize(tmp_path / "runs.jsonl") == "no iterations recorded yet"


def test_counts_iterations_with_a_recorded_diff(tmp_path):
    out = _summary(tmp_path, [
        _record(0, primary=0.60),                                 # baseline, no patch
        _record(1, primary=0.61, patch_path="/runs/1/changes.patch"),
        _record(2, primary=0.59, patch_path="/runs/2/changes.patch"),
    ])

    assert "code diffs recorded: 2/3 iterations" in out


def test_reports_zero_when_no_producer_made_a_patch(tmp_path):
    """A Fake-agent-only run. 0/N is informative -- it says the code for these
    iterations can't be inspected from the log -- not a fault."""
    out = _summary(tmp_path, [_record(1, primary=0.6), _record(2, primary=0.6)])

    assert "code diffs recorded: 0/2 iterations" in out


def test_counts_failed_iterations_too(tmp_path):
    """A failed attempt is exactly when you most want the diff."""
    out = _summary(tmp_path, [
        _record(1, primary=None, patch_path="/runs/1/changes.patch"),
        _record(2, primary=0.6, patch_path="/runs/2/changes.patch"),
    ])

    assert "code diffs recorded: 2/2 iterations" in out


def test_an_old_log_without_the_field_summarizes_cleanly(tmp_path):
    """runs.jsonl is append-only: lines written before patch_path existed must
    still summarize rather than raise."""
    import json

    payload = _record(1, primary=0.6).to_json()
    del payload["patch_path"]
    (tmp_path / "runs.jsonl").write_text(json.dumps(payload) + "\n")

    out = summarize(tmp_path / "runs.jsonl")

    assert "code diffs recorded: 0/1 iterations" in out


def test_the_rest_of_the_summary_is_unchanged(tmp_path):
    """The addition must not disturb what was already reported."""
    out = _summary(tmp_path, [
        _record(1, primary=0.60, patch_path="/p"),
        _record(2, primary=0.65, patch_path="/p"),
        _record(3, primary=None),
    ])

    assert out.startswith("3 iterations recorded")
    assert "  success: 2" in out
    assert "  failed: 1" in out
    assert "best validation primary: 0.6500 (iteration 2: idea 2)" in out


# ---------------------------------------------------------------------------
# AUDIT-3(b): the manual-intervention count is the SUM of auto-detected
# tier-2 halt/resumes (RunRecord.manual_intervention) and hand-logged entries
# in logs/interventions.md -- the orchestrator cannot see interventions that
# happen outside the loop (SSH in, restart, patch a file), so those need a
# human-maintained record, and the report must count both, not just one.
# ---------------------------------------------------------------------------

def test_intervention_count_is_zero_with_no_flags_and_no_log_file(tmp_path):
    out = _summary(tmp_path, [_record(1, primary=0.6)])
    assert "manual interventions: 0 (0 auto-detected halt/resume, 0 logged in interventions.md)" in out


def test_intervention_count_includes_auto_detected_flags(tmp_path):
    out = _summary(tmp_path, [
        _record(1, primary=0.6),
        _record(2, primary=0.6, manual_intervention=True),
    ])
    assert "manual interventions: 1 (1 auto-detected halt/resume, 0 logged in interventions.md)" in out


def test_intervention_count_includes_hand_logged_entries(tmp_path):
    (tmp_path / "interventions.md").write_text(
        "# Manual interventions log\n\n"
        "## Entries\n\n"
        "- 2026-01-01T00:00:00Z — restarted the process after an OOM kill\n"
        "- 2026-01-02T00:00:00Z — patched a stray import in a generated file\n"
    )
    out = _summary(tmp_path, [_record(1, primary=0.6)])
    assert "manual interventions: 2 (0 auto-detected halt/resume, 2 logged in interventions.md)" in out


def test_intervention_count_sums_both_sources(tmp_path):
    (tmp_path / "interventions.md").write_text("## Entries\n\n- 2026-01-01T00:00:00Z — did a thing\n")
    out = _summary(tmp_path, [
        _record(1, primary=0.6, manual_intervention=True),
        _record(2, primary=0.6, manual_intervention=True),
    ])
    assert "manual interventions: 3 (2 auto-detected halt/resume, 1 logged in interventions.md)" in out


def test_count_manual_interventions_ignores_a_log_file_with_no_entries_section(tmp_path):
    (tmp_path / "interventions.md").write_text("just some unrelated text, no ## Entries header\n")
    result = count_manual_interventions([_record(1, primary=0.6)], tmp_path / "interventions.md")
    assert result == {"auto_detected": 0, "logged": 0, "total": 0}


def test_count_manual_interventions_handles_a_missing_log_file(tmp_path):
    result = count_manual_interventions([_record(1, primary=0.6)], tmp_path / "does_not_exist.md")
    assert result == {"auto_detected": 0, "logged": 0, "total": 0}
