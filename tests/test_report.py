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
from runlog.report import summarize


def _record(iteration, primary=None, patch_path=None, status=None):
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
