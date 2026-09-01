"""runlog/report.py's human-readable summary.

report.py had no test coverage. It is debug/reporting only -- nothing under
agent/ imports it -- but it is the thing a human reads to find out what a run
did, so it silently drifting out of step with RunRecord is worth catching.
"""
from pathlib import Path

import pytest

from agent.records import (
    AggregateMetrics,
    Decision,
    Event,
    FailureKind,
    ResourceUsage,
    RunLog,
    RunRecord,
    SeedMetrics,
    Status,
)
from runlog.report import (
    RunSpec,
    count_manual_interventions,
    count_manual_interventions_multi,
    render_full_report,
    render_markdown_report,
    summarize,
)


def _record(iteration, primary=None, patch_path=None, status=None, manual_intervention=False, hypothesis=None):
    agg = None
    if primary is not None:
        agg = AggregateMetrics(primary, 0.0, primary, primary, 1)
    return RunRecord(
        iteration=iteration, parent_iteration=None,
        timestamp="2026-01-01T00:00:00+00:00", hypothesis=hypothesis or f"idea {iteration}",
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


# ---------------------------------------------------------------------------
# AUDIT-5: render_markdown_report() -- the actual D3 artifact. summarize()
# only ever gave aggregate counts; a grader reading D3 needs, per iteration,
# the hypothesis, the code diff, the resulting metrics, and every event with
# how it was handled. These tests pin that each of those actually appears,
# not just that the function runs.
# ---------------------------------------------------------------------------

def _successful_record(iteration, patch_path="/runs/1/changes.patch"):
    return RunRecord(
        iteration=iteration, parent_iteration=iteration - 1 if iteration else None,
        timestamp="2026-01-01T00:00:00+00:00",
        hypothesis="Replace pointwise logloss with a pairwise BPR ranking loss",
        diff_path=f"/runs/{iteration}/config.json", patch_path=patch_path,
        status=Status.SUCCESS,
        seeds=[
            SeedMetrics(seed=0, primary=0.61, gauc=0.67, ndcg5=0.54, epochs_run=10, wall_s=30.0, cpu_s=29.5),
            SeedMetrics(seed=1, primary=0.60, gauc=0.66, ndcg5=0.53, epochs_run=9, wall_s=28.0, cpu_s=27.1),
        ],
        aggregate=AggregateMetrics(0.605, 0.005, 0.665, 0.535, 2),
        delta_vs_current_best=0.0035, decision=Decision.ACCEPT,
        events=[Event(type="eval_finished", detail="primary=0.6050", agent_action="evaluator")],
        resources=ResourceUsage(wall_s=58.0, cpu_hours=(29.5 + 27.1) / 3600, tokens_in=1200, tokens_out=340),
    )


def _failed_then_retried_record(iteration):
    """A FAILED (not yet abandoned) record -- exactly where "how an error was
    handled" needs to be visible, per the audit's own wording."""
    return RunRecord(
        iteration=iteration, parent_iteration=0,
        timestamp="2026-01-01T00:05:00+00:00",
        hypothesis="Try a listwise softmax objective",
        diff_path=f"/runs/{iteration}/config.json", patch_path=f"/runs/{iteration}/changes.patch",
        status=Status.FAILED,
        seeds=[SeedMetrics(
            seed=0, primary=None, gauc=None, ndcg5=None, epochs_run=None, wall_s=4.0, cpu_s=3.8,
            failure_kind=FailureKind.CRASH, traceback_tail="NameError: name 'sotfmax' is not defined",
        )],
        aggregate=None, delta_vs_current_best=None, decision=None,
        events=[Event(
            type="retry", detail="fix_attempts=1 idea_elapsed_s=4 reason=will_retry",
            agent_action="orchestrator",
        )],
        resources=ResourceUsage(wall_s=4.0, cpu_hours=3.8 / 3600),
    )


def _write(tmp_path, records):
    log = RunLog(tmp_path / "runs.jsonl")
    for r in records:
        log.append(r)
    return tmp_path / "runs.jsonl"


def test_empty_report_says_so_without_crashing(tmp_path):
    out = render_markdown_report(tmp_path / "runs.jsonl")
    assert "No iterations recorded yet" in out


def test_report_includes_the_summary_block(tmp_path):
    path = _write(tmp_path, [_successful_record(0)])
    out = render_markdown_report(path)
    assert "## Summary" in out
    assert "manual interventions:" in out
    assert "compute:" in out


def test_report_renders_hypothesis_and_code_diff_per_iteration(tmp_path):
    path = _write(tmp_path, [_successful_record(0, patch_path="/runs/0/changes.patch")])
    out = render_markdown_report(path)
    assert "## Iteration 0" in out
    assert "Replace pointwise logloss with a pairwise BPR ranking loss" in out
    assert "/runs/0/changes.patch" in out


def test_report_falls_back_to_the_config_path_when_no_patch_exists(tmp_path):
    path = _write(tmp_path, [_successful_record(0, patch_path=None)])
    out = render_markdown_report(path)
    assert "not available" in out
    assert "/runs/0/config.json" in out


def test_report_renders_resulting_metrics(tmp_path):
    path = _write(tmp_path, [_successful_record(0)])
    out = render_markdown_report(path)
    assert "primary=0.6050" in out
    assert "GAUC=0.6650" in out
    assert "nDCG@5=0.5350" in out
    assert "seed 0: primary=0.6100" in out
    assert "seed 1: primary=0.6000" in out


def test_report_renders_a_failed_iterations_error_and_how_it_was_handled(tmp_path):
    path = _write(tmp_path, [_successful_record(0), _failed_then_retried_record(1)])
    out = render_markdown_report(path)
    assert "## Iteration 1 — failed" in out
    assert "Metrics:** none" in out
    assert "**crash**" in out
    assert "NameError: name 'sotfmax' is not defined" in out
    # The event that says HOW the error was handled, not just that it happened.
    assert "`retry` (orchestrator): fix_attempts=1 idea_elapsed_s=4 reason=will_retry" in out


def test_report_renders_manual_intervention_flag(tmp_path):
    flagged = _successful_record(0)
    flagged.manual_intervention = True  # RunRecord is a plain (non-frozen) dataclass
    path = _write(tmp_path, [flagged])
    out = render_markdown_report(path)
    assert "Manual intervention:** yes" in out


def test_report_generated_against_real_records_from_this_session(tmp_path):
    """Not synthetic: runs against the actual logs/runs.jsonl this session
    produced from a real --offline run_loop.py execution, to prove the
    renderer works on genuine data, not just hand-built fixtures."""
    real_log = Path(__file__).resolve().parent.parent / "logs" / "runs.jsonl"
    if not real_log.exists() or not real_log.read_text().strip():
        pytest.skip("no real logs/runs.jsonl present in this checkout")

    out = render_markdown_report(real_log)

    assert out.startswith("# Run Report")
    assert "## Iteration 0" in out
    assert "**Hypothesis:**" in out
    assert "**Metrics:**" in out


# ---------------------------------------------------------------------------
# Multi-run report: this project relaunches the loop (competition convergence
# stops a run early) and archives the prior run's logs/ rather than
# overwriting it. A grader reading Deliverable 3 needs every run, clearly
# separated, and one intervention count that isn't inflated by counting the
# shared interventions.md once per run.
# ---------------------------------------------------------------------------

def _write_run(tmp_path, name, records):
    path = tmp_path / name / "runs.jsonl"
    log = RunLog(path)
    for r in records:
        log.append(r)
    return path


def test_multi_run_intervention_count_does_not_double_count_the_shared_log(tmp_path):
    iv_path = tmp_path / "interventions.md"
    iv_path.write_text("## Entries\n\n- 2026-01-01T00:00:00Z — relaunched after run 1 converged\n")

    run1 = [_record(1, primary=0.6, manual_intervention=True)]
    run2 = [_record(1, primary=0.6), _record(2, primary=0.6, manual_intervention=True)]

    result = count_manual_interventions_multi([run1, run2], iv_path)

    # 2 auto-detected (one per run) + 1 logged ONCE, not once per run.
    assert result == {"auto_detected": 2, "logged": 1, "total": 3}


def test_multi_run_report_separates_runs_and_shows_the_authoritative_total(tmp_path):
    iv_path = tmp_path / "interventions.md"
    iv_path.write_text("## Entries\n\n- 2026-01-01T00:00:00Z — archived run 1, relaunched\n")

    run1_path = _write_run(tmp_path, "run1", [_successful_record(0)])
    run2_path = _write_run(tmp_path, "run2", [_successful_record(0), _failed_then_retried_record(1)])

    out = render_full_report(
        [
            RunSpec("Run 1", run1_path, "archived -- converged early"),
            RunSpec("Run 2", run2_path, "live -- in progress"),
        ],
        iv_path,
        convergence_narrative="Both runs proposed the same backlog entries in the same order.",
    )

    assert out.startswith("# Full Run Report")
    assert "## Run 1 — archived -- converged early" in out
    assert "## Run 2 — live -- in progress" in out
    assert "Both runs proposed the same backlog entries in the same order." in out
    assert "**Total: 1** (0 auto-detected halt/resume across all runs + 1 logged by hand" in out
    # Each run's own detail still renders (reuses the single-run machinery).
    assert out.count("## Iteration 0") == 2
    assert "## Iteration 1 — failed" in out


def test_summary_truncates_a_multi_paragraph_hypothesis_to_one_line(tmp_path):
    """The full text of a [RESEARCH_PROPOSAL v1] hypothesis is many lines. If
    summarize() inlines all of it, render_markdown_report's per-line bullet
    split turns one summary line into dozens (found generating a real report
    against logs/archive/run_20260901_0934/runs.jsonl)."""
    proposal = (
        "[RESEARCH_PROPOSAL v1]\n"
        "ID: OFFLINE-GAUC-WEIGHTED-BPR\n"
        "TITLE: Positive-count-weighted within-user BPR sampling\n"
        "PARENT ITERATION: 0\n\n"
        "HYPOTHESIS:\nWeight within-user BPR sampling by positive count.\n"
    )
    out = _summary(tmp_path, [_record(0, primary=0.60, hypothesis=proposal)])

    assert "best validation primary: 0.6000 (iteration 0: " in out
    assert "Positive-count-weighted within-user BPR sampling)" in out
    assert "HYPOTHESIS:" not in out  # the full proposal must not leak in here
    assert out.count("\n") == 5  # one summary fact per line, not one per proposal line


def test_multi_run_report_handles_a_run_with_no_records(tmp_path):
    iv_path = tmp_path / "interventions.md"
    empty_path = tmp_path / "empty" / "runs.jsonl"

    out = render_full_report([RunSpec("Run 3", empty_path, "not started")], iv_path)

    assert "## Run 3 — not started" in out
    assert "No iterations recorded." in out
