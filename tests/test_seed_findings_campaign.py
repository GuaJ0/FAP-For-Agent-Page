"""The one-time exploration campaign (scripts/seed_findings.py).

What this guards: the campaign exists to seed agent/research/findings.jsonl
with HONEST outcomes before the graded run. Two ways that goes wrong quietly --
it runs in --offline mode (whose template library cannot implement most of the
directions, so the records would describe ideas that were never built), or it
writes into the graded run's logs and ledger. Both are checked here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import seed_findings
from agent.research.findings import (
    CONFIDENCE_WELL_TESTED,
    VERDICT_DONT,
    Finding,
    FindingsLedger,
)


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["seed_findings.py", *args])
    return seed_findings.main()


def _capture_argv(monkeypatch) -> dict:
    """Replace run_loop.main with a stub that records the argv built for it."""
    seen: dict = {}

    def _stub() -> int:
        seen["argv"] = list(sys.argv)
        return 0

    monkeypatch.setattr(seed_findings.run_loop, "main", _stub)
    return seen


def test_dry_run_plans_every_backlog_direction_and_spends_nothing(monkeypatch, tmp_path, capsys):
    from agent.research.offline import DEFAULT_BACKLOG

    def _explode(*a, **k):  # nothing may reach the real loop
        raise AssertionError("dry run must not invoke run_loop")

    monkeypatch.setattr(seed_findings.run_loop, "main", _explode)
    assert _run(monkeypatch, "--root", str(tmp_path / "scratch"), "--dry-run") == 0

    out = capsys.readouterr().out
    for entry in DEFAULT_BACKLOG:
        assert entry.key in out
    assert "nothing executed" in out


def test_the_campaign_defaults_to_a_scratch_root_and_scratch_ledger(monkeypatch, tmp_path):
    from agent.research.findings import DEFAULT_FINDINGS_PATH

    seen = _capture_argv(monkeypatch)
    root = tmp_path / "exploration"
    assert _run(monkeypatch, "--root", str(root)) == 0

    argv = seen["argv"]
    assert argv[argv.index("--root") + 1] == str(root)
    ledger = Path(argv[argv.index("--findings-path") + 1])
    assert ledger == root / "findings.jsonl"
    assert ledger != DEFAULT_FINDINGS_PATH


def test_the_campaign_never_runs_in_offline_mode(monkeypatch, tmp_path):
    """--offline swaps in the hand-written template library, which can only
    express ranking-loss variants. A campaign run that way would produce
    real-looking RunRecords for directions nothing ever implemented."""
    seen = _capture_argv(monkeypatch)
    _run(monkeypatch, "--root", str(tmp_path / "s"))

    assert "--offline" not in seen["argv"]


def test_the_campaign_loosens_the_budget_beyond_the_graded_defaults(monkeypatch, tmp_path):
    from agent.config import RetryConfig

    seen = _capture_argv(monkeypatch)
    _run(monkeypatch, "--root", str(tmp_path / "s"))

    argv = seen["argv"]
    attempts = int(argv[argv.index("--max-fix-attempts") + 1])
    backstop = float(argv[argv.index("--idea-backstop-s") + 1])
    assert attempts > RetryConfig.max_fix_attempts
    assert backstop > RetryConfig.idea_time_backstop_s


def test_promote_merges_a_reviewed_scratch_ledger_into_the_committed_one(monkeypatch, tmp_path, capsys):
    scratch, target = tmp_path / "scratch.jsonl", tmp_path / "target.jsonl"
    source = FindingsLedger(scratch)
    for i, (variant, delta) in enumerate(
        [("DIN-SHORT-HISTORY", -0.021), ("DIN-LONG-HISTORY", -0.014), ("DIN-MEAN-POOL", -0.033)],
        start=1,
    ):
        source.record(Finding(
            direction="DIN-SEQUENCE", title="candidate-conditioned history",
            verdict=VERDICT_DONT, decision="revert", delta_vs_incumbent=delta,
            validation_primary=0.59, why="lost to the incumbent", iteration=i,
            attempts=1, variants=(variant,), deltas=(delta,),
            coverage=f"history_length: [{i * 20}]",
        ))

    assert _run(monkeypatch, "--promote",
                "--findings-path", str(scratch), "--promote-into", str(target)) == 0

    promoted = FindingsLedger(target).load()
    assert len(promoted) == 1
    assert promoted[0].attempts == 3
    assert promoted[0].confidence == CONFIDENCE_WELL_TESTED
    assert "well_tested" in capsys.readouterr().out


def test_promote_refuses_an_empty_or_missing_scratch_ledger(monkeypatch, tmp_path):
    missing = tmp_path / "nope.jsonl"
    assert _run(monkeypatch, "--promote",
                "--findings-path", str(missing),
                "--promote-into", str(tmp_path / "t.jsonl")) == 2


def test_promote_runs_nothing(monkeypatch, tmp_path):
    """It is a bookkeeping step: reviewing findings must never re-spend money."""
    def _explode():
        raise AssertionError("--promote must not invoke run_loop")

    monkeypatch.setattr(seed_findings.run_loop, "main", _explode)
    scratch = tmp_path / "s.jsonl"
    FindingsLedger(scratch).record(Finding(
        direction="D", title="t", verdict=VERDICT_DONT, decision="revert",
        delta_vs_incumbent=-0.01, validation_primary=0.5, why="w", iteration=1,
    ))

    assert _run(monkeypatch, "--promote", "--findings-path", str(scratch),
                "--promote-into", str(tmp_path / "t.jsonl")) == 0


# ---------------------------------------------------------------------------
# Diagnostic-only entries must never reach the committed ledger.
#
# LOG-RANDOM-DIAGNOSTIC changes no model: it scores the unchanged incumbent on
# the randomized-exposure log and reports the number. Its delta on validation
# primary is therefore zero by construction, which an Evaluator reads as
# REVERT. Promoting that would plant a "Don't" against a direction nobody has
# actually measured -- the precise false-Don't this campaign exists to prevent.
# Excluded automatically at promote time rather than by remembering to delete
# a line from the scratch ledger first.
# ---------------------------------------------------------------------------

def _dont(direction, variant, delta=-0.0004, iteration=1):
    return Finding(
        direction=direction, title=f"{direction} title", verdict=VERDICT_DONT,
        decision="revert", delta_vs_incumbent=delta, validation_primary=0.6011,
        why="no measurable change against the incumbent", iteration=iteration,
        attempts=1, variants=(variant,), deltas=(delta,), coverage="report_only: [true]",
    )


def test_a_report_only_family_is_excluded_while_real_findings_promote(monkeypatch, tmp_path, capsys):
    scratch, target = tmp_path / "scratch.jsonl", tmp_path / "target.jsonl"
    source = FindingsLedger(scratch)
    source.record(_dont("UNBIASED-VALIDATION", "OFFLINE-LOG-RANDOM-DIAGNOSTIC"))
    source.record(_dont("DIN-SEQUENCE", "OFFLINE-DIN-SHORT-HISTORY", delta=-0.021, iteration=2))
    assert len(source.load()) == 2, "both must be in the scratch ledger to start"

    assert _run(monkeypatch, "--promote",
                "--findings-path", str(scratch), "--promote-into", str(target)) == 0

    promoted = {f.direction for f in FindingsLedger(target).load()}
    assert promoted == {"DIN-SEQUENCE"}          # the real measurement came through
    assert "UNBIASED-VALIDATION" not in promoted  # the diagnostic did not

    out = capsys.readouterr().out
    assert "SKIPPED" in out and "UNBIASED-VALIDATION" in out
    assert "changes no model" in out              # and it says why


def test_the_scratch_ledger_still_keeps_the_diagnostic_finding(monkeypatch, tmp_path):
    """Excluded from promotion, not destroyed: the campaign's own record of
    what the diagnostic reported stays auditable in the scratch ledger."""
    scratch, target = tmp_path / "scratch.jsonl", tmp_path / "target.jsonl"
    FindingsLedger(scratch).record(_dont("UNBIASED-VALIDATION", "OFFLINE-LOG-RANDOM-DIAGNOSTIC"))

    _run(monkeypatch, "--promote", "--findings-path", str(scratch), "--promote-into", str(target))

    assert [f.direction for f in FindingsLedger(scratch).load()] == ["UNBIASED-VALIDATION"]


def test_report_only_families_are_detected_from_the_backlog_not_hardcoded():
    """A future diagnostic entry must be excluded without anyone editing a list."""
    import dataclasses

    from agent.research.offline import DEFAULT_BACKLOG
    from scripts import seed_findings as sf

    assert sf._report_only_families() == {"UNBIASED-VALIDATION"}

    diagnostic = dataclasses.replace(
        next(e for e in DEFAULT_BACKLOG if e.key == "TIME-DRIFT"),
        key="SOME-FUTURE-DIAGNOSTIC",
        hyperparameters={"report_only": [True]},
    )
    with_new = DEFAULT_BACKLOG + (diagnostic,)
    import unittest.mock as mock
    with mock.patch.object(sf, "DEFAULT_BACKLOG", with_new):
        assert "SOME-FUTURE-DIAGNOSTIC" in sf._report_only_families()


def test_a_family_mixing_a_diagnostic_with_real_variants_still_promotes():
    """`all`, not `any`: such a family has genuinely measured something, so its
    verdict is real and must not be silently dropped."""
    import dataclasses
    import unittest.mock as mock

    from agent.research.offline import DEFAULT_BACKLOG
    from scripts import seed_findings as sf

    # Give UNBIASED-VALIDATION a second member that does change the model.
    real_variant = dataclasses.replace(
        next(e for e in DEFAULT_BACKLOG if e.key == "LOG-RANDOM-DIAGNOSTIC"),
        key="LOG-RANDOM-IPS-REWEIGHT",
        hyperparameters={"lambda_ips": [0.1]},
    )
    with mock.patch.object(sf, "DEFAULT_BACKLOG", DEFAULT_BACKLOG + (real_variant,)), \
         mock.patch.dict("agent.research.findings._FAMILY_BY_MEMBER",
                         {"LOG-RANDOM-IPS-REWEIGHT": "UNBIASED-VALIDATION"}):
        assert "UNBIASED-VALIDATION" not in sf._report_only_families()


def test_the_campaign_cannot_be_stopped_early_by_the_stall_rule(monkeypatch, tmp_path):
    """The first real campaign stopped after 3 of 14 directions on "no
    improvement > 0.002 over the last 3 scored iterations", leaving 11
    unmeasured.

    The stall rule is a graded-run mechanism: stop burning budget once you have
    stopped improving. For a coverage pass it is exactly wrong -- most
    directions are EXPECTED to fail, and failing is the measurement. So the
    window has to sit above the number of ideas to be covered, leaving backlog
    exhaustion as the real stopping condition."""
    from agent.research.offline import DEFAULT_BACKLOG

    seen = _capture_argv(monkeypatch)
    _run(monkeypatch, "--root", str(tmp_path / "s"))

    argv = seen["argv"]
    window = int(argv[argv.index("--stall-window") + 1])
    rounds = int(argv[argv.index("--max-iterations") + 1])
    assert rounds == len(DEFAULT_BACKLOG)
    assert window > rounds, "the stall rule could still fire before the backlog is covered"


def test_a_run_of_all_failing_iterations_would_now_survive_the_stall_rule():
    """Directly reproduces the stop we hit: three consecutive non-improving
    iterations. Under the graded default that halts the run; under the
    campaign's window it does not."""
    from datetime import datetime, timedelta, timezone

    from agent.config import ConvergenceConfig
    from agent.convergence import should_stop
    from agent.records import AggregateMetrics, Decision, Event, ResourceUsage, RunRecord, Status
    from scripts.seed_findings import EXPLORATION_STALL_WINDOW

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = []
    for i, primary in enumerate([0.6016, 0.6016, 0.5971, 0.6001]):   # the real numbers
        history.append(RunRecord(
            iteration=i, parent_iteration=0 if i else None,
            timestamp=(t0 + timedelta(minutes=5 * i)).isoformat(),
            hypothesis=f"idea {i}", diff_path="c.json", status=Status.SUCCESS, seeds=[],
            aggregate=AggregateMetrics(primary, 0.0003, primary + 0.066, primary - 0.066, 2),
            delta_vs_current_best=primary - 0.6016,
            decision=Decision.ACCEPT if i == 0 else Decision.REVERT,
            events=[Event("eval_finished", "", "evaluator")],
            resources=ResourceUsage(wall_s=1.0),
        ))

    graded_stop, reason = should_stop(history, ConvergenceConfig())
    assert graded_stop is True and "improvement" in reason      # what actually happened

    campaign_stop, _ = should_stop(
        history, ConvergenceConfig(max_iterations=14, n_window=EXPLORATION_STALL_WINDOW))
    assert campaign_stop is False, "the campaign would still stop early"
