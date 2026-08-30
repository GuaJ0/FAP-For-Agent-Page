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
