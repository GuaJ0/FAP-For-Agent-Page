"""Orchestrator.bootstrap_baseline(): the seeded baseline as a real iteration 0.

The bug this fixes: with an empty registry, _handle_successful_run computes
`delta = primary_mean - best.val_primary if best else primary_mean`, so the
first real iteration's "delta" is its absolute score and any Evaluator judging
on `delta > margin` accepts it -- even a result that lost to the baseline.

These tests follow tests/test_orchestrator.py's conventions (tmp_path,
FakeCodingAgent's scripted outcomes, no real data). tests/test_orchestrator.py
itself is deliberately untouched: it encodes the empty-registry behaviour for
callers that never bootstrap, which this change must not alter.
"""
import json
import shutil
from pathlib import Path

import pytest

from agent.agents import Diff, FakeCodingAgent, Idea
from agent.orchestrator import BOOTSTRAP_ITERATION, BootstrapError
from agent.records import Decision, Status
from conftest import make_orchestrator, make_test_config

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "fake_train.py"

BASELINE_IDEA = Idea(
    hypothesis="Baseline: FM with pointwise logloss (seeded solution/train.py)",
    parent_iteration=None,
)


def _baseline_solution(tmp_path, primary=0.6015, name="baseline"):
    """A solution dir standing in for solution/ -- fake_train.py with its
    simulated primary pinned, so a test can script "the baseline scores X"."""
    sol = tmp_path / name
    sol.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE, sol / "train.py")
    cfg = sol / "config.json"
    cfg.write_text(json.dumps({"mode": "normal", "sleep_s": 0.0, "mean": primary, "std": 0.0}))
    # The config is a sibling of train.py, as the real solution/ layout is.
    return Diff(diff_path=str(cfg), solution_dir=str(sol))


def _orc(tmp_path, cfg, outcomes=(), **kw):
    return make_orchestrator(tmp_path, cfg, list(outcomes), **kw)


# ---------------------------------------------------------------------------
# The baseline becomes a real iteration 0.
# ---------------------------------------------------------------------------

def test_baseline_lands_as_a_success_accept_record_at_iteration_zero(tmp_path):
    cfg = make_test_config(tmp_path)
    orc = _orc(tmp_path, cfg)

    record = orc.bootstrap_baseline(BASELINE_IDEA, _baseline_solution(tmp_path))

    assert record.iteration == BOOTSTRAP_ITERATION == 0
    assert record.status == Status.SUCCESS
    assert record.decision == Decision.ACCEPT
    assert record.parent_iteration is None          # iteration 0 is the root
    assert record.aggregate is not None
    assert record.aggregate.primary_mean == pytest.approx(0.6015, abs=1e-3)

    # It is a real record in the shared history, not a side channel.
    history = orc.run_log.read_all()
    assert len(history) == 1 and history[0].iteration == 0


def test_delta_is_none_for_the_baseline_not_its_own_score(tmp_path):
    """Writing the absolute score into a field named "delta" is the original
    bug. There is no current best yet -- establishing one is the point."""
    cfg = make_test_config(tmp_path)
    orc = _orc(tmp_path, cfg)

    record = orc.bootstrap_baseline(BASELINE_IDEA, _baseline_solution(tmp_path))

    assert record.delta_vs_current_best is None


def test_baseline_becomes_the_registry_incumbent_with_a_real_artifact(tmp_path):
    """If nothing ever beats the baseline, registry.best() IS the baseline --
    so a complete artifact has to exist at that path, not just a number."""
    cfg = make_test_config(tmp_path)
    orc = _orc(tmp_path, cfg)

    orc.bootstrap_baseline(BASELINE_IDEA, _baseline_solution(tmp_path))

    entry = orc.registry.best()
    assert entry is not None
    assert entry.iteration == 0
    assert entry.val_primary == pytest.approx(0.6015, abs=1e-3)
    assert Path(entry.checkpoint_path).is_dir()
    assert (Path(entry.checkpoint_path) / "result.json").exists()


def test_the_evaluator_is_not_consulted_for_the_baseline(tmp_path):
    """Judging the baseline against nothing is incoherent, and
    FakeEvaluatorAgent judges on delta_vs_current_best -- the one field that
    has no meaning yet."""
    class ExplodingEvaluator:
        def judge(self, record, history):
            raise AssertionError("the evaluator must not be asked to judge the baseline")

    cfg = make_test_config(tmp_path)
    orc = _orc(tmp_path, cfg, evaluator=ExplodingEvaluator())

    record = orc.bootstrap_baseline(BASELINE_IDEA, _baseline_solution(tmp_path))

    assert record.decision == Decision.ACCEPT


# ---------------------------------------------------------------------------
# The bug it fixes: a losing result must now REVERT.
# ---------------------------------------------------------------------------

def test_a_result_that_loses_to_the_baseline_now_reverts(tmp_path):
    """Replays the documented BPR outcome: baseline 0.6015, BPR 0.5989.

    Before this change the registry was empty, so iteration 1's delta was
    +0.5989 (its own score) and FakeEvaluatorAgent accepted it. It must now
    compute -0.0026 and REVERT.
    """
    cfg = make_test_config(tmp_path)
    orc = _orc(tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0, "mean": 0.5989, "std": 0.0}])

    orc.bootstrap_baseline(BASELINE_IDEA, _baseline_solution(tmp_path, primary=0.6015))
    orc._step(orc.run_log.read_all())

    history = orc.run_log.read_all()
    assert [r.iteration for r in history] == [0, 1]

    result = history[1]
    assert result.status == Status.SUCCESS
    assert result.delta_vs_current_best == pytest.approx(-0.0026, abs=5e-4)
    assert result.decision == Decision.REVERT

    # And the incumbent is still the baseline.
    assert orc.registry.best().iteration == 0


def test_a_result_that_beats_the_baseline_still_accepts(tmp_path):
    """Negative control: the fix must not make everything revert."""
    cfg = make_test_config(tmp_path)
    orc = _orc(tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0, "mean": 0.6200, "std": 0.0}])

    orc.bootstrap_baseline(BASELINE_IDEA, _baseline_solution(tmp_path, primary=0.6015))
    orc._step(orc.run_log.read_all())

    result = orc.run_log.read_all()[1]
    assert result.delta_vs_current_best == pytest.approx(0.0185, abs=5e-4)
    assert result.decision == Decision.ACCEPT
    assert orc.registry.best().iteration == 1


def test_first_real_hypothesis_now_has_the_baseline_as_its_parent(tmp_path):
    """FakeResearchAgent reads parent from history[-1]. With the baseline in
    history that resolves to 0 instead of None -- fixed for free."""
    cfg = make_test_config(tmp_path)
    orc = _orc(tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0}])

    orc.bootstrap_baseline(BASELINE_IDEA, _baseline_solution(tmp_path))
    orc._step(orc.run_log.read_all())

    assert orc.run_log.read_all()[1].parent_iteration == 0


def test_research_iterations_still_start_at_one(tmp_path):
    """Iteration 0 is reserved; _step must not collide with it."""
    cfg = make_test_config(tmp_path)
    orc = _orc(tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0}] * 2)

    orc.bootstrap_baseline(BASELINE_IDEA, _baseline_solution(tmp_path))
    orc._step(orc.run_log.read_all())
    orc._step(orc.run_log.read_all())

    assert [r.iteration for r in orc.run_log.read_all()] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Convergence: the baseline is now the first scored iteration.
# ---------------------------------------------------------------------------

def test_baseline_counts_as_the_first_scored_iteration_for_convergence(tmp_path):
    """should_stop needs n_window+1 scored records before it evaluates
    progress. With n_window=3 that used to mean 4 research iterations; the
    baseline now supplies the first, so 3 flat results are enough to stop."""
    from agent.config import ConvergenceConfig
    from agent.convergence import should_stop

    cfg = make_test_config(tmp_path, convergence=ConvergenceConfig(n_window=3, max_iterations=50))
    flat = {"mode": "normal", "sleep_s": 0.0, "mean": 0.5989, "std": 0.0}
    orc = _orc(tmp_path, cfg, outcomes=[flat] * 3)

    orc.bootstrap_baseline(BASELINE_IDEA, _baseline_solution(tmp_path, primary=0.6015))

    stop, _ = should_stop(orc.run_log.read_all(), cfg.convergence)
    assert stop is False  # one scored record so far

    for _ in range(3):
        orc._step(orc.run_log.read_all())

    scored = [r for r in orc.run_log.read_all() if r.aggregate is not None]
    assert len(scored) == 4          # baseline + 3, the window's minimum
    stop, reason = should_stop(orc.run_log.read_all(), cfg.convergence)
    assert stop is True
    assert "no improvement" in reason


def test_baseline_does_not_count_toward_max_iterations(tmp_path):
    """max_iterations counts research attempts, and the baseline is not one --
    it is the incumbent they are measured against.

    This assertion is the reverse of what it was when bootstrap_baseline()
    first landed. Back then the baseline consumed a slot, so a run configured
    for N research iterations really got N-1 and callers had to pass N+1 to
    compensate. convergence.should_stop() now excludes BOOTSTRAP_ITERATION
    from that count, so max_iterations means what it says.
    """
    from agent.config import ConvergenceConfig

    cfg = make_test_config(tmp_path, convergence=ConvergenceConfig(max_iterations=2))
    orc = _orc(tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0, "mean": 0.7, "std": 0.0}] * 5)

    orc.bootstrap_baseline(BASELINE_IDEA, _baseline_solution(tmp_path))
    orc.run()

    # Baseline + a full 2 research iterations. Previously this stopped at [0, 1].
    assert [r.iteration for r in orc.run_log.read_all()] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Idempotence and crash-resume.
# ---------------------------------------------------------------------------

def test_calling_it_twice_does_not_duplicate_iteration_zero(tmp_path):
    cfg = make_test_config(tmp_path)
    orc = _orc(tmp_path, cfg)
    diff = _baseline_solution(tmp_path)

    first = orc.bootstrap_baseline(BASELINE_IDEA, diff)
    second = orc.bootstrap_baseline(BASELINE_IDEA, diff)

    assert len(orc.run_log.read_all()) == 1
    assert second.iteration == first.iteration == 0
    assert second.aggregate.primary_mean == first.aggregate.primary_mean
    assert len(orc.registry._entries) == 1


def test_a_restart_after_bootstrapping_is_a_no_op(tmp_path):
    """Simulates a crash: a brand-new Orchestrator over the same on-disk
    log/state/registry, as a restarted process would be."""
    cfg = make_test_config(tmp_path)
    diff = _baseline_solution(tmp_path)

    orc1 = _orc(tmp_path, cfg)
    first = orc1.bootstrap_baseline(BASELINE_IDEA, diff)

    orc2 = _orc(tmp_path, cfg)
    again = orc2.bootstrap_baseline(BASELINE_IDEA, diff)

    assert len(orc2.run_log.read_all()) == 1
    assert again.timestamp == first.timestamp     # the same record, not a re-run
    assert orc2.registry.best().iteration == 0


def test_bootstrapping_after_the_loop_has_run_is_refused(tmp_path):
    cfg = make_test_config(tmp_path)
    orc = _orc(tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0}])

    orc._step(orc.run_log.read_all())     # a real iteration, with no baseline

    with pytest.raises(BootstrapError, match="already holds"):
        orc.bootstrap_baseline(BASELINE_IDEA, _baseline_solution(tmp_path))


# ---------------------------------------------------------------------------
# Failure of the baseline itself.
# ---------------------------------------------------------------------------

def test_a_failing_baseline_is_logged_then_raised(tmp_path):
    cfg = make_test_config(tmp_path)
    orc = _orc(tmp_path, cfg)
    sol = tmp_path / "broken"
    sol.mkdir()
    shutil.copy(FIXTURE, sol / "train.py")
    (sol / "config.json").write_text(json.dumps({"mode": "crash", "sleep_s": 0.0}))
    diff = Diff(diff_path=str(sol / "config.json"), solution_dir=str(sol))

    with pytest.raises(BootstrapError, match="baseline itself failed"):
        orc.bootstrap_baseline(BASELINE_IDEA, diff)

    history = orc.run_log.read_all()
    assert len(history) == 1
    assert history[0].iteration == 0
    assert history[0].status == Status.FAILED
    assert history[0].aggregate is None
    assert orc.registry.best() is None       # nothing was registered


def test_bootstrap_can_be_retried_after_a_failure(tmp_path):
    """A failed bootstrap registers nothing, so re-running after fixing the
    cause is the obviously right behaviour -- not a duplicate."""
    cfg = make_test_config(tmp_path)
    orc = _orc(tmp_path, cfg)
    sol = tmp_path / "flaky"
    sol.mkdir()
    shutil.copy(FIXTURE, sol / "train.py")
    cfg_path = sol / "config.json"
    cfg_path.write_text(json.dumps({"mode": "crash", "sleep_s": 0.0}))
    diff = Diff(diff_path=str(cfg_path), solution_dir=str(sol))

    with pytest.raises(BootstrapError):
        orc.bootstrap_baseline(BASELINE_IDEA, diff)

    cfg_path.write_text(json.dumps({"mode": "normal", "sleep_s": 0.0, "mean": 0.6015, "std": 0.0}))
    record = orc.bootstrap_baseline(BASELINE_IDEA, diff)

    assert record.status == Status.SUCCESS
    assert orc.registry.best().iteration == 0
    statuses = [r.status for r in orc.run_log.read_all()]
    assert statuses == [Status.FAILED, Status.SUCCESS]


# ---------------------------------------------------------------------------
# Additive: callers that never bootstrap are unaffected.
# ---------------------------------------------------------------------------

def test_without_bootstrapping_the_old_behaviour_is_unchanged(tmp_path):
    """The empty-registry semantics tests/test_orchestrator.py encodes: the
    first success becomes the provisional incumbent, delta is its own score."""
    cfg = make_test_config(tmp_path)
    orc = _orc(tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0, "mean": 0.5989, "std": 0.0}])

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[0]
    assert record.iteration == 1
    assert record.delta_vs_current_best == pytest.approx(0.5989, abs=1e-3)
    assert record.decision == Decision.ACCEPT
    assert orc.registry.best().iteration == 1
