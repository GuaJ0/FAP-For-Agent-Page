from datetime import datetime, timedelta, timezone

from agent.config import BOOTSTRAP_ITERATION, ConvergenceConfig
from agent.convergence import should_stop
from agent.records import AggregateMetrics, ResourceUsage, RunRecord, Status


def _record(iteration, primary=None, ts=None):
    ts = ts or (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=iteration)).isoformat()
    agg = None
    if primary is not None:
        agg = AggregateMetrics(primary_mean=primary, primary_std=0.0, gauc_mean=primary, ndcg5_mean=primary, n_seeds=1)
    return RunRecord(
        iteration=iteration, parent_iteration=None, timestamp=ts, hypothesis="h",
        diff_path=None, status=Status.SUCCESS if agg else Status.FAILED,
        seeds=[], aggregate=agg, delta_vs_current_best=None, decision=None,
        events=[], resources=ResourceUsage(wall_s=1.0),
    )


def test_no_stop_on_empty_history():
    stop, reason = should_stop([], ConvergenceConfig())
    assert stop is False


def test_stops_at_max_iterations():
    cfg = ConvergenceConfig(max_iterations=3, epsilon=0.002, n_window=3, max_wall_s=1e9)
    history = [_record(i, primary=0.5 + i * 0.01) for i in range(1, 4)]
    stop, reason = should_stop(history, cfg)
    assert stop is True
    assert "max_iterations" in reason


def test_stops_at_wall_clock_budget():
    cfg = ConvergenceConfig(max_iterations=1000, epsilon=0.002, n_window=3, max_wall_s=100)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = [
        _record(1, primary=0.5, ts=base.isoformat()),
        _record(2, primary=0.55, ts=(base + timedelta(seconds=200)).isoformat()),
    ]
    stop, reason = should_stop(history, cfg)
    assert stop is True
    assert "wall clock" in reason


def test_stops_when_no_improvement_over_window():
    cfg = ConvergenceConfig(max_iterations=1000, epsilon=0.002, n_window=3, max_wall_s=1e9)
    # best-so-far sequence: 0.50, 0.60, 0.601, 0.6011, 0.6012 -- last 3 deltas all <= eps
    primaries = [0.50, 0.60, 0.601, 0.6011, 0.6012]
    history = [_record(i + 1, primary=p) for i, p in enumerate(primaries)]
    stop, reason = should_stop(history, cfg)
    assert stop is True
    assert "no improvement" in reason


def test_does_not_stop_while_still_improving():
    cfg = ConvergenceConfig(max_iterations=1000, epsilon=0.002, n_window=3, max_wall_s=1e9)
    primaries = [0.50, 0.55, 0.60, 0.65, 0.70]  # each step beats eps
    history = [_record(i + 1, primary=p) for i, p in enumerate(primaries)]
    stop, reason = should_stop(history, cfg)
    assert stop is False


def test_failed_iterations_do_not_count_toward_the_improvement_window():
    cfg = ConvergenceConfig(max_iterations=1000, epsilon=0.002, n_window=3, max_wall_s=1e9)
    history = [
        _record(1, primary=0.50),
        _record(2, primary=None),  # failed, no aggregate
        _record(3, primary=None),
        _record(4, primary=0.60),
    ]
    # only 2 scored records exist -- fewer than n_window + 1 -- so no verdict yet
    stop, reason = should_stop(history, cfg)
    assert stop is False


# ---------------------------------------------------------------------------
# The bootstrap baseline and the three checks.
#
# Orchestrator.bootstrap_baseline() writes a real, non-FAILED RunRecord at
# BOOTSTRAP_ITERATION, so it is visible to all three of should_stop's checks --
# and each one wants something different from it. These tests pin that
# asymmetry, because "making it consistent" in either direction is a plausible
# and wrong future edit.
# ---------------------------------------------------------------------------

def _baseline(primary=0.6015, ts=None):
    """A record shaped like the one bootstrap_baseline() writes."""
    return _record(BOOTSTRAP_ITERATION, primary=primary, ts=ts)


def _research(n, primary=0.55):
    """n research iterations, numbered from 1 as _step() numbers them."""
    return [_record(i, primary=primary + i * 1e-6) for i in range(1, n + 1)]


# -- check (1): max_iterations EXCLUDES the baseline -------------------------

def test_baseline_does_not_consume_a_max_iterations_slot():
    """The headline fix: a run configured for N research iterations must get
    N, not N-1. Before this, baseline + (N-1) research already hit the cap."""
    cfg = ConvergenceConfig(max_iterations=5, n_window=100, max_wall_s=1e9)
    history = [_baseline()] + _research(4)   # 5 non-FAILED records, only 4 research

    stop, reason = should_stop(history, cfg)

    assert stop is False, reason


def test_max_iterations_still_fires_on_the_nth_research_iteration():
    cfg = ConvergenceConfig(max_iterations=5, n_window=100, max_wall_s=1e9)
    history = [_baseline()] + _research(5)

    stop, reason = should_stop(history, cfg)

    assert stop is True
    assert "max_iterations" in reason
    assert "5 concluded research ideas" in reason


def test_max_iterations_is_unchanged_when_there_is_no_baseline():
    """Protects the no-bootstrap case: callers that never bootstrap must see
    exactly the behaviour they always did."""
    cfg = ConvergenceConfig(max_iterations=5, n_window=100, max_wall_s=1e9)

    assert should_stop(_research(4), cfg)[0] is False
    assert should_stop(_research(5), cfg)[0] is True


def test_a_failed_baseline_record_also_does_not_count():
    """bootstrap_baseline() logs a FAILED iteration-0 record when the baseline
    itself fails. That must not consume a slot either -- it is excluded twice
    over (FAILED, and iteration 0)."""
    cfg = ConvergenceConfig(max_iterations=3, n_window=100, max_wall_s=1e9)
    history = [_record(BOOTSTRAP_ITERATION, primary=None)] + _research(2)

    assert should_stop(history, cfg)[0] is False


# -- check (3): the stalled-progress window INCLUDES the baseline ------------

def test_baseline_is_the_first_scored_iteration_in_the_stall_window():
    """The counterpart to the exclusion above, and the reason this fix had to
    be surgical. The baseline is the score everything must beat, so it belongs
    in the trailing window -- otherwise a run that stalls right at the baseline
    takes an extra iteration to notice, which is the failure bootstrapping
    exists to prevent.

    Baseline 0.6015 then three research results that never beat it: with the
    baseline counted that is n_window+1 = 4 scored records, so a verdict is
    available now rather than one iteration later.
    """
    cfg = ConvergenceConfig(max_iterations=1000, epsilon=0.002, n_window=3, max_wall_s=1e9)
    history = [_baseline(0.6015)] + [_record(i, primary=0.5989) for i in (1, 2, 3)]

    scored = [r for r in history if r.aggregate is not None]
    assert len(scored) == 4, "the baseline must be one of the scored records"

    stop, reason = should_stop(history, cfg)

    assert stop is True
    assert "no improvement" in reason


def test_without_the_baseline_the_same_three_results_give_no_verdict_yet():
    """Negative control for the test above: it is genuinely the baseline
    supplying the fourth scored record, not the three research results."""
    cfg = ConvergenceConfig(max_iterations=1000, epsilon=0.002, n_window=3, max_wall_s=1e9)
    history = [_record(i, primary=0.5989) for i in (1, 2, 3)]

    assert should_stop(history, cfg)[0] is False


# -- check (2): wall clock INCLUDES the baseline record ----------------------

def test_wall_clock_window_starts_at_the_baseline_record():
    """Deliberate, and easy to get backwards. RunRecords are timestamped at
    *completion*, so the iteration-0 record marks when the baseline FINISHED --
    i.e. when research began. Its own runtime is already outside the window.

    Keeping it as history[0] therefore makes the window cover iterations 1..N
    in full. Skipping it would restart the clock at iteration 1's completion
    and silently drop iteration 1's duration from the budget -- less accurate,
    and more permissive, than counting it.
    """
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cfg = ConvergenceConfig(max_iterations=1000, n_window=100, max_wall_s=100)
    history = [
        _baseline(ts=base.isoformat()),                                    # research starts
        _record(1, primary=0.55, ts=(base + timedelta(seconds=150)).isoformat()),
    ]

    stop, reason = should_stop(history, cfg)

    assert stop is True
    assert "wall clock" in reason
    # 150s measured from the baseline record, not 0s from iteration 1 alone.
    assert "150s" in reason


def test_dropping_the_baseline_would_have_hidden_that_budget_overrun():
    """Pins the consequence of the alternative: with the baseline record gone,
    the same elapsed time measures as 0s and the run keeps going."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cfg = ConvergenceConfig(max_iterations=1000, n_window=100, max_wall_s=100)
    without_baseline = [_record(1, primary=0.55, ts=(base + timedelta(seconds=150)).isoformat())]

    assert should_stop(without_baseline, cfg)[0] is False
