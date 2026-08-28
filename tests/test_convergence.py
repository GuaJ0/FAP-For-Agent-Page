from datetime import datetime, timedelta, timezone

from agent.config import ConvergenceConfig
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
