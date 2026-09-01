import json
from pathlib import Path

import pytest

from agent.config import Config, ExecutorConfig, Paths
from agent.executor import Executor, QuarantineLeakError, assert_no_forbidden_keys
from agent.records import FailureKind
from runlog.emit import read_lines

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "fake_train.py"


def _solution(tmp_path, mode="normal", **cfg_overrides):
    import shutil

    sol_dir = tmp_path / "sol"
    sol_dir.mkdir(exist_ok=True)
    shutil.copy(FIXTURE, sol_dir / "train.py")
    config = {"mode": mode, "sleep_s": 0.01, "mean": 0.6, "std": 0.01, **cfg_overrides}
    config_path = sol_dir / "config.json"
    config_path.write_text(json.dumps(config))
    return sol_dir, config_path


def _executor(tmp_path, per_run_timeout_s=5.0):
    paths = Paths(
        logs_dir=tmp_path / "logs",
        runs_jsonl=tmp_path / "logs" / "runs.jsonl",
        quarantine_dir=tmp_path / "logs" / "quarantine",
        test_metrics_jsonl=tmp_path / "logs" / "quarantine" / "test_metrics.jsonl",
        orchestrator_state=tmp_path / "logs" / "state.json",
        registry_json=tmp_path / "logs" / "registry.json",
        artifacts_dir=tmp_path / "logs" / "artifacts",
    )
    cfg = Config(executor=ExecutorConfig(per_run_timeout_s=per_run_timeout_s), paths=paths)
    return Executor(cfg=cfg), cfg


def test_normal_run_produces_validation_metrics_and_quarantines_test_metrics(tmp_path):
    sol_dir, config_path = _solution(tmp_path, mode="normal")
    ex, cfg = _executor(tmp_path)

    result = ex.run_seed(sol_dir, config_path, seed=0, iteration=1)

    assert result.failure_kind is None
    assert result.primary is not None
    assert result.epochs_run == 10

    # artifact_dir must survive past this call (it's not a tempdir) so the
    # checkpoint registry can point at it later.
    assert result.artifact_dir is not None
    assert Path(result.artifact_dir).is_dir()
    assert (Path(result.artifact_dir) / "result.json").exists()

    quarantined = list(read_lines(cfg.paths.test_metrics_jsonl))
    assert len(quarantined) == 1
    assert quarantined[0]["iteration"] == 1
    assert quarantined[0]["seed"] == 0
    assert "primary" in quarantined[0]["test_metrics"]


def test_crash_is_classified_and_traceback_preserved(tmp_path):
    sol_dir, config_path = _solution(tmp_path, mode="crash")
    ex, _ = _executor(tmp_path)

    result = ex.run_seed(sol_dir, config_path, seed=0, iteration=1)

    assert result.failure_kind == FailureKind.CRASH
    assert result.primary is None
    assert result.artifact_dir is None  # no checkpoint on a failed run
    assert "forced crash" in result.traceback_tail


def test_syntax_error_is_caught_before_any_seed_is_dispatched(tmp_path):
    sol_dir, config_path = _solution(tmp_path, mode="normal")
    (sol_dir / "train.py").write_text('print("unterminated string)\n')
    ex, _ = _executor(tmp_path)

    seeds, agg = ex.run_seeds(sol_dir, config_path, iteration=1, seeds=[0, 1])

    assert agg is None
    assert len(seeds) == 2
    assert all(s.failure_kind == FailureKind.CRASH for s in seeds)
    assert all(s.artifact_dir is None for s in seeds)
    assert "SyntaxError" in seeds[0].traceback_tail


def test_bad_output_is_classified(tmp_path):
    sol_dir, config_path = _solution(tmp_path, mode="bad_output")
    ex, _ = _executor(tmp_path)

    result = ex.run_seed(sol_dir, config_path, seed=0, iteration=1)

    assert result.failure_kind == FailureKind.BAD_OUTPUT
    assert result.primary is None


def test_timeout_is_classified(tmp_path):
    sol_dir, config_path = _solution(tmp_path, mode="timeout")
    ex, _ = _executor(tmp_path, per_run_timeout_s=0.3)

    result = ex.run_seed(sol_dir, config_path, seed=0, iteration=1)

    assert result.failure_kind == FailureKind.TIMEOUT
    assert result.primary is None


def test_run_seeds_aggregates_only_successful_seeds(tmp_path):
    sol_dir, config_path = _solution(tmp_path, mode="normal")
    ex, _ = _executor(tmp_path)

    seeds, agg = ex.run_seeds(sol_dir, config_path, iteration=1, seeds=[0, 1, 2])

    assert len(seeds) == 3
    assert agg is not None
    assert agg.n_seeds == 3


def test_run_seeds_returns_no_aggregate_when_all_seeds_fail(tmp_path):
    sol_dir, config_path = _solution(tmp_path, mode="crash")
    ex, _ = _executor(tmp_path)

    seeds, agg = ex.run_seeds(sol_dir, config_path, iteration=1, seeds=[0, 1])

    assert agg is None
    assert all(s.failure_kind == FailureKind.CRASH for s in seeds)


def test_returned_seed_metrics_never_contain_test_metrics(tmp_path):
    sol_dir, config_path = _solution(tmp_path, mode="normal")
    ex, _ = _executor(tmp_path)

    result = ex.run_seed(sol_dir, config_path, seed=0, iteration=1)
    payload = json.dumps(result.to_json())

    assert "TEST_METRICS" not in payload


def test_assert_no_forbidden_keys_guard():
    assert_no_forbidden_keys({"primary": 0.6, "nested": {"gauc": 0.5}})  # should not raise
    with pytest.raises(QuarantineLeakError):
        assert_no_forbidden_keys({"test_primary": 0.6})
    with pytest.raises(QuarantineLeakError):
        assert_no_forbidden_keys({"nested": [{"gauc_test": 0.6}]})


# ---------------------------------------------------------------------------
# AUDIT-4: cpu_s must measure real CPU time (resource.getrusage deltas), not
# stand in for wall_s under a different name. The two tests below are a pair
# specifically because either one alone wouldn't prove that: a large cpu_s
# could just mean "wall_s was copied here too," and a small one could just
# mean "the fixture barely ran." Together they show the measurement tracks
# actual CPU-bound work and does NOT count time spent merely sleeping.
# ---------------------------------------------------------------------------

def _cpu_burn_solution(tmp_path, burn_s):
    """A minimal train.py that busy-loops on real CPU (via time.process_time,
    not time.sleep) for ~burn_s seconds, then writes a valid result.json.
    Deliberately not fake_train.py: that fixture's "work" is time.sleep,
    which yields the CPU and would prove nothing about a CPU-time measurement."""
    sol_dir = tmp_path / "sol"
    sol_dir.mkdir(exist_ok=True)
    (sol_dir / "train.py").write_text(f'''
import argparse, json, time
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
t0 = time.process_time()
x = 0
while time.process_time() - t0 < {burn_s}:
    x += 1  # real CPU-bound work, not a sleep
json.dump({{"primary": 0.6, "gauc": 0.6, "ndcg5": 0.6, "epochs_run": 1}}, open(a.out, "w"))
''')
    config_path = sol_dir / "config.json"
    config_path.write_text("{}")
    return sol_dir, config_path


def test_cpu_s_measures_real_cpu_bound_work(tmp_path):
    sol_dir, config_path = _cpu_burn_solution(tmp_path, burn_s=0.3)
    ex, _ = _executor(tmp_path)

    result = ex.run_seed(sol_dir, config_path, seed=0, iteration=1)

    assert result.failure_kind is None, result.traceback_tail
    assert result.cpu_s >= 0.25, f"expected ~0.3s of measured CPU time, got {result.cpu_s}"


def test_cpu_s_does_not_count_pure_sleep_as_cpu_time(tmp_path):
    """The whole point of measuring via getrusage instead of wall_s: a
    process that sleeps for 0.3s (yielding the CPU the entire time, like
    fake_train.py's simulated training delay) must NOT show ~0.3s of cpu_s."""
    sol_dir, config_path = _solution(tmp_path, mode="normal", sleep_s=0.3)
    ex, _ = _executor(tmp_path)

    result = ex.run_seed(sol_dir, config_path, seed=0, iteration=1)

    assert result.failure_kind is None
    assert result.cpu_s < 0.1, f"a 0.3s sleep should not register as CPU time, got {result.cpu_s}"


# ---------------------------------------------------------------------------
# Config-only sweeps (agent/sweep.py): drive run_seeds() directly against ONE
# fixed implementation with only the config swapped. No research.propose(),
# no coding.implement() -- see agent/sweep.py's docstring for why this is a
# separate mode rather than a flag on Orchestrator._step().
# ---------------------------------------------------------------------------

def _sweep(tmp_path, evaluator=None, seeds=(0,)):
    from agent.agents import FakeEvaluatorAgent
    from agent.records import RunLog
    from agent.sweep import ConfigSweep

    ex, cfg = _executor(tmp_path)
    return ConfigSweep(
        executor=ex,
        evaluator=evaluator or FakeEvaluatorAgent(),
        run_log=RunLog(cfg.paths.runs_jsonl),
        cfg=cfg,
        seeds=seeds,
    ), cfg


def test_config_sweep_reruns_one_implementation_at_several_config_points(tmp_path):
    """The implementation is fixed; only the config changes. Every point must
    produce a real, Evaluator-judged RunRecord in the run log."""
    from agent.sweep import ConfigPoint

    sol_dir, base_config = _solution(tmp_path, mode="normal", mean=0.60)
    sweep, cfg = _sweep(tmp_path)

    results = sweep.run(
        solution_dir=sol_dir,
        base_config=base_config,
        points=[ConfigPoint("mean-0.55", {"mean": 0.55}),
                ConfigPoint("mean-0.70", {"mean": 0.70})],
        hypothesis="config sensitivity of the fixed implementation",
        incumbent_primary=0.60,
    )

    assert [r.point.label for r in results] == ["mean-0.55", "mean-0.70"]
    assert all(r.record.aggregate is not None for r in results)
    assert all(r.record.decision is not None for r in results)   # really judged
    # The delta is measured against the incumbent passed in, not against 0.
    assert results[0].record.delta_vs_current_best < 0
    assert results[1].record.delta_vs_current_best > 0
    # And every point is durably in the run log, like any other iteration.
    assert len(list(read_lines(cfg.paths.runs_jsonl))) == 2


def test_config_sweep_leaves_the_swept_implementation_untouched(tmp_path):
    """The point of a config-only sweep is that the code is the control. If the
    driver wrote configs into the solution dir it would be mutating the thing
    it is holding fixed."""
    from agent.sweep import ConfigPoint

    sol_dir, base_config = _solution(tmp_path, mode="normal")
    before = {p.name: p.read_bytes() for p in sorted(sol_dir.iterdir()) if p.is_file()}
    sweep, _ = _sweep(tmp_path)

    sweep.run(
        solution_dir=sol_dir, base_config=base_config,
        points=[ConfigPoint("mean-0.5", {"mean": 0.5})],
        hypothesis="h",
    )

    after = {p.name: p.read_bytes() for p in sorted(sol_dir.iterdir()) if p.is_file()}
    assert after == before


def test_config_sweep_merges_overrides_over_the_base_config(tmp_path):
    """Keys the point does not mention must survive from the base config --
    otherwise a 'config-only' sweep silently changes more than one thing."""
    from agent.sweep import ConfigPoint, load_flat_config

    sol_dir, base_config = _solution(tmp_path, mode="normal", mean=0.6, std=0.01)
    sweep, _ = _sweep(tmp_path)

    results = sweep.run(
        solution_dir=sol_dir, base_config=base_config,
        points=[ConfigPoint("mean-0.42", {"mean": 0.42})],
        hypothesis="h",
    )

    written = load_flat_config(results[0].config_path)
    assert written["mean"] == 0.42          # overridden
    assert written["std"] == 0.01           # inherited
    assert written["mode"] == "normal"      # inherited


def test_config_sweep_logs_a_failed_point_without_judging_it(tmp_path):
    """A point whose every seed crashed has no aggregate, so there is nothing
    for the Evaluator to judge -- same rule the main loop applies."""
    from agent.sweep import ConfigPoint

    sol_dir, base_config = _solution(tmp_path, mode="normal")
    sweep, cfg = _sweep(tmp_path)

    results = sweep.run(
        solution_dir=sol_dir, base_config=base_config,
        points=[ConfigPoint("crash", {"mode": "crash"})],
        hypothesis="h",
    )

    record = results[0].record
    assert record.aggregate is None
    assert record.decision is None                      # not judged
    assert record.status.value == "failed"
    assert len(list(read_lines(cfg.paths.runs_jsonl))) == 1   # but still logged


def test_config_sweep_never_registers_a_checkpoint(tmp_path):
    """A sweep is a measurement, not a promotion path: it must not be able to
    move the incumbent, even when a point beats it."""
    from agent.registry import CheckpointRegistry
    from agent.sweep import ConfigPoint

    sol_dir, base_config = _solution(tmp_path, mode="normal")
    sweep, cfg = _sweep(tmp_path)

    sweep.run(
        solution_dir=sol_dir, base_config=base_config,
        points=[ConfigPoint("great", {"mean": 0.99})],
        hypothesis="h",
        incumbent_primary=0.10,
    )

    assert CheckpointRegistry(cfg.paths.registry_json).best() is None


def test_config_sweep_rejects_a_solution_dir_with_no_train_py(tmp_path):
    from agent.sweep import ConfigPoint, ConfigSweepError

    sol_dir, base_config = _solution(tmp_path, mode="normal")
    (sol_dir / "train.py").unlink()
    sweep, _ = _sweep(tmp_path)

    with pytest.raises(ConfigSweepError):
        sweep.run(solution_dir=sol_dir, base_config=base_config,
                  points=[ConfigPoint("p", {})], hypothesis="h")


def test_sweep_cli_parses_single_key_and_multi_key_points():
    """--sweep k=a,b,c is one point per value; --point label:k=v,k=v is one
    explicitly labelled point. Values keep their JSON type, so a config gets
    the number 0.05 rather than the string "0.05"."""
    from scripts.config_sweep import parse_points

    points = parse_points(["lambda_bpr=0.05,0.1,0.2"], ['wide:k=32,loss="bpr"'])

    assert [p.label for p in points] == [
        "lambda_bpr=0.05", "lambda_bpr=0.1", "lambda_bpr=0.2", "wide",
    ]
    assert points[0].overrides == {"lambda_bpr": 0.05}      # float, not "0.05"
    assert points[3].overrides == {"k": 32, "loss": "bpr"}


def test_sweep_cli_rejects_malformed_point_specs():
    from agent.sweep import ConfigSweepError
    from scripts.config_sweep import parse_points

    for sweeps, points in ((["nokey"], []), ([], ["nolabel"]), ([], ["label:novalue"])):
        with pytest.raises(ConfigSweepError):
            parse_points(sweeps, points)
