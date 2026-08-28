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
