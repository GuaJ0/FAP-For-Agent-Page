"""Scoring integrity: executor.py must not believe a result.json that claims
metrics its own persisted predictions don't support.

Follows tests/test_executor.py's conventions -- tmp_path, deterministic, no
network, no KuaiRand data. The adversary is fixtures/lying_train.py, which
generates a small synthetic ranking problem and can be told to report numbers
it never computed.
"""
import json
import shutil
from pathlib import Path

import pytest

from slow_helpers import kuairand_path, requires_data

from agent.config import Config, ExecutorConfig, Paths
from agent.executor import Executor
from agent.records import FailureKind
from agent.verification import (
    Status,
    VerificationOutcome,
    find_raw_predictions,
    load_evaluate,
    verify_result,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LIAR = REPO_ROOT / "fixtures" / "lying_train.py"
HONEST_FIXTURE = REPO_ROOT / "fixtures" / "fake_train.py"
VENDORED_EVALUATE = REPO_ROOT / "harness" / "evaluate.py"
SOLUTION_TRAIN_PY = REPO_ROOT / "solution" / "train.py"


def _solution(tmp_path, **cfg):
    """A solution dir laid out the way the CodingAgent lays real ones out:
    train.py plus a copy of the vendored evaluate.py next to it."""
    sol_dir = tmp_path / "sol"
    sol_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(LIAR, sol_dir / "train.py")
    shutil.copy(VENDORED_EVALUATE, sol_dir / "evaluate.py")
    config_path = sol_dir / "config.json"
    config_path.write_text(json.dumps({"mode": "honest", **cfg}))
    return sol_dir, config_path


def _executor(tmp_path, **kwargs):
    logs = tmp_path / "logs"
    paths = Paths(
        logs_dir=logs,
        runs_jsonl=logs / "runs.jsonl",
        quarantine_dir=logs / "quarantine",
        test_metrics_jsonl=logs / "quarantine" / "test_metrics.jsonl",
        orchestrator_state=logs / "state.json",
        registry_json=logs / "registry.json",
        artifacts_dir=logs / "artifacts",
    )
    cfg = Config(executor=ExecutorConfig(per_run_timeout_s=60.0), paths=paths)
    return Executor(cfg=cfg, **kwargs)


# --------------------------------------------------------------------------
# The headline case: a train.py that misreports its metrics gets caught.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw_format", ["npz", "json"])
def test_executor_rejects_a_train_py_that_misreports_its_metrics(tmp_path, raw_format):
    sol_dir, config_path = _solution(
        tmp_path, mode="inflate", raw_format=raw_format,
        claimed={"primary": 0.95, "gauc": 0.97, "ndcg5": 0.93},
    )
    ex = _executor(tmp_path)

    result = ex.run_seed(sol_dir, config_path, seed=0, iteration=1)

    assert result.failure_kind == FailureKind.METRIC_MISMATCH
    assert result.primary is None
    assert result.artifact_dir is None  # a caught liar is a failed run, not a checkpoint
    assert "claimed 0.950000" in result.traceback_tail
    assert "recomputed" in result.traceback_tail


def test_the_same_fixture_being_honest_passes(tmp_path):
    """Negative control: the mismatch above is caused by the lying, not by the
    fixture or the npz round-trip being inherently unverifiable."""
    sol_dir, config_path = _solution(tmp_path, mode="honest")
    ex = _executor(tmp_path)

    result = ex.run_seed(sol_dir, config_path, seed=0, iteration=1)

    assert result.failure_kind is None
    assert result.primary is not None
    assert result.artifact_dir is not None


def test_a_small_but_real_overstatement_is_caught(tmp_path):
    """The check has to bite well below "obviously fabricated" -- an error of
    1e-3 is half the convergence epsilon and would still corrupt decisions."""
    sol_dir, config_path = _solution(tmp_path, mode="honest")
    ex = _executor(tmp_path)
    honest = ex.run_seed(sol_dir, config_path, seed=0, iteration=1)

    sol_dir2, config_path2 = _solution(
        tmp_path / "b", mode="inflate",
        claimed={
            "primary": honest.primary + 1e-3,
            "gauc": honest.gauc,
            "ndcg5": honest.ndcg5,
        },
    )
    result = _executor(tmp_path / "b").run_seed(sol_dir2, config_path2, seed=0, iteration=1)

    assert result.failure_kind == FailureKind.METRIC_MISMATCH
    assert "primary" in result.traceback_tail


# --------------------------------------------------------------------------
# Backwards compatibility: solutions that persist nothing are unaffected.
# --------------------------------------------------------------------------

def test_solution_that_persists_no_predictions_is_skipped_not_failed(tmp_path):
    sol_dir, config_path = _solution(tmp_path, mode="no_raw")
    ex = _executor(tmp_path)

    result = ex.run_seed(sol_dir, config_path, seed=0, iteration=1)

    assert result.failure_kind is None
    assert result.primary is not None


def test_the_preexisting_fake_train_fixture_still_succeeds(tmp_path):
    """fixtures/fake_train.py predates verification and writes no raw
    predictions. It must keep working untouched."""
    sol_dir = tmp_path / "sol"
    sol_dir.mkdir()
    shutil.copy(HONEST_FIXTURE, sol_dir / "train.py")
    config_path = sol_dir / "config.json"
    config_path.write_text(json.dumps({"mode": "normal", "sleep_s": 0.0}))

    result = _executor(tmp_path).run_seed(sol_dir, config_path, seed=0, iteration=1)

    assert result.failure_kind is None
    assert result.artifact_dir is not None


def test_verification_can_be_switched_off(tmp_path):
    sol_dir, config_path = _solution(
        tmp_path, mode="inflate", claimed={"primary": 0.95, "gauc": 0.97, "ndcg5": 0.93},
    )
    ex = _executor(tmp_path, verify_metrics=False)

    result = ex.run_seed(sol_dir, config_path, seed=0, iteration=1)

    assert result.failure_kind is None
    assert result.primary == pytest.approx(0.95)


# --------------------------------------------------------------------------
# Malformed prediction dumps fail rather than being silently skipped.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["ragged", "nan_score"])
def test_unusable_prediction_dumps_fail_the_run(tmp_path, mode):
    sol_dir, config_path = _solution(tmp_path, mode=mode, raw_format="json")
    ex = _executor(tmp_path)

    result = ex.run_seed(sol_dir, config_path, seed=0, iteration=1)

    assert result.failure_kind == FailureKind.METRIC_MISMATCH
    assert result.primary is None


def test_corrupt_npz_is_reported_not_raised(tmp_path):
    out_dir = tmp_path / "art"
    out_dir.mkdir()
    (out_dir / "val_predictions.npz").write_bytes(b"definitely not an npz")

    outcome = verify_result(out_dir, {"primary": 0.6, "gauc": 0.6, "ndcg5": 0.6})

    assert outcome.status == Status.UNREADABLE
    assert outcome.failed


# --------------------------------------------------------------------------
# verify_result unit-level behaviour.
# --------------------------------------------------------------------------

def _discriminative_split():
    """Three users with mixed labels and imperfect ranking, so GAUC and
    nDCG@5 come out to different numbers."""
    users = ["a", "a", "a", "a", "b", "b", "b", "c", "c", "c", "c"]
    labels = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    scores = [0.9, 0.8, 0.3, 0.1, 0.2, 0.85, 0.7, 0.4, 0.1, 0.6, 0.5]
    return users, labels, scores


def _dump(out_dir, users, labels, scores):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "val_predictions.json").write_text(
        json.dumps({"user_ids": users, "labels": labels, "scores": scores})
    )


def test_verify_result_recomputes_through_the_vendored_evaluate(tmp_path):
    users, labels, scores = _discriminative_split()
    truth = load_evaluate()(users, labels, scores)
    _dump(tmp_path, users, labels, scores)

    outcome = verify_result(tmp_path, {
        "primary": truth["primary"], "gauc": truth["GAUC"], "ndcg5": truth["nDCG@5"],
    })

    assert outcome.status == Status.OK
    assert not outcome.failed
    assert outcome.recomputed["primary"] == pytest.approx(truth["primary"])


def test_verify_result_catches_a_swapped_gauc_and_ndcg(tmp_path):
    """A plausible real bug, not just fabrication: reporting evaluate()'s two
    metrics the wrong way round. primary stays right, so only a per-metric
    check finds it."""
    # Deliberately a split where GAUC (0.5) and nDCG@5 (0.83) differ -- on
    # data where they coincide, a swap is undetectable and the test is vacuous.
    users, labels, scores = _discriminative_split()
    truth = load_evaluate()(users, labels, scores)
    assert truth["GAUC"] != pytest.approx(truth["nDCG@5"]), "swap would be a no-op"
    _dump(tmp_path, users, labels, scores)

    outcome = verify_result(tmp_path, {
        "primary": truth["primary"],
        "gauc": truth["nDCG@5"],   # swapped
        "ndcg5": truth["GAUC"],    # swapped
    })

    assert outcome.status == Status.MISMATCH
    assert "gauc" in outcome.detail and "ndcg5" in outcome.detail


def test_missing_file_skips(tmp_path):
    outcome = verify_result(tmp_path, {"primary": 0.6, "gauc": 0.6, "ndcg5": 0.6})
    assert outcome.status == Status.SKIPPED
    assert not outcome.failed
    assert find_raw_predictions(tmp_path) is None


def test_tolerance_is_honoured(tmp_path):
    users, labels, scores = ["a", "a"], [1, 0], [0.9, 0.1]
    truth = load_evaluate()(users, labels, scores)
    _dump(tmp_path, users, labels, scores)
    claimed = {"primary": truth["primary"] + 1e-4, "gauc": truth["GAUC"], "ndcg5": truth["nDCG@5"]}

    assert verify_result(tmp_path, claimed, tol=1e-6).status == Status.MISMATCH
    assert verify_result(tmp_path, claimed, tol=1e-2).status == Status.OK


def test_verification_outcome_detail_carries_no_forbidden_keys(tmp_path):
    """The mismatch detail becomes traceback_tail, which is agent-facing, so
    it goes through the same quarantine guard as everything else."""
    from agent.executor import assert_no_forbidden_keys

    users, labels, scores = ["a", "a"], [1, 0], [0.9, 0.1]
    _dump(tmp_path, users, labels, scores)
    outcome = verify_result(tmp_path, {"primary": 0.99, "gauc": 0.99, "ndcg5": 0.99})

    assert outcome.status == Status.MISMATCH
    assert_no_forbidden_keys({
        "detail": outcome.detail, "claimed": outcome.claimed, "recomputed": outcome.recomputed,
    })
    assert "TEST_METRICS" not in outcome.detail


def test_vendored_evaluate_is_loaded_by_path_not_sys_path():
    """load_evaluate() must not put harness/ on sys.path, and must not claim
    the bare name `evaluate` in sys.modules.

    harness/ is a verbatim copy of the starter kit, whose modules are flat
    top-level names (`evaluate`, `data`). Either kind of pollution would let
    the executor's own process shadow -- or be shadowed by -- an unrelated
    module of the same name. So it loads by path under a private alias.
    """
    import sys

    before = list(sys.path)
    fn = load_evaluate()

    assert sys.path == before, "load_evaluate() mutated sys.path"
    assert sys.modules["_vendored_evaluate"].__file__ == str(VENDORED_EVALUATE)
    assert fn(["a", "a"], [1, 0], [0.9, 0.1])["primary"] == pytest.approx(1.0)


def test_outcome_failed_property():
    assert VerificationOutcome(Status.MISMATCH, "x").failed
    assert VerificationOutcome(Status.UNREADABLE, "x").failed
    assert not VerificationOutcome(Status.OK, "x").failed
    assert not VerificationOutcome(Status.SKIPPED, "x").failed


# --------------------------------------------------------------------------
# The full-scale honest case: verification must not be so tight that a real
# 1.14M-row training run trips it. Slow and opt-in (needs the real dataset).
# --------------------------------------------------------------------------

@pytest.mark.slow
@requires_data
def test_iteration_zero_survives_its_own_verification(tmp_path):
    """End-to-end: the real baseline, run through the real executor, passes
    the scoring-integrity check. Proves the check isn't calibrated so tightly
    that an honest full-scale run trips it."""
    import shutil

    sol = tmp_path / "sol"
    sol.mkdir()
    shutil.copy(SOLUTION_TRAIN_PY, sol / "train.py")
    for f in ("evaluate.py", "data.py"):
        shutil.copy(REPO_ROOT / "harness" / f, sol / f)
    cfg_path = sol / "config.json"
    cfg_path.write_text(json.dumps({
        "data_dir": str(kuairand_path()), "epochs": 3, "patience": 4,
    }))

    logs = tmp_path / "logs"
    paths = Paths(
        logs_dir=logs, runs_jsonl=logs / "runs.jsonl",
        quarantine_dir=logs / "q", test_metrics_jsonl=logs / "q" / "test_metrics.jsonl",
        orchestrator_state=logs / "s.json", registry_json=logs / "r.json",
        artifacts_dir=logs / "artifacts",
    )
    ex = Executor(cfg=Config(executor=ExecutorConfig(per_run_timeout_s=900.0), paths=paths))

    result = ex.run_seed(sol, cfg_path, seed=0, iteration=0)

    assert result.failure_kind is None, result.traceback_tail
    assert result.primary is not None
    outcome = verify_result(Path(result.artifact_dir), json.loads(
        (Path(result.artifact_dir) / "result.json").read_text()))
    assert outcome.status == Status.OK
