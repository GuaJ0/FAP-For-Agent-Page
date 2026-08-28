"""Iteration 0 must actually reproduce the published FM baseline.

Slow and opt-in: trains the real FM on the real 1.14M-row split (~20s/seed).
Run with:

    RUN_SLOW_TESTS=1 KUAIRAND_PATH=/path/to/KuaiRand-Pure/data pytest -m slow

The fast part of this file (contract shape, config parsing) runs always -- if
train.py stops honouring the executor's CLI, that should fail in the default
suite, not only in the opt-in one.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from slow_helpers import kuairand_path, requires_data

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = REPO_ROOT / "solution" / "train.py"
CONFIG = REPO_ROOT / "solution" / "config.yaml"

PUBLISHED = json.loads((REPO_ROOT / "harness" / "baseline_scores.json").read_text())
FM = PUBLISHED["scores"]["fm_official"]
SEED_STD = FM["std_over_5_seeds"]["test_primary"]  # 0.0008


# --------------------------------------------------------------------------
# Fast: the contract, without touching data.
# --------------------------------------------------------------------------

def test_config_yaml_parses_identically_with_and_without_pyyaml():
    """train.py falls back to a stdlib flat-YAML reader when PyYAML is absent.
    The two readers must agree, or a solution dir silently trains on different
    hyperparameters depending on what's installed."""
    import importlib.util

    import yaml

    # Load train.py by path under a private name rather than importing it as
    # `train`: its module-level import fallback appends harness/ to sys.path
    # (correct when it runs as a script in a solution dir, pollution here),
    # and `train`/`evaluate`/`data` are names other tests should not inherit.
    before_path = list(sys.path)
    spec = importlib.util.spec_from_file_location("_solution_train", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        text = CONFIG.read_text()
        assert yaml.safe_load(text) == module._parse_flat_yaml(text)
    finally:
        # Restore sys.path only. Do NOT prune sys.modules: exec_module pulls in
        # numpy's submodules, and evicting those leaves numpy half-initialised
        # for every later test in the session.
        sys.path[:] = before_path


def test_train_py_rejects_a_loss_it_does_not_implement(tmp_path):
    """Iteration 0 is pointwise-only. Asked for a ranking loss it must fail
    loudly rather than silently training logloss and reporting the result as
    if the hypothesis had been tested."""
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"loss": "bpr", "data_dir": str(tmp_path)}))

    proc = subprocess.run(
        [sys.executable, str(TRAIN_PY), "--config", str(cfg), "--seed", "0",
         "--out", str(tmp_path / "result.json")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )

    assert proc.returncode != 0
    assert "logloss" in (proc.stderr + proc.stdout)


def test_missing_data_dir_fails_with_an_actionable_message(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"data_dir": str(tmp_path / "nope")}))

    proc = subprocess.run(
        [sys.executable, str(TRAIN_PY), "--config", str(cfg), "--seed", "0",
         "--out", str(tmp_path / "result.json")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )

    assert proc.returncode != 0
    assert "KUAIRAND_PATH" in proc.stderr


# --------------------------------------------------------------------------
# Slow: the actual number.
# --------------------------------------------------------------------------

@pytest.mark.slow
@requires_data
def test_iteration_zero_reproduces_the_published_fm_baseline(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({
        "data_dir": str(kuairand_path()),
        "loss": "logloss", "k": 16, "lr": 0.001, "l2": 1e-6,
        "epochs": 40, "batch_size": 8192, "patience": 4,
    }))
    out = tmp_path / "art" / "result.json"

    proc = subprocess.run(
        [sys.executable, str(TRAIN_PY), "--config", str(cfg), "--seed", "0", "--out", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr

    result = json.loads(out.read_text())

    # Validation is what the agent loop actually optimises against.
    assert result["primary"] == pytest.approx(FM["valid"]["primary"], abs=3 * SEED_STD)
    assert result["gauc"] == pytest.approx(FM["valid"]["GAUC"], abs=3 * SEED_STD)
    assert result["ndcg5"] == pytest.approx(FM["valid"]["nDCG@5"], abs=3 * SEED_STD)

    # Hidden test metrics go to stdout for the executor to quarantine, and are
    # deliberately absent from result.json.
    assert not any(k in result for k in ("test_primary", "test_metrics", "hidden_test"))
    test_line = [l for l in proc.stdout.splitlines() if l.startswith("TEST_METRICS:")]
    assert len(test_line) == 1
    test_metrics = json.loads(test_line[0][len("TEST_METRICS:"):])
    assert test_metrics["primary"] == pytest.approx(FM["test"]["primary"], abs=3 * SEED_STD)

    # Artifacts the executor and the registry rely on.
    art = out.parent
    assert (art / "val_predictions.npz").exists()
    assert (art / "checkpoint.npz").exists()
