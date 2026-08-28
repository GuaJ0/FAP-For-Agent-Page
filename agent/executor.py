"""Runs one solution (train.py + config.yaml) in a subprocess.

Two output channels come back from a run, and they must never merge:

  1. --out result.json: validation-split metrics (primary/gauc/ndcg5/epochs_run).
     These become a SeedMetrics and flow into the agent-facing RunRecord.
  2. stdout's `TEST_METRICS: {...}` line: hidden-test-split metrics. These are
     written straight to logs/quarantine/test_metrics.jsonl and never touch
     the value this module returns -- run_seed()'s return type has no field
     that could hold them, and assert_no_forbidden_keys() double-checks that
     before anything is handed back to the caller.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agent.config import Config, DEFAULT_CONFIG, FORBIDDEN_PAYLOAD_KEYS, TEST_METRICS_SENTINEL
from agent.records import AggregateMetrics, FailureKind, SeedMetrics
from runlog.emit import append_line

REQUIRED_RESULT_KEYS = {"primary", "gauc", "ndcg5", "epochs_run"}


class QuarantineLeakError(RuntimeError):
    """A test-metric key was about to enter an agent-facing payload."""


def assert_no_forbidden_keys(payload: Any) -> None:
    """Recursively check a JSON-able structure for any key in
    FORBIDDEN_PAYLOAD_KEYS. This is the structural backstop behind the
    "never merge" rule above -- called on every value executor.py returns."""
    if isinstance(payload, dict):
        for k, v in payload.items():
            if isinstance(k, str) and k.lower() in FORBIDDEN_PAYLOAD_KEYS:
                raise QuarantineLeakError(f"forbidden key {k!r} in agent-facing payload")
            assert_no_forbidden_keys(v)
    elif isinstance(payload, (list, tuple)):
        for v in payload:
            assert_no_forbidden_keys(v)


def _extract_test_metrics_line(stdout: str) -> Optional[dict]:
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(TEST_METRICS_SENTINEL):
            try:
                return json.loads(line[len(TEST_METRICS_SENTINEL):].strip())
            except json.JSONDecodeError:
                return None
    return None


def _strip_test_metrics_lines(text: str) -> str:
    """text with any TEST_METRICS line removed -- applied before anything
    (stdout or stderr) is kept as a traceback_tail, since that tail is
    agent-facing."""
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith(TEST_METRICS_SENTINEL)
    )


def _tail(text: str, n: int) -> str:
    return "\n".join(text.splitlines()[-n:])


@dataclass
class Executor:
    cfg: Config = DEFAULT_CONFIG

    def run_seed(
        self,
        solution_dir: Path,
        config_path: Path,
        seed: int,
        iteration: int,
        timeout_s: Optional[float] = None,
    ) -> SeedMetrics:
        """Always returns a SeedMetrics; failures are encoded via
        failure_kind rather than raised, so callers can retry/classify
        uniformly without a try/except around every call site."""
        timeout_s = timeout_s if timeout_s is not None else self.cfg.executor.per_run_timeout_s
        train_py = solution_dir / "train.py"
        cmd = [
            sys.executable, str(train_py),
            "--config", str(config_path),
            "--seed", str(seed),
        ]

        # Persistent, not a tempdir: this directory (and whatever result.json
        # sits in it) must survive past this call so the checkpoint registry
        # can point a "validation-best" lookup at it later. Convention: a
        # real train.py should drop its checkpoint file(s) alongside
        # result.json in this same directory.
        out_dir = self.cfg.paths.artifacts_dir / f"iter_{iteration}" / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "result.json"

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd + ["--out", str(out_path)],
                cwd=solution_dir, capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            wall_s = time.time() - t0
            self._quarantine_if_present(e.stdout or "", iteration, seed)
            return SeedMetrics(
                seed=seed, primary=None, gauc=None, ndcg5=None, epochs_run=None,
                wall_s=wall_s, failure_kind=FailureKind.TIMEOUT,
                traceback_tail=f"timed out after {timeout_s}s",
            )

        wall_s = time.time() - t0
        self._quarantine_if_present(proc.stdout, iteration, seed)

        if proc.returncode != 0:
            kind = (
                FailureKind.OOM if proc.returncode in self.cfg.executor.oom_returncodes
                else FailureKind.CRASH
            )
            tail = _tail(_strip_test_metrics_lines(proc.stderr), self.cfg.executor.traceback_tail_lines)
            return SeedMetrics(
                seed=seed, primary=None, gauc=None, ndcg5=None, epochs_run=None,
                wall_s=wall_s, failure_kind=kind, traceback_tail=tail,
            )

        metrics = self._parse_result(out_path)
        if metrics is None:
            tail = _tail(_strip_test_metrics_lines(proc.stdout), self.cfg.executor.traceback_tail_lines)
            return SeedMetrics(
                seed=seed, primary=None, gauc=None, ndcg5=None, epochs_run=None,
                wall_s=wall_s, failure_kind=FailureKind.BAD_OUTPUT,
                traceback_tail=tail or "result.json missing, malformed, or missing required keys",
            )

        result = SeedMetrics(
            seed=seed, primary=metrics["primary"], gauc=metrics["gauc"],
            ndcg5=metrics["ndcg5"], epochs_run=metrics["epochs_run"], wall_s=wall_s,
            artifact_dir=str(out_dir),
        )
        assert_no_forbidden_keys(result.to_json())
        return result

    def _quarantine_if_present(self, stdout: str, iteration: int, seed: int) -> None:
        test_metrics = _extract_test_metrics_line(stdout)
        if test_metrics is not None:
            append_line(self.cfg.paths.test_metrics_jsonl, {
                "iteration": iteration, "seed": seed, "test_metrics": test_metrics,
            })

    @staticmethod
    def _parse_result(out_path: Path) -> Optional[dict]:
        if not out_path.exists():
            return None
        try:
            d = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            return None
        if not REQUIRED_RESULT_KEYS.issubset(d.keys()):
            return None
        for k in ("primary", "gauc", "ndcg5"):
            v = d[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v or v in (float("inf"), float("-inf")):
                return None
        if not isinstance(d["epochs_run"], int) or isinstance(d["epochs_run"], bool):
            return None
        return d

    def run_seeds(
        self,
        solution_dir: Path,
        config_path: Path,
        iteration: int,
        seeds: list[int],
        timeout_s: Optional[float] = None,
    ) -> tuple[list[SeedMetrics], Optional[AggregateMetrics]]:
        """Run every seed, then aggregate over the ones that succeeded. An
        empty `ok` set (every seed failed) yields aggregate=None -- the
        orchestrator treats that as a FAILED iteration."""
        results = [self.run_seed(solution_dir, config_path, s, iteration, timeout_s) for s in seeds]
        ok = [r for r in results if r.failure_kind is None]
        if not ok:
            return results, None
        agg = AggregateMetrics(
            primary_mean=statistics.mean(r.primary for r in ok),
            primary_std=statistics.pstdev([r.primary for r in ok]) if len(ok) > 1 else 0.0,
            gauc_mean=statistics.mean(r.gauc for r in ok),
            ndcg5_mean=statistics.mean(r.ndcg5 for r in ok),
            n_seeds=len(ok),
        )
        return results, agg
