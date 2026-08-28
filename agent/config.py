"""Every tunable threshold for the harness, in one place.

Nothing in agent/ or logging/ should hardcode eps, N, caps, timeouts, or seed
counts inline -- import from here so the competition's numbers are visible
and changeable in exactly one spot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Sentinel prefix train.py prints on stdout before its one line of hidden-test
# metrics, e.g.: TEST_METRICS: {"primary": 0.61, "gauc": 0.60, "ndcg5": 0.62}
# executor.py greps stdout for this prefix and redirects the payload straight
# to the quarantine file -- it never becomes part of a returned SeedMetrics.
TEST_METRICS_SENTINEL = "TEST_METRICS:"

# Keys that must never appear in any JSON payload handed to an agent. Used by
# the quarantine assertion in executor.py and by tests/test_no_test_leak.py.
FORBIDDEN_PAYLOAD_KEYS = frozenset({
    "test_primary", "test_gauc", "test_ndcg5",
    "primary_test", "gauc_test", "ndcg5_test",
    "test_metrics", "hidden_test",
})


@dataclass(frozen=True)
class ConvergenceConfig:
    """Competition stopping rule: whichever of the three fires first."""
    epsilon: float = 0.002          # min primary improvement to not count as "stalled"
    n_window: int = 3               # consecutive stalled iterations before stopping
    max_iterations: int = 50
    max_wall_s: float = 6 * 3600.0


@dataclass(frozen=True)
class RetryConfig:
    """Tier-1/tier-2 abandonment policy for the orchestrator."""
    max_fix_attempts: int = 3               # tier 1: fixes on the same idea before abandoning it
    max_consecutive_abandonments: int = 2   # tier 2: back-to-back abandonments before halting for a human
    idea_time_backstop_s: float = 45 * 60.0  # abandon an idea past this wall time regardless of attempt count


@dataclass(frozen=True)
class ExecutorConfig:
    per_run_timeout_s: float = 900.0   # hard ceiling on a single subprocess run (separate from idea_time_backstop_s)
    traceback_tail_lines: int = 40      # stderr lines kept for the Coding agent on failure
    oom_returncodes: tuple[int, ...] = (-9, 137)  # SIGKILL / OOM-killer exit codes on POSIX


@dataclass(frozen=True)
class SeedingConfig:
    """Adaptive seeding: drop to min_seeds if projected cost would blow the wall-clock budget."""
    max_seeds: int = 3
    min_seeds: int = 1


@dataclass(frozen=True)
class Paths:
    logs_dir: Path = Path("logs")
    runs_jsonl: Path = Path("logs/runs.jsonl")
    quarantine_dir: Path = Path("logs/quarantine")
    test_metrics_jsonl: Path = Path("logs/quarantine/test_metrics.jsonl")
    orchestrator_state: Path = Path("logs/orchestrator_state.json")
    registry_json: Path = Path("logs/registry.json")
    # Persistent per-(iteration, seed) run directory: executor.py writes
    # --out result.json here (not a tempdir -- see executor.py's docstring)
    # so a checkpoint train.py drops alongside it survives past the
    # subprocess call and can be pointed to by the registry.
    artifacts_dir: Path = Path("logs/artifacts")


@dataclass(frozen=True)
class Config:
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    seeding: SeedingConfig = field(default_factory=SeedingConfig)
    paths: Paths = field(default_factory=Paths)


DEFAULT_CONFIG = Config()
