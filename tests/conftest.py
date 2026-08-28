import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.agents import FakeCodingAgent, FakeEvaluatorAgent, FakeResearchAgent  # noqa: E402
from agent.config import (  # noqa: E402
    Config,
    ConvergenceConfig,
    ExecutorConfig,
    Paths,
    RetryConfig,
    SeedingConfig,
)
from agent.executor import Executor  # noqa: E402
from agent.orchestrator import Orchestrator  # noqa: E402
from agent.records import RunLog  # noqa: E402
from agent.registry import CheckpointRegistry  # noqa: E402
from agent.state import StateStore  # noqa: E402


def make_test_config(tmp_path, **overrides) -> Config:
    logs_dir = tmp_path / "logs"
    paths = Paths(
        logs_dir=logs_dir,
        runs_jsonl=logs_dir / "runs.jsonl",
        quarantine_dir=logs_dir / "quarantine",
        test_metrics_jsonl=logs_dir / "quarantine" / "test_metrics.jsonl",
        orchestrator_state=logs_dir / "orchestrator_state.json",
        registry_json=logs_dir / "registry.json",
        artifacts_dir=logs_dir / "artifacts",
    )
    return Config(
        convergence=overrides.get("convergence", ConvergenceConfig()),
        retry=overrides.get("retry", RetryConfig()),
        executor=overrides.get("executor", ExecutorConfig(per_run_timeout_s=5.0)),
        seeding=overrides.get("seeding", SeedingConfig(max_seeds=1, min_seeds=1)),
        paths=paths,
    )


def make_orchestrator(tmp_path, cfg, outcomes, hypotheses=("try X",), research=None, coding=None, evaluator=None):
    work_dir = tmp_path / "solutions"
    research = research or FakeResearchAgent(list(hypotheses))
    coding = coding or FakeCodingAgent(work_dir, list(outcomes))
    evaluator = evaluator or FakeEvaluatorAgent()
    executor = Executor(cfg=cfg)
    run_log = RunLog(cfg.paths.runs_jsonl)
    registry = CheckpointRegistry(cfg.paths.registry_json)
    state_store = StateStore(cfg.paths.orchestrator_state)
    return Orchestrator(research, coding, evaluator, executor, run_log, registry, state_store, cfg=cfg)
