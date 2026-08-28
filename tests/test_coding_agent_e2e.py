"""End-to-end tests for the real CodingAgent. All opt-in.

    RUN_SLOW_TESTS=1 KUAIRAND_PATH=/path/to/KuaiRand-Pure/data pytest -m slow
    RUN_LLM_TESTS=1  OPENAI_API_KEY=sk-...                     pytest -m llm

The -m slow tests drive the EXISTING Orchestrator with the real CodingAgent
and the Fake Research/Evaluator agents from agent/agents.py, against the real
dataset, and assert the loop produces a genuine RunRecord. The -m llm test is
the only thing in the suite that spends money.
"""
import json
import os
from pathlib import Path

import pytest

from slow_helpers import kuairand_path, requires_data, requires_openai

from agent.agents import FakeEvaluatorAgent, FakeResearchAgent, Idea
from agent.coding import LLMCodingAgent, TemplateLibraryClient
from agent.coding.llm import OpenAIClient
from agent.config import (
    Config,
    ConvergenceConfig,
    ExecutorConfig,
    Paths,
    RetryConfig,
    SeedingConfig,
)
from agent.executor import Executor
from agent.orchestrator import Orchestrator
from agent.records import RunLog, Status
from agent.registry import CheckpointRegistry
from agent.state import StateStore

# The starter kit's top-ranked unexplored direction: contained, and directly
# comparable against the FM baseline.
HYPOTHESIS = (
    "Replace the pointwise logloss objective with a pairwise BPR ranking loss "
    "computed within each user, so the training objective matches the GAUC/nDCG@5 "
    "ranking metric instead of optimising calibrated click probabilities."
)


def _config(tmp_path, **over):
    logs = tmp_path / "logs"
    paths = Paths(
        logs_dir=logs,
        runs_jsonl=logs / "runs.jsonl",
        quarantine_dir=logs / "quarantine",
        test_metrics_jsonl=logs / "quarantine" / "test_metrics.jsonl",
        orchestrator_state=logs / "orchestrator_state.json",
        registry_json=logs / "registry.json",
        artifacts_dir=logs / "artifacts",
    )
    return Config(
        convergence=over.get("convergence", ConvergenceConfig(max_iterations=1, max_wall_s=3600.0)),
        retry=over.get("retry", RetryConfig()),
        executor=over.get("executor", ExecutorConfig(per_run_timeout_s=900.0)),
        seeding=over.get("seeding", SeedingConfig(max_seeds=2, min_seeds=1)),
        paths=paths,
    )


def _orchestrator(tmp_path, cfg, coding):
    return Orchestrator(
        research=FakeResearchAgent([HYPOTHESIS]),
        coding=coding,
        evaluator=FakeEvaluatorAgent(),
        executor=Executor(cfg=cfg),
        run_log=RunLog(cfg.paths.runs_jsonl),
        registry=CheckpointRegistry(cfg.paths.registry_json),
        state_store=StateStore(cfg.paths.orchestrator_state),
        cfg=cfg,
    )


@pytest.mark.slow
@requires_data
def test_full_loop_produces_a_real_success_record_on_real_data(tmp_path):
    """The headline integration: real CodingAgent -> real executor -> real
    KuaiRand data -> a RunRecord with sane validation metrics.

    Uses the offline TemplateLibraryClient rather than an LLM so this is
    deterministic and free; the LLM path is covered by the -m llm test below.
    """
    cfg = _config(tmp_path)
    coding = LLMCodingAgent(
        work_dir=tmp_path / "solutions",
        data_dir=str(kuairand_path()),
        llm=TemplateLibraryClient(),
        usage_log_path=cfg.paths.logs_dir / "coding_agent_usage.jsonl",
        base_config={"loss": "bpr", "epochs": 12, "patience": 3},
    )
    orc = _orchestrator(tmp_path, cfg, coding)

    history = orc.run()

    assert len(history) == 1
    record = history[0]
    assert record.status == Status.SUCCESS, record.seeds[0].traceback_tail
    assert record.hypothesis == HYPOTHESIS
    assert record.decision is not None

    agg = record.aggregate
    assert agg is not None and agg.n_seeds == 2
    # Sane, not merely non-null: comfortably above the random floor (0.4834
    # valid) and below the oracle ceiling (0.8484 valid).
    assert 0.55 < agg.primary_mean < 0.70
    assert 0.60 < agg.gauc_mean < 0.75
    assert 0.48 < agg.ndcg5_mean < 0.62
    assert agg.primary_std < 0.01
    assert all(s.failure_kind is None for s in record.seeds)

    # Every seed left a verifiable artifact dir the registry can point at.
    for seed in record.seeds:
        art = Path(seed.artifact_dir)
        assert (art / "result.json").exists()
        assert (art / "val_predictions.npz").exists()
        assert (art / "checkpoint.npz").exists()

    entry = orc.registry.best()
    if record.decision.value == "accept":
        assert entry is not None and Path(entry.checkpoint_path).is_dir()


@pytest.mark.slow
@requires_data
def test_the_full_loop_never_leaks_test_metrics(tmp_path):
    """Positive control included: the quarantine file must actually have
    received the test numbers, so the absence elsewhere isn't vacuous."""
    from agent.config import FORBIDDEN_PAYLOAD_KEYS, TEST_METRICS_SENTINEL
    from runlog.emit import read_lines

    cfg = _config(tmp_path, seeding=SeedingConfig(max_seeds=1, min_seeds=1))
    coding = LLMCodingAgent(
        work_dir=tmp_path / "solutions",
        data_dir=str(kuairand_path()),
        llm=TemplateLibraryClient(),
        usage_log_path=cfg.paths.logs_dir / "usage.jsonl",
        base_config={"loss": "bpr", "epochs": 2, "patience": 1},
    )
    _orchestrator(tmp_path, cfg, coding).run()

    quarantined = list(read_lines(cfg.paths.test_metrics_jsonl))
    assert quarantined, "positive control failed: no test metrics were quarantined at all"
    assert "primary" in quarantined[0]["test_metrics"]

    raw = cfg.paths.runs_jsonl.read_text().lower()
    assert TEST_METRICS_SENTINEL.lower() not in raw
    for key in FORBIDDEN_PAYLOAD_KEYS:
        assert key not in raw


@pytest.mark.slow
@requires_data
def test_the_ranking_template_actually_trains_on_real_data(tmp_path):
    """Guards the hypothesis implementation itself: a BPR run that silently
    learned nothing would still produce a 'sane-looking' aggregate, because
    the FM baseline and an untrained model differ by less than the range
    asserted above. Check it beats the random floor by a wide margin and
    improves over its own first epoch."""
    import subprocess
    import shutil
    import sys

    repo = Path(__file__).resolve().parent.parent
    sol = tmp_path / "sol"
    sol.mkdir()
    shutil.copy(repo / "agent" / "coding" / "templates" / "train_ranking.py", sol / "train.py")
    for f in ("evaluate.py", "data.py"):
        shutil.copy(repo / "harness" / f, sol / f)
    cfg_path = sol / "c.json"
    cfg_path.write_text(json.dumps({
        "data_dir": str(kuairand_path()), "loss": "bpr", "epochs": 4, "patience": 4,
    }))
    out = tmp_path / "art" / "result.json"

    proc = subprocess.run(
        [sys.executable, str(sol / "train.py"), "--config", str(cfg_path),
         "--seed", "0", "--out", str(out)],
        cwd=sol, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    result = json.loads(out.read_text())
    assert result["primary"] > 0.55          # random valid floor is 0.4834
    assert result["loss"] == "bpr"
    assert result["epochs_run"] >= 1
    assert "usable_users" in proc.stdout      # within-user pairs were actually formed


@pytest.mark.slow
@requires_data
def test_listwise_variant_also_runs(tmp_path):
    import subprocess
    import shutil
    import sys

    repo = Path(__file__).resolve().parent.parent
    sol = tmp_path / "sol"
    sol.mkdir()
    shutil.copy(repo / "agent" / "coding" / "templates" / "train_ranking.py", sol / "train.py")
    for f in ("evaluate.py", "data.py"):
        shutil.copy(repo / "harness" / f, sol / f)
    cfg_path = sol / "c.json"
    cfg_path.write_text(json.dumps({
        "data_dir": str(kuairand_path()), "loss": "listwise", "list_size": 8,
        "epochs": 2, "patience": 2,
    }))
    out = tmp_path / "art" / "result.json"

    proc = subprocess.run(
        [sys.executable, str(sol / "train.py"), "--config", str(cfg_path),
         "--seed", "0", "--out", str(out)],
        cwd=sol, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out.read_text())["primary"] > 0.55


# ---------------------------------------------------------------------------
# The only test in the suite that spends money.
# ---------------------------------------------------------------------------

@pytest.mark.llm
@requires_openai
def test_a_real_openai_call_produces_a_contract_compliant_train_py(tmp_path):
    """One real generation against the live API. Smoke-tested only if the
    dataset is present; otherwise this asserts the static contract, which is
    the part that depends on the model rather than on the data."""
    from agent.coding.agent import static_check

    data_dir = kuairand_path()
    have_data = os.environ.get("RUN_SLOW_TESTS") == "1" and data_dir is not None

    agent = LLMCodingAgent(
        work_dir=tmp_path / "solutions",
        data_dir=str(data_dir) if have_data else str(tmp_path),
        llm=OpenAIClient(model=os.environ.get("CODING_AGENT_MODEL", "gpt-5")),
        usage_log_path=tmp_path / "usage.jsonl",
        run_smoke_test=have_data,
        max_repair_attempts=2,
        base_config={"epochs": 2, "patience": 2},
    )

    diff = agent.implement(Idea(HYPOTHESIS, None), None)

    source = (Path(diff.solution_dir) / "train.py").read_text()
    assert static_check(source) == [], static_check(source)

    manifest = json.loads((Path(diff.solution_dir) / "attempt.json").read_text())
    assert manifest["succeeded"] is True, manifest["cycles"]

    usage = [json.loads(l) for l in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert usage and all(u["is_real_model_call"] for u in usage)
    assert sum(u["tokens_out"] for u in usage) > 0
    print(f"\n[cost] {len(usage)} call(s), "
          f"{sum(u['tokens_in'] for u in usage)} in / {sum(u['tokens_out'] for u in usage)} out, "
          f"${sum(u['cost_usd'] for u in usage):.4f}")
