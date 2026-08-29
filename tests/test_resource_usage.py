"""RunRecord.resources actually carries LLM tokens now.

ResourceUsage has had tokens_in/tokens_out since the schema was written and
nothing populated them, so every RunRecord reported zero tokens regardless of
what the run spent. The CodingAgent reports usage on the Diff it returns and
orchestrator.py folds it in.

The compatibility constraint: agents that don't track usage (FakeCodingAgent,
and anything else satisfying the Protocol) must produce byte-identical records
to before -- so tests/test_orchestrator.py keeps passing unmodified.
"""
import json
from pathlib import Path

import pytest

from agent.agents import AgentUsage, Diff, Idea
from agent.coding import LLMCodingAgent, ScriptedClient
from agent.records import Status
from conftest import make_orchestrator, make_test_config

REPO_ROOT = Path(__file__).resolve().parent.parent
RANKING_TEMPLATE = (REPO_ROOT / "agent" / "coding" / "templates" / "train_ranking.py").read_text()


def _fenced(src):
    return f"```python\n{src}\n```\n"


class _UsageCodingAgent:
    """Minimal CodingAgent that reports fixed usage, so orchestrator-level
    tests don't need a real LLM or real data."""

    def __init__(self, inner, usage: AgentUsage):
        self.inner = inner
        self.usage = usage

    def implement(self, idea, feedback):
        d = self.inner.implement(idea, feedback)
        return Diff(config_path=d.config_path, solution_dir=d.solution_dir,
                    patch_path=d.patch_path, usage=self.usage)


# ---------------------------------------------------------------------------
# The CodingAgent reports it.
# ---------------------------------------------------------------------------

def test_the_coding_agent_reports_usage_on_the_diff(tmp_path):
    client = ScriptedClient([_fenced(RANKING_TEMPLATE)])
    agent = LLMCodingAgent(
        work_dir=tmp_path / "s", data_dir=str(tmp_path), llm=client,
        usage_log_path=tmp_path / "u.jsonl", run_smoke_test=False,
    )

    diff = agent.implement(Idea("an idea", None), None)

    assert diff.usage is not None
    assert diff.usage.tokens_in > 0 and diff.usage.tokens_out > 0
    assert diff.usage.tokens_in == agent.last_usage["tokens_in"]
    assert diff.usage.cost_usd == agent.last_usage["cost_usd"]


def test_reported_usage_includes_the_inner_repair_cycles(tmp_path):
    """A repaired attempt cost more than one call, and the RunRecord should
    say so -- the repairs are part of what that iteration cost."""
    client = ScriptedClient([_fenced("import torch\n"), _fenced(RANKING_TEMPLATE)])
    agent = LLMCodingAgent(
        work_dir=tmp_path / "s", data_dir=str(tmp_path), llm=client,
        usage_log_path=tmp_path / "u.jsonl", run_smoke_test=False,
    )

    diff = agent.implement(Idea("an idea", None), None)

    rows = [json.loads(l) for l in (tmp_path / "u.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert diff.usage.tokens_in == sum(r["tokens_in"] for r in rows)
    assert diff.usage.tokens_out == sum(r["tokens_out"] for r in rows)


# ---------------------------------------------------------------------------
# The orchestrator records it -- on both the success and the failure path.
# ---------------------------------------------------------------------------

def test_successful_iteration_records_tokens(tmp_path):
    cfg = make_test_config(tmp_path)
    from agent.agents import FakeCodingAgent

    inner = FakeCodingAgent(tmp_path / "sols", [{"mode": "normal", "sleep_s": 0.0}])
    coding = _UsageCodingAgent(inner, AgentUsage(tokens_in=1200, tokens_out=800, cost_usd=0.0095))
    orc = make_orchestrator(tmp_path, cfg, outcomes=[], coding=coding)

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    assert record.status == Status.SUCCESS
    assert record.resources.tokens_in == 1200
    assert record.resources.tokens_out == 800
    assert record.resources.wall_s > 0


def test_failed_iteration_also_records_tokens(tmp_path):
    """The case the docs called out: a failed attempt still costs tokens, and
    is often the most expensive, since it is the one that burned repairs."""
    cfg = make_test_config(tmp_path)
    from agent.agents import FakeCodingAgent

    inner = FakeCodingAgent(tmp_path / "sols", [{"mode": "crash", "sleep_s": 0.0}])
    coding = _UsageCodingAgent(inner, AgentUsage(tokens_in=5000, tokens_out=3000, cost_usd=0.0362))
    orc = make_orchestrator(tmp_path, cfg, outcomes=[], coding=coding)

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    assert record.status == Status.FAILED
    assert record.resources.tokens_in == 5000
    assert record.resources.tokens_out == 3000


def test_cost_usd_lands_in_the_audit_trail_as_an_event(tmp_path):
    """cost_usd has no ResourceUsage field on purpose (a dollar figure derived
    from a mutable price table would go stale in an append-only log), so it is
    recorded as an event instead -- visible in runs.jsonl, no schema change."""
    cfg = make_test_config(tmp_path)
    from agent.agents import FakeCodingAgent

    inner = FakeCodingAgent(tmp_path / "sols", [{"mode": "normal", "sleep_s": 0.0}])
    coding = _UsageCodingAgent(inner, AgentUsage(tokens_in=1200, tokens_out=800, cost_usd=0.0095))
    orc = make_orchestrator(tmp_path, cfg, outcomes=[], coding=coding)

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    usage_events = [e for e in record.events if e.type == "coding_usage"]
    assert len(usage_events) == 1
    assert "cost_usd=0.009500" in usage_events[0].detail
    assert usage_events[0].agent_action == "coding"


def test_usage_survives_the_jsonl_round_trip(tmp_path):
    cfg = make_test_config(tmp_path)
    from agent.agents import FakeCodingAgent

    inner = FakeCodingAgent(tmp_path / "sols", [{"mode": "normal", "sleep_s": 0.0}])
    coding = _UsageCodingAgent(inner, AgentUsage(tokens_in=7, tokens_out=11, cost_usd=0.001))
    orc = make_orchestrator(tmp_path, cfg, outcomes=[], coding=coding)

    orc._step(orc.run_log.read_all())

    raw = json.loads(cfg.paths.runs_jsonl.read_text().strip())
    assert raw["resources"]["tokens_in"] == 7
    assert raw["resources"]["tokens_out"] == 11
    assert orc.run_log.read_all()[-1].resources.tokens_out == 11


# ---------------------------------------------------------------------------
# Backwards compatibility: agents that don't track usage are unaffected.
# ---------------------------------------------------------------------------

def test_an_agent_that_does_not_track_usage_produces_the_old_record(tmp_path):
    """FakeCodingAgent leaves usage=None. Records must be exactly as before,
    which is what keeps tests/test_orchestrator.py passing unmodified."""
    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0}])

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    assert record.resources.tokens_in == 0
    assert record.resources.tokens_out == 0
    assert record.resources.gpu_s == 0.0
    assert [e.type for e in record.events] == ["eval_finished"]  # no usage event


def test_zero_usage_emits_no_event(tmp_path):
    """An agent that reports AgentUsage() with nothing in it shouldn't clutter
    the trail either."""
    cfg = make_test_config(tmp_path)
    from agent.agents import FakeCodingAgent

    inner = FakeCodingAgent(tmp_path / "sols", [{"mode": "normal", "sleep_s": 0.0}])
    coding = _UsageCodingAgent(inner, AgentUsage())
    orc = make_orchestrator(tmp_path, cfg, outcomes=[], coding=coding)

    orc._step(orc.run_log.read_all())

    assert not [e for e in orc.run_log.read_all()[-1].events if e.type == "coding_usage"]


def test_diff_usage_defaults_to_none():
    assert Diff(config_path="c", solution_dir="d").usage is None
