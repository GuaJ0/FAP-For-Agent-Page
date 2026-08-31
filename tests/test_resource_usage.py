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


# ---------------------------------------------------------------------------
# The Research agent's tokens, too.
#
# The same leak, found later on the other side: an LLM-backed ResearchAgent
# spends real tokens producing the hypothesis, and those appeared only in
# research_agent_usage.jsonl -- never in any RunRecord. Reconciling a real
# campaign's runs.jsonl against its usage logs showed the deficit was exactly
# the Research log, to the token.
# ---------------------------------------------------------------------------

class _UsageResearchAgent:
    """ResearchAgent that reports fixed usage and counts how often it was
    asked, so a test can tell "proposed twice" from "retried one proposal"."""

    def __init__(self, usage: AgentUsage, hypotheses=("try X",)):
        self.usage = usage
        self._hypotheses = list(hypotheses)
        self.propose_calls = 0
        self.last_usage = {}

    def propose(self, history):
        h = self._hypotheses[self.propose_calls % len(self._hypotheses)]
        self.propose_calls += 1
        self.last_usage = {
            "tokens_in": self.usage.tokens_in,
            "tokens_out": self.usage.tokens_out,
            "cost_usd": self.usage.cost_usd,
        }
        parent = history[-1].iteration if history else None
        return Idea(hypothesis=h, parent_iteration=parent, usage=self.usage)


def test_research_tokens_land_in_the_record(tmp_path):
    cfg = make_test_config(tmp_path)
    research = _UsageResearchAgent(AgentUsage(tokens_in=900, tokens_out=400, cost_usd=0.005125))
    orc = make_orchestrator(
        tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0}], research=research,
    )

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    assert record.status == Status.SUCCESS
    assert record.resources.tokens_in == 900
    assert record.resources.tokens_out == 400


def test_research_and_coding_tokens_are_summed(tmp_path):
    """Both agents bill the same iteration; the record carries the total, and
    the two events keep the split attributable."""
    cfg = make_test_config(tmp_path)
    from agent.agents import FakeCodingAgent

    inner = FakeCodingAgent(tmp_path / "sols", [{"mode": "normal", "sleep_s": 0.0}])
    coding = _UsageCodingAgent(inner, AgentUsage(tokens_in=1200, tokens_out=800, cost_usd=0.0095))
    research = _UsageResearchAgent(AgentUsage(tokens_in=900, tokens_out=400, cost_usd=0.005125))
    orc = make_orchestrator(tmp_path, cfg, outcomes=[], coding=coding, research=research)

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    assert record.resources.tokens_in == 1200 + 900
    assert record.resources.tokens_out == 800 + 400
    by_type = {e.type: e for e in record.events}
    assert "cost_usd=0.009500" in by_type["coding_usage"].detail
    assert "cost_usd=0.005125" in by_type["research_usage"].detail
    assert by_type["research_usage"].agent_action == "research"


def test_research_tokens_are_charged_once_across_a_retry(tmp_path):
    """The double-count guard.

    One propose() produces one idea, which a failure can turn into several
    RunRecords. Charging idea.usage to every record would bill that single
    call two or three times over -- and would do so silently, since the total
    would still look plausible. Only the first iteration for an idea carries
    the Research tokens; the retry carries none.
    """
    cfg = make_test_config(tmp_path)
    research = _UsageResearchAgent(AgentUsage(tokens_in=900, tokens_out=400, cost_usd=0.005125))
    orc = make_orchestrator(
        tmp_path, cfg,
        outcomes=[{"mode": "crash", "sleep_s": 0.0}, {"mode": "normal", "sleep_s": 0.0}],
        research=research,
    )

    orc._step(orc.run_log.read_all())   # fails; the idea stays in flight
    orc._step(orc.run_log.read_all())   # retries the SAME idea

    first, second = orc.run_log.read_all()
    assert research.propose_calls == 1, "the retry must reuse the in-flight idea"
    assert first.status == Status.FAILED
    assert second.status == Status.SUCCESS_AFTER_RETRY
    assert first.resources.tokens_in == 900
    assert second.resources.tokens_in == 0
    assert not [e for e in second.events if e.type == "research_usage"]
    total_in = sum(r.resources.tokens_in for r in (first, second))
    assert total_in == 900, "one propose() call, billed once"


def test_a_failed_proposal_still_bills_its_tokens(tmp_path):
    """A propose() that raised after two model calls cost exactly what one
    that returned would have. No Idea comes back to carry the usage, so it is
    read off the agent -- the research_failed record is the only one that will
    ever be written for that spend."""
    cfg = make_test_config(tmp_path)

    class _FailingResearchAgent:
        def __init__(self):
            self.last_usage = {}

        def propose(self, history):
            self.last_usage = {"tokens_in": 2200, "tokens_out": 1300, "cost_usd": 0.01575}
            raise RuntimeError("citation validation failed")

    orc = make_orchestrator(tmp_path, cfg, outcomes=[], research=_FailingResearchAgent())

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    assert record.status == Status.FAILED
    assert [e.type for e in record.events][0] == "research_failed"
    assert record.resources.tokens_in == 2200
    assert record.resources.tokens_out == 1300
    usage_event = [e for e in record.events if e.type == "research_usage"]
    assert "cost_usd=0.015750" in usage_event[0].detail


def test_a_failed_proposal_that_spent_nothing_records_nothing(tmp_path):
    """An agent with no last_usage at all (OfflineResearchAgent) must produce
    the byte-identical research_failed record it did before."""
    cfg = make_test_config(tmp_path)

    class _BareFailingResearchAgent:
        def propose(self, history):
            raise RuntimeError("no proposal")

    orc = make_orchestrator(tmp_path, cfg, outcomes=[], research=_BareFailingResearchAgent())

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    assert record.resources.tokens_in == 0 and record.resources.tokens_out == 0
    assert [e.type for e in record.events] == ["research_failed"]


def test_offline_research_produces_the_old_record(tmp_path):
    """FakeResearchAgent (and OfflineResearchAgent) leave Idea.usage None.
    "Not measured" must stay distinguishable from "measured as zero"."""
    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0}])

    orc._step(orc.run_log.read_all())

    record = orc.run_log.read_all()[-1]
    assert record.resources.tokens_in == 0
    assert [e.type for e in record.events] == ["eval_finished"]


def test_idea_usage_defaults_to_none():
    assert Idea("h", None).usage is None


def test_an_in_flight_idea_survives_the_state_round_trip_with_usage_intact():
    """OrchestratorState persists the in-flight Idea via asdict(), which
    flattens the nested AgentUsage into a plain dict. Without rehydrating it,
    a resumed run would carry an Idea whose .usage is a dict -- something that
    looks like usage and has no attributes."""
    from agent.state import OrchestratorState

    state = OrchestratorState()
    state.set_current_idea(Idea("h", 3, AgentUsage(tokens_in=900, tokens_out=400, cost_usd=0.005)))

    restored = OrchestratorState.from_json(state.to_json()).get_current_idea()

    assert restored == Idea("h", 3, AgentUsage(900, 400, 0.005))
    assert restored.usage.tokens_in == 900


def test_a_state_file_written_before_idea_had_usage_still_resumes():
    """Old orchestrator_state.json has no `usage` key at all; the field's
    default applies and the checkpoint resumes unchanged."""
    from agent.state import OrchestratorState

    state = OrchestratorState(current_idea={"hypothesis": "h", "parent_iteration": None})

    assert state.get_current_idea() == Idea("h", None)


def test_the_real_research_agent_reconciles_against_its_own_usage_log(tmp_path):
    """The end-to-end property the accounting is for, with no stubs on the
    Research side: after a run, the tokens in runs.jsonl equal the tokens in
    research_agent_usage.jsonl.

    This is exactly the check that failed on runs/run2 before the fix -- its
    runs.jsonl was short by 15799/10282 tokens, which was, to the token, the
    whole of its research_agent_usage.jsonl.
    """
    import json as _json

    from agent.records import Decision
    from test_research_agent import _agent, _proposal, _record

    research, _ = _agent(tmp_path, [_json.dumps(_proposal(parent=3))])
    cfg = make_test_config(tmp_path)
    orc = make_orchestrator(
        tmp_path, cfg, outcomes=[{"mode": "normal", "sleep_s": 0.0}], research=research,
    )
    # An accepted parent for the proposal to build on. Carries no tokens, so it
    # cannot perturb the reconciliation.
    orc.run_log.append(_record(3, 0.62, Decision.ACCEPT))

    orc._step(orc.run_log.read_all())

    logged = sum(
        _json.loads(line)["tokens_in"]
        for line in (tmp_path / "research_usage.jsonl").read_text().splitlines()
    )
    recorded = sum(r.resources.tokens_in for r in orc.run_log.read_all())
    assert logged > 0
    assert recorded == logged
