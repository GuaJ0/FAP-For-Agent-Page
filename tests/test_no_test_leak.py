"""Explicit proof that hidden-test-split metrics never reach an
agent-facing payload: not the RunRecord passed to the Evaluator, not the
history passed to Research/the Evaluator, not the JSONL on disk, and not
what a real ResearchAgent/CodingAgent/EvaluatorAgent would have been shown.

The forbidden-key scan (agent.config.FORBIDDEN_PAYLOAD_KEYS) is checked
positively -- via a spy on all three agent Protocols and a direct scan of
logs/runs.jsonl -- alongside a *negative* control that proves the quarantine
file really did receive the test metrics, so the assertions above aren't
trivially true because nothing was ever produced.
"""
import json

from agent.agents import Diff, Idea
from agent.config import FORBIDDEN_PAYLOAD_KEYS, TEST_METRICS_SENTINEL
from conftest import make_orchestrator, make_test_config
from runlog.emit import read_lines


class SpyResearch:
    def __init__(self, inner):
        self.inner = inner
        self.payloads: list[str] = []

    def propose(self, history) -> Idea:
        self.payloads.append(json.dumps([r.to_json() for r in history]))
        return self.inner.propose(history)


class SpyCoding:
    def __init__(self, inner):
        self.inner = inner
        self.payloads: list[str] = []

    def implement(self, idea, feedback) -> Diff:
        self.payloads.append(json.dumps({"idea": idea.__dict__, "feedback": feedback}))
        return self.inner.implement(idea, feedback)


class SpyEvaluator:
    def __init__(self, inner):
        self.inner = inner
        self.payloads: list[str] = []

    def judge(self, record, history):
        self.payloads.append(json.dumps({"record": record.to_json(), "history": [r.to_json() for r in history]}))
        return self.inner.judge(record, history)


def _no_forbidden_substrings(text: str) -> bool:
    if TEST_METRICS_SENTINEL in text:
        return False
    lowered = text.lower()
    return not any(k in lowered for k in FORBIDDEN_PAYLOAD_KEYS)


def test_agent_facing_payloads_never_contain_test_metrics(tmp_path):
    from agent.agents import FakeCodingAgent, FakeEvaluatorAgent, FakeResearchAgent

    cfg = make_test_config(tmp_path)
    work_dir = tmp_path / "solutions"
    spy_research = SpyResearch(FakeResearchAgent(["idea A"]))
    spy_coding = SpyCoding(FakeCodingAgent(work_dir, [{"mode": "normal", "sleep_s": 0.0}] * 3))
    spy_evaluator = SpyEvaluator(FakeEvaluatorAgent())

    orc = make_orchestrator(
        tmp_path, cfg, outcomes=[],  # unused: spy_coding already has its own outcomes queue
        research=spy_research, coding=spy_coding, evaluator=spy_evaluator,
    )

    for _ in range(3):
        orc._step(orc.run_log.read_all())

    # Positive control: the quarantine file actually received test metrics,
    # so the absence below is because of quarantining, not because nothing
    # was ever produced.
    quarantined = list(read_lines(cfg.paths.test_metrics_jsonl))
    assert len(quarantined) == 3
    assert all("primary" in q["test_metrics"] for q in quarantined)

    all_payloads = spy_research.payloads + spy_coding.payloads + spy_evaluator.payloads
    assert all_payloads, "spies were never called -- test would pass vacuously"
    for payload in all_payloads:
        assert _no_forbidden_substrings(payload), f"leak found in agent-facing payload: {payload}"

    # And the on-disk agent-facing log itself.
    raw_runs_jsonl = cfg.paths.runs_jsonl.read_text()
    assert _no_forbidden_substrings(raw_runs_jsonl)


def test_forbidden_key_scan_catches_a_deliberately_injected_leak():
    from agent.executor import QuarantineLeakError, assert_no_forbidden_keys
    import pytest

    with pytest.raises(QuarantineLeakError):
        assert_no_forbidden_keys({"seeds": [{"primary": 0.6, "test_primary": 0.7}]})


def test_the_cross_run_findings_ledger_never_contains_test_metrics(tmp_path):
    """agent/research/findings.jsonl is a new persistent path into a Research
    prompt, and unlike runs.jsonl it deliberately outlives the run -- so a leak
    here would be permanent and would be replayed into every future run.

    Same structure as the test above: a positive control proves the quarantine
    really did receive test metrics, so the absence in the ledger is because of
    the boundary rather than because nothing was produced.
    """
    from agent.agents import FakeCodingAgent, FakeEvaluatorAgent, FakeResearchAgent
    from agent.research.findings import FindingsLedger

    handoff = (
        "[RESEARCH_PROPOSAL v1]\nID: RP-LEAK\nTITLE: t\nPARENT ITERATION: 0\n"
        "\nHYPOTHESIS:\nsome direction\n"
    )
    cfg = make_test_config(tmp_path)
    ledger_path = tmp_path / "findings.jsonl"
    orc = make_orchestrator(
        tmp_path, cfg,
        outcomes=[{"mode": "normal", "sleep_s": 0.0, "mean": 0.6, "std": 0.0}],
        hypotheses=(handoff,),
    )
    orc.findings = FindingsLedger(ledger_path)

    orc._step(orc.run_log.read_all())

    # Positive control: the run really did produce hidden-test metrics.
    quarantined = list(read_lines(cfg.paths.test_metrics_jsonl))
    assert quarantined, "no test metrics were quarantined -- test would be vacuous"

    # And the ledger really was written, so the scan below isn't trivially true.
    assert ledger_path.exists()
    raw = ledger_path.read_text()
    assert raw.strip(), "ledger is empty -- test would be vacuous"
    assert _no_forbidden_substrings(raw)
