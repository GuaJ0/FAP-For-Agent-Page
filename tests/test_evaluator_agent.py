"""Real LLMEvaluatorAgent tests. Uses agent.coding's ScriptedClient (generic,
deterministic, no network) -- this agent reuses that module's LLM plumbing
rather than duplicating it, and these tests exercise exactly that path.
"""
import json

import pytest

from agent.agents import Decision
from agent.coding import ScriptedClient
from agent.evaluator import JudgeParseError, LLMEvaluatorAgent, parse_verdict_json
from agent.evaluator.agent import _extract_json_object
from agent.records import AggregateMetrics, Event, ResourceUsage, RunRecord, Status


def _record(iteration=1, primary=0.60, delta=0.01, hypothesis="try X", history_events=None):
    return RunRecord(
        iteration=iteration, parent_iteration=iteration - 1,
        timestamp="2026-01-01T00:00:00+00:00", hypothesis=hypothesis,
        diff_path=f"/runs/{iteration}/config.json", status=Status.SUCCESS,
        seeds=[], aggregate=AggregateMetrics(primary, 0.001, primary, primary, 2),
        delta_vs_current_best=delta, decision=None,
        events=history_events or [], resources=ResourceUsage(wall_s=1.0),
    )


def _agent(tmp_path, responses):
    return LLMEvaluatorAgent(llm=ScriptedClient(responses), usage_log_path=tmp_path / "usage.jsonl")


def test_judge_returns_accept_from_a_clean_json_response(tmp_path):
    agent = _agent(tmp_path, ['{"decision": "accept", "commentary": "beats baseline clearly"}'])

    verdict = agent.judge(_record(delta=0.02), [])

    assert verdict.decision == Decision.ACCEPT
    assert verdict.commentary == "beats baseline clearly"
    assert verdict.usage is not None and verdict.usage.tokens_in > 0


def test_judge_returns_revert(tmp_path):
    agent = _agent(tmp_path, ['{"decision": "revert", "commentary": "did not beat the incumbent"}'])

    verdict = agent.judge(_record(delta=-0.01), [])

    assert verdict.decision == Decision.REVERT


def test_judge_returns_abandon_when_the_model_says_so(tmp_path):
    """Unlike the fallback rule, a real parsed verdict CAN abandon."""
    agent = _agent(tmp_path, [
        '{"decision": "abandon", "commentary": "third variant of this idea, still far below baseline"}'
    ])

    verdict = agent.judge(_record(delta=-0.08), [])

    assert verdict.decision == Decision.ABANDON
    assert "third variant" in verdict.commentary


def test_judge_handles_a_markdown_fenced_response(tmp_path):
    agent = _agent(tmp_path, ['Here is my verdict:\n```json\n{"decision": "accept", "commentary": "ok"}\n```\n'])

    verdict = agent.judge(_record(delta=0.01), [])

    assert verdict.decision == Decision.ACCEPT


def test_judge_falls_back_when_response_has_no_json_at_all(tmp_path):
    agent = _agent(tmp_path, ["I think this looks pretty good, ship it."])

    verdict = agent.judge(_record(delta=0.02), [])   # would ACCEPT under the margin rule

    assert verdict.decision == Decision.ACCEPT
    assert "fallback" in verdict.commentary.lower()
    assert verdict.usage.tokens_in > 0, "the call still cost tokens even though it was unusable"


def test_judge_falls_back_on_an_unrecognised_decision_value(tmp_path):
    agent = _agent(tmp_path, ['{"decision": "maybe", "commentary": "unsure"}'])

    verdict = agent.judge(_record(delta=-0.01), [])   # would REVERT under the margin rule

    assert verdict.decision == Decision.REVERT
    assert "fallback" in verdict.commentary.lower()


def test_fallback_never_returns_abandon_even_for_a_very_bad_result(tmp_path):
    """A deterministic margin rule has no basis to call a whole research
    direction a dead end -- only a successfully parsed LLM verdict can."""
    agent = _agent(tmp_path, ["not json at all"])

    verdict = agent.judge(_record(delta=-0.50), [])

    assert verdict.decision == Decision.REVERT
    assert verdict.decision != Decision.ABANDON


def test_judge_raises_on_a_record_with_no_aggregate(tmp_path):
    agent = _agent(tmp_path, ['{"decision": "accept", "commentary": "x"}'])
    broken = _record()
    broken.aggregate = None

    with pytest.raises(ValueError):
        agent.judge(broken, [])


def test_usage_is_logged_to_the_usage_log_file(tmp_path):
    agent = _agent(tmp_path, ['{"decision": "accept", "commentary": "x"}'])

    agent.judge(_record(delta=0.01), [])

    lines = (tmp_path / "usage.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["agent"] == "evaluator"  # NOT "coding" -- see _EvaluatorUsageLog's docstring for why this matters
    assert row["purpose"] == "judge"
    assert row["tokens_in"] > 0


def test_usage_totals_matches_the_coding_agents_usage_log_shape(tmp_path):
    """Same dict shape as agent.coding.llm.UsageLog.totals(), so
    scripts/run_loop.py can print coding.usage.totals() and
    evaluator.usage.totals() symmetrically."""
    agent = _agent(tmp_path, [
        '{"decision": "accept", "commentary": "x"}',
        '{"decision": "revert", "commentary": "y"}',
    ])

    agent.judge(_record(delta=0.01), [])
    agent.judge(_record(delta=-0.01), [])

    totals = agent.usage.totals()
    assert totals["calls"] == 2
    assert totals["real_model_calls"] == 2   # ScriptedClient defaults is_real_model_call=True
    assert totals["tokens_in"] > 0
    assert set(totals) == {"calls", "real_model_calls", "tokens_in", "tokens_out", "cost_usd"}


def test_last_usage_attribute_is_populated_for_orchestrator_style_reads(tmp_path):
    agent = _agent(tmp_path, ['{"decision": "accept", "commentary": "x"}'])

    agent.judge(_record(delta=0.01), [])

    assert agent.last_usage["tokens_in"] > 0
    assert agent.last_usage["llm_calls"] == 1


def test_prompt_bounds_history_to_the_most_recent_items(tmp_path):
    agent = _agent(tmp_path, ['{"decision": "accept", "commentary": "x"}'])
    long_history = [_record(iteration=i, primary=0.5, delta=-0.01, hypothesis=f"idea {i}") for i in range(20)]

    agent.judge(_record(iteration=21, delta=0.01), long_history)

    _, user_prompt, _ = agent.llm.calls[0]
    for i in range(15):   # older entries must NOT appear
        assert f"idea {i}\"" not in user_prompt
    for i in range(15, 20):   # only the most recent ones should
        assert f"idea {i}\"" in user_prompt


def test_prompt_includes_prior_evaluator_commentary_from_history(tmp_path):
    agent = _agent(tmp_path, ['{"decision": "accept", "commentary": "x"}'])
    prior = _record(
        iteration=1, primary=0.5, delta=-0.02, hypothesis="idea 1",
        history_events=[Event(type="evaluator_commentary", detail="looked promising but underpowered", agent_action="evaluator")],
    )

    agent.judge(_record(iteration=2, delta=0.01), [prior])

    _, user_prompt, _ = agent.llm.calls[0]
    assert "looked promising but underpowered" in user_prompt


def test_extract_json_object_prefers_the_raw_text_when_it_parses_cleanly():
    assert _extract_json_object('{"decision": "accept"}') == {"decision": "accept"}


def test_extract_json_object_raises_judge_parse_error_on_total_garbage():
    with pytest.raises(JudgeParseError):
        _extract_json_object("no json here whatsoever")


def test_parse_verdict_json_rejects_a_non_string_decision():
    with pytest.raises(JudgeParseError):
        parse_verdict_json('{"decision": 42, "commentary": "x"}')
