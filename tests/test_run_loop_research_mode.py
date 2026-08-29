"""Run-loop wiring tests for offline and optional LLM-backed Research.

Every pipeline dependency is replaced with an in-process fake. These tests do
not read KuaiRand data, construct the OpenAI SDK client, or make network calls.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

from scripts import run_loop


class _Usage:
    def __init__(self, agent: str):
        self.agent = agent

    def totals(self):
        return {
            "agent": self.agent,
            "calls": 1,
            "real_model_calls": 1,
            "tokens_in": 10,
            "tokens_out": 5,
            "cost_usd": 0.001,
        }


def _install_pipeline_fakes(monkeypatch):
    captured = {
        "openai_calls": [],
        "template_calls": 0,
    }
    live_client = object()
    offline_client = object()

    def make_openai_client(*, model):
        captured["openai_calls"].append(model)
        return live_client

    def make_template_client():
        captured["template_calls"] += 1
        return offline_client

    class OfflineResearch:
        def __init__(self, *, convergence):
            self.convergence = convergence
            captured["research"] = self

    class LiveResearch:
        def __init__(self, *, llm, usage_log_path, convergence):
            self.llm = llm
            self.usage_log_path = Path(usage_log_path)
            self.convergence = convergence
            self.usage = _Usage("research")
            captured["research"] = self

    class Coding:
        def __init__(self, **kwargs):
            self.llm = kwargs["llm"]
            self.usage = _Usage("coding")
            captured["coding"] = self

    class LiveEvaluator:
        def __init__(self, *, llm, usage_log_path):
            self.llm = llm
            self.usage_log_path = Path(usage_log_path)
            self.usage = _Usage("evaluator")
            captured["evaluator"] = self

    class OfflineEvaluator:
        def __init__(self):
            captured["evaluator"] = self

    class InertComponent:
        def __init__(self, *args, **kwargs):
            pass

    class Orchestrator:
        def __init__(self, **kwargs):
            captured["orchestrator"] = self
            self.kwargs = kwargs

        def run(self):
            return []

    monkeypatch.setattr(run_loop, "load_dotenv", lambda path: None)
    monkeypatch.setattr(run_loop, "OpenAIClient", make_openai_client)
    monkeypatch.setattr(run_loop, "TemplateLibraryClient", make_template_client)
    monkeypatch.setattr(run_loop, "OfflineResearchAgent", OfflineResearch)
    monkeypatch.setattr(run_loop, "LLMResearchAgent", LiveResearch)
    monkeypatch.setattr(run_loop, "LLMCodingAgent", Coding)
    monkeypatch.setattr(run_loop, "LLMEvaluatorAgent", LiveEvaluator)
    monkeypatch.setattr(run_loop, "FakeEvaluatorAgent", OfflineEvaluator)
    monkeypatch.setattr(run_loop, "Executor", InertComponent)
    monkeypatch.setattr(run_loop, "RunLog", InertComponent)
    monkeypatch.setattr(run_loop, "CheckpointRegistry", InertComponent)
    monkeypatch.setattr(run_loop, "StateStore", InertComponent)
    monkeypatch.setattr(run_loop, "Orchestrator", Orchestrator)

    captured.update({
        "live_client": live_client,
        "offline_client": offline_client,
        "OfflineResearch": OfflineResearch,
        "LiveResearch": LiveResearch,
        "LiveEvaluator": LiveEvaluator,
        "OfflineEvaluator": OfflineEvaluator,
    })
    return captured


def _run_main(monkeypatch, tmp_path, *extra_args):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    root = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "run_loop.py",
        "--data-dir", str(data_dir),
        "--root", str(root),
        "--skip-baseline",
        *extra_args,
    ])
    return run_loop.main(), root


def test_default_selects_offline_research_with_live_coding_and_evaluator(monkeypatch, tmp_path):
    captured = _install_pipeline_fakes(monkeypatch)

    result, _ = _run_main(monkeypatch, tmp_path)

    assert result == 0
    assert isinstance(captured["research"], captured["OfflineResearch"])
    assert captured["openai_calls"] == ["gpt-5"]
    assert captured["template_calls"] == 0
    assert captured["coding"].llm is captured["live_client"]
    assert isinstance(captured["evaluator"], captured["LiveEvaluator"])
    assert captured["evaluator"].llm is captured["live_client"]
    assert captured["orchestrator"].kwargs["research"] is captured["research"]


def test_live_research_shares_client_convergence_and_reports_usage(
    monkeypatch, tmp_path, capsys,
):
    captured = _install_pipeline_fakes(monkeypatch)

    result, root = _run_main(monkeypatch, tmp_path, "--live-research")

    assert result == 0
    research = captured["research"]
    orchestrator_args = captured["orchestrator"].kwargs
    assert isinstance(research, captured["LiveResearch"])
    assert research.llm is captured["live_client"]
    assert captured["coding"].llm is research.llm
    assert captured["evaluator"].llm is research.llm
    assert research.convergence is orchestrator_args["cfg"].convergence
    assert research.usage_log_path == root / "logs" / "research_agent_usage.jsonl"
    assert orchestrator_args["research"] is research

    output = capsys.readouterr().out
    assert f"research usage:   {root / 'logs' / 'research_agent_usage.jsonl'}" in output
    assert "research totals:" in output
    assert "'agent': 'research'" in output
    assert "coding totals:" in output
    assert "evaluator totals:" in output


def test_offline_preserves_offline_research_coding_and_evaluator(monkeypatch, tmp_path):
    captured = _install_pipeline_fakes(monkeypatch)

    result, _ = _run_main(monkeypatch, tmp_path, "--offline")

    assert result == 0
    assert isinstance(captured["research"], captured["OfflineResearch"])
    assert captured["openai_calls"] == []
    assert captured["template_calls"] == 1
    assert captured["coding"].llm is captured["offline_client"]
    assert isinstance(captured["evaluator"], captured["OfflineEvaluator"])


def test_offline_and_live_research_conflict_before_client_construction(
    monkeypatch, capsys,
):
    captured = _install_pipeline_fakes(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_loop.py", "--offline", "--live-research"],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_loop.main()

    assert exc_info.value.code == 2
    assert "--offline and --live-research are incompatible" in capsys.readouterr().err
    assert captured["openai_calls"] == []
    assert captured["template_calls"] == 0
    assert "research" not in captured
    assert "coding" not in captured


def test_run_loop_does_not_wire_fake_research_agent():
    source = inspect.getsource(run_loop)

    assert "FakeResearchAgent" not in source
    assert "from agent.research import LLMResearchAgent, OfflineResearchAgent" in source
