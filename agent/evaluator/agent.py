"""Real implementation of the EvaluatorAgent Protocol (agent/agents.py)."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agent.agents import AgentUsage, Verdict
from agent.coding import LLMClient, LLMResponse
from agent.evaluator.prompts import SYSTEM_PROMPT, build_judge_prompt
from agent.records import Decision, RunRecord
from runlog.emit import append_line, read_lines

VALID_DECISIONS = {"accept": Decision.ACCEPT, "revert": Decision.REVERT, "abandon": Decision.ABANDON}

# ```json ... ``` or ``` ... ``` around the verdict, in case the model fences
# it despite the system prompt asking it not to.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class JudgeParseError(ValueError):
    """The LLM's response couldn't be parsed into a usable verdict."""


def _extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of the one JSON object a judge response should
    contain: try the raw text, then a fenced block, then the first
    '{' .. last '}' span. Raises JudgeParseError (never a bare JSONDecodeError)
    on total failure, so the caller has one exception type to catch."""
    candidates = [text.strip()]
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise JudgeParseError(f"no JSON object found in response: {text[:300]!r}")


def parse_verdict_json(text: str) -> tuple[Decision, str]:
    """Parses a judge response into (Decision, commentary). Raises
    JudgeParseError on anything unusable -- an unrecognised decision value,
    a missing key, or no parseable JSON at all. Never guesses."""
    obj = _extract_json_object(text)
    raw_decision = obj.get("decision")
    if not isinstance(raw_decision, str) or raw_decision.strip().lower() not in VALID_DECISIONS:
        raise JudgeParseError(f"missing or unrecognised 'decision' field: {raw_decision!r}")
    commentary = obj.get("commentary", "")
    if not isinstance(commentary, str):
        commentary = str(commentary)
    return VALID_DECISIONS[raw_decision.strip().lower()], commentary.strip()


@dataclass
class _EvaluatorUsageLog:
    """Append-only per-call usage log, same on-disk shape as
    agent.coding.llm.UsageLog's entries. Not that class reused directly:
    its record() hardcodes "agent": "coding", so every Evaluator call would
    be mislabeled. Written via runlog.emit (harness-owned, generic JSONL
    append) instead."""
    path: Path

    def record(self, response: LLMResponse, *, purpose: str, idea: str) -> None:
        append_line(self.path, {
            "timestamp": time.time(),
            "agent": "evaluator",
            "purpose": purpose,
            "model": response.model,
            "is_real_model_call": response.is_real_model_call,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "cost_usd": round(response.cost_usd, 6),
            "idea": idea[:200],
        })

    def totals(self) -> dict[str, Any]:
        """Same shape as agent.coding.llm.UsageLog.totals(), so callers (e.g.
        scripts/run_loop.py) can print both agents' usage symmetrically:
        coding.usage.totals() and evaluator.usage.totals()."""
        rows = list(read_lines(self.path))
        real = [r for r in rows if r.get("is_real_model_call")]
        return {
            "calls": len(rows),
            "real_model_calls": len(real),
            "tokens_in": sum(r.get("tokens_in", 0) for r in rows),
            "tokens_out": sum(r.get("tokens_out", 0) for r in rows),
            "cost_usd": round(sum(r.get("cost_usd", 0.0) for r in rows), 6),
        }


@dataclass
class LLMEvaluatorAgent:
    """LLM-backed EvaluatorAgent. Falls back to a deterministic margin rule
    (identical to agent.agents.FakeEvaluatorAgent's) whenever the LLM call
    fails outright or its response can't be parsed into a usable verdict --
    a malformed judge response must never crash the run or silently corrupt
    an ACCEPT/REVERT decision, since that decision drives the checkpoint
    registry and the convergence rule for the rest of the loop."""

    llm: LLMClient
    usage_log_path: Path
    margin: float = 0.0
    last_usage: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.usage = _EvaluatorUsageLog(self.usage_log_path)

    def judge(self, record: RunRecord, history: list[RunRecord]) -> Verdict:
        if record.aggregate is None:
            raise ValueError("judge() called on a record with no aggregate metrics")

        current_best_primary = self._current_best_primary(record, history)
        user_prompt = build_judge_prompt(record, history, current_best_primary)

        response = self.llm.complete(SYSTEM_PROMPT, user_prompt, purpose="judge")
        self.usage.record(response, purpose="judge", idea=record.hypothesis)
        self.last_usage = {
            "tokens_in": response.tokens_in, "tokens_out": response.tokens_out,
            "cost_usd": response.cost_usd, "llm_calls": 1,
            "real_model_calls": 1 if response.is_real_model_call else 0,
        }
        usage = AgentUsage(
            tokens_in=response.tokens_in, tokens_out=response.tokens_out, cost_usd=response.cost_usd,
        )

        try:
            decision, commentary = parse_verdict_json(response.text)
        except JudgeParseError as e:
            # The call still cost real tokens even though the answer is
            # unusable -- usage above is kept, only the decision falls back.
            decision, commentary = self._fallback_decision(record)
            commentary = f"[fallback: judge response unparseable ({e}); used margin rule] {commentary}"

        return Verdict(decision=decision, commentary=commentary, usage=usage)

    def _fallback_decision(self, record: RunRecord) -> tuple[Decision, str]:
        """Same rule as FakeEvaluatorAgent: accept if this beats the current
        best by more than margin, else revert. Never falls back to ABANDON --
        a deterministic margin comparison has no basis to judge a whole
        research direction a dead end; only a real, successfully-parsed
        LLM verdict can return that."""
        delta = record.delta_vs_current_best
        if delta is not None and delta > self.margin:
            return Decision.ACCEPT, "accepted: beat the current best by more than the margin."
        return Decision.REVERT, "reverted: did not beat the current best by more than the margin."

    @staticmethod
    def _current_best_primary(record: RunRecord, history: list[RunRecord]) -> Optional[float]:
        """Reconstructed from delta_vs_current_best rather than passed in
        separately, since the orchestrator already computed it into `record`
        before calling judge() and this agent has no registry access of its
        own -- one source of truth for "what was the incumbent," not two."""
        if record.delta_vs_current_best is None:
            return None
        return record.aggregate.primary_mean - record.delta_vs_current_best
