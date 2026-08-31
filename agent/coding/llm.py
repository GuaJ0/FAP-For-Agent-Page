"""LLM client layer for the Coding agent, plus per-call token/cost accounting.

Three implementations behind one Protocol:

  OpenAIClient          real, billable OpenAI API calls.
  ScriptedClient        returns a fixed queue of responses. Unit tests use
                        this so the suite stays deterministic and free.
  TemplateLibraryClient NOT an LLM. Serves pre-written implementations keyed
                        off the hypothesis text, so the whole pipeline can be
                        exercised end-to-end without an API key. Always
                        labelled as such in the usage log -- never counted or
                        reported as model output.

COST ACCOUNTING -- KNOWN GAP
----------------------------
RunRecord.resources (ResourceUsage.tokens_in / tokens_out) exists and nothing
populates it. Filling it in means threading usage from the CodingAgent up
through Orchestrator._handle_successful_run / _handle_failed_run, which means
editing orchestrator.py -- not authorised this round. So usage is written to
its own JSONL (logs/coding_agent_usage.jsonl) and RunRecord.resources stays at
its wall_s-only default. See UsageLog.as_resource_usage() for the shim that
should make the eventual wiring a couple of lines.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional, Protocol

from runlog.emit import append_line, read_lines

DEFAULT_MODEL = os.environ.get("CODING_AGENT_MODEL", "gpt-5")

# USD per 1M tokens, (input, output).
#
# !! VERIFY BEFORE TRUSTING THE DOLLAR COLUMN !! These are list prices as of
# writing and OpenAI changes them. Token counts come straight from the API
# response and are always exact; only the derived `cost_usd` depends on this
# table. Override without a code change:
#     OPENAI_PRICE_IN_PER_MTOK=1.25 OPENAI_PRICE_OUT_PER_MTOK=10.00
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}
UNKNOWN_MODEL_PRICE = (0.0, 0.0)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    is_real_model_call: bool = True   # False for TemplateLibraryClient
    raw_finish_reason: Optional[str] = None


class LLMClient(Protocol):
    def complete(self, system: str, user: str, *, purpose: str = "") -> LLMResponse:
        ...


def price_for(model: str) -> tuple[float, float]:
    env_in = os.environ.get("OPENAI_PRICE_IN_PER_MTOK")
    env_out = os.environ.get("OPENAI_PRICE_OUT_PER_MTOK")
    if env_in and env_out:
        try:
            return float(env_in), float(env_out)
        except ValueError:
            pass
    if model in PRICING_USD_PER_MTOK:
        return PRICING_USD_PER_MTOK[model]
    # Longest prefix wins, or "gpt-5-mini" would be priced as "gpt-5" -- an
    # 8x overstatement, since dict order would reach the shorter key first.
    for known in sorted(PRICING_USD_PER_MTOK, key=len, reverse=True):
        if model.startswith(known + "-"):
            return PRICING_USD_PER_MTOK[known]
    return UNKNOWN_MODEL_PRICE


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = price_for(model)
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000.0


# ---------------------------------------------------------------------------
# Usage log
# ---------------------------------------------------------------------------

@dataclass
class UsageLog:
    """Append-only record of every LLM call the Coding agent makes.

    Separate file, not RunRecord.resources -- see the module docstring.
    """
    path: Path

    def record(self, response: LLMResponse, *, purpose: str, idea: str, attempt: int) -> None:
        append_line(self.path, {
            "timestamp": time.time(),
            "agent": "coding",
            "purpose": purpose,             # "generate" | "repair"
            "attempt": attempt,
            "model": response.model,
            "is_real_model_call": response.is_real_model_call,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "cost_usd": round(response.cost_usd, 6),
            # Truncated: the log is for accounting, and full hypotheses are
            # already in logs/runs.jsonl.
            "idea": idea[:200],
        })

    def totals(self) -> dict[str, Any]:
        rows = list(read_lines(self.path))
        real = [r for r in rows if r.get("is_real_model_call")]
        return {
            "calls": len(rows),
            "real_model_calls": len(real),
            "tokens_in": sum(r.get("tokens_in", 0) for r in rows),
            "tokens_out": sum(r.get("tokens_out", 0) for r in rows),
            "cost_usd": round(sum(r.get("cost_usd", 0.0) for r in rows), 6),
        }

    def as_resource_usage(self, wall_s: float):
        """The shim for the gap in the module docstring.

        When someone is authorised to edit orchestrator.py, this is what
        RunRecord.resources should be built from -- ideally scoped to one
        iteration rather than the whole run, which needs the CodingAgent to
        report usage per implement() call (see LLMCodingAgent.last_usage).
        """
        from agent.records import ResourceUsage

        t = self.totals()
        return ResourceUsage(wall_s=wall_s, tokens_in=t["tokens_in"], tokens_out=t["tokens_out"])


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

class OpenAIClientError(RuntimeError):
    pass


@dataclass
class OpenAIClient:
    """Direct OpenAI API calls. The key is read from the environment (or a
    .env file) and never logged, never written into a solution dir, and never
    included in a prompt."""
    model: str = DEFAULT_MODEL
    api_key: Optional[str] = None
    timeout_s: float = 300.0
    max_retries: int = 3
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise OpenAIClientError(
                "OPENAI_API_KEY is not set. Put it in .env (already gitignored) "
                "or export it. See .env.example."
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise OpenAIClientError("the `openai` package is not installed: pip install openai") from e
        self._client = OpenAI(api_key=key, timeout=self.timeout_s, max_retries=self.max_retries)

    def complete(self, system: str, user: str, *, purpose: str = "") -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0
        return LLMResponse(
            text=choice.message.content or "",
            model=resp.model or self.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=estimate_cost(resp.model or self.model, tokens_in, tokens_out),
            raw_finish_reason=getattr(choice, "finish_reason", None),
        )

    def complete_structured(
        self,
        system: str,
        user: str,
        *,
        schema_name: str,
        json_schema: dict[str, Any],
        purpose: str = "",
    ) -> LLMResponse:
        """Request strict JSON for Research without changing ``complete`` callers."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": json_schema,
                },
            },
        )
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0
        return LLMResponse(
            text=choice.message.content or "",
            model=resp.model or self.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=estimate_cost(resp.model or self.model, tokens_in, tokens_out),
            raw_finish_reason=getattr(choice, "finish_reason", None),
        )


@dataclass
class ScriptedClient:
    """Deterministic stand-in for tests: pops one canned response per call."""
    responses: list[str]
    model: str = "scripted"
    calls: list[tuple[str, str, str]] = field(default_factory=list)

    def complete(self, system: str, user: str, *, purpose: str = "") -> LLMResponse:
        self.calls.append((system, user, purpose))
        if not self.responses:
            raise AssertionError(
                f"ScriptedClient ran out of responses on call {len(self.calls)} "
                f"(purpose={purpose!r}). The agent asked for more repairs than the "
                "test scripted -- that's usually the finding, not a fixture bug."
            )
        text = self.responses.pop(0)
        # Plausible non-zero counts so cost accounting is exercised, derived
        # from length rather than random so the suite stays deterministic.
        tokens_in = max(1, len(system) + len(user)) // 4
        tokens_out = max(1, len(text)) // 4
        return LLMResponse(
            text=text, model=self.model, tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=estimate_cost(self.model, tokens_in, tokens_out),
        )


@dataclass
class TemplateLibraryClient:
    """NOT a language model.

    Serves hand-written implementations from agent/coding/templates/, picked by
    keyword-matching the hypothesis. Exists so the orchestrator, executor,
    verification and a real training run can all be exercised end-to-end
    against real KuaiRand data with no API key and no spend -- which is also
    the only way to get a genuine SUCCESS RunRecord before the key lands.

    Every response is flagged is_real_model_call=False, and it ignores
    `feedback` entirely (it has no repair capability). If it can't match the
    hypothesis it says so in plain text rather than guessing, and the agent
    surfaces that as a failed attempt.
    """
    templates_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "templates")
    model: str = "template-library(not-an-llm)"
    calls: list[str] = field(default_factory=list)

    # ClassVar, not a field: a bare annotation here would make this a
    # constructor argument on the dataclass.
    ROUTES: ClassVar[tuple[tuple[tuple[str, ...], str], ...]] = (
        (("listwise", "softmax"), "train_ranking.py"),
        (("pairwise",), "train_ranking.py"),
        (("bpr",), "train_ranking.py"),
        (("ranking loss",), "train_ranking.py"),
        (("logloss",), "train_ranking.py"),
    )

    # The prompt embeds solution/ideas.md, which lists every unexplored
    # direction by name -- so keyword-matching the whole prompt matches
    # "bpr"/"pairwise"/"listwise" for *any* hypothesis. Match the hypothesis
    # section alone.
    HYPOTHESIS_HEADERS: ClassVar[tuple[str, ...]] = (
        "## hypothesis to implement", "## hypothesis being implemented",
    )

    @classmethod
    def _hypothesis_of(cls, user: str) -> str:
        lowered = user.lower()
        for header in cls.HYPOTHESIS_HEADERS:
            start = lowered.find(header)
            if start == -1:
                continue
            body = user[start + len(header):]
            end = body.find("\n## ")
            return (body[:end] if end != -1 else body).strip().lower()
        return lowered  # not one of our prompts; match against the whole thing

    def complete(self, system: str, user: str, *, purpose: str = "") -> LLMResponse:
        self.calls.append(purpose)
        lowered = self._hypothesis_of(user)
        for keywords, filename in self.ROUTES:
            if any(k in lowered for k in keywords):
                path = self.templates_dir / filename
                if not path.exists():
                    break
                return self._respond(f"```python\n{path.read_text()}\n```")
        return self._respond(
            "NO_TEMPLATE: the offline template library has no implementation matching "
            "this hypothesis. Configure a real LLM client (OPENAI_API_KEY) to generate "
            "one, or add a template to agent/coding/templates/."
        )

    def _respond(self, text: str) -> LLMResponse:
        return LLMResponse(
            text=text, model=self.model, tokens_in=0, tokens_out=0,
            cost_usd=0.0, is_real_model_call=False,
        )


def default_client(model: Optional[str] = None, allow_offline: bool = True) -> LLMClient:
    """OpenAIClient when a key and the SDK are available, else the offline
    template library. Never silently downgrades without saying so."""
    try:
        return OpenAIClient(model=model or DEFAULT_MODEL)
    except OpenAIClientError as e:
        if not allow_offline:
            raise
        print(f"[coding-agent] falling back to the offline template library: {e}", flush=True)
        return TemplateLibraryClient()
