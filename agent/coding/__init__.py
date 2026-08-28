"""Real CodingAgent implementation (agent.agents.CodingAgent Protocol)."""
from agent.coding.agent import LLMCodingAgent, extract_code, static_check
from agent.coding.llm import (
    LLMClient,
    LLMResponse,
    OpenAIClient,
    OpenAIClientError,
    ScriptedClient,
    TemplateLibraryClient,
    UsageLog,
    default_client,
    estimate_cost,
)

__all__ = [
    "LLMCodingAgent", "extract_code", "static_check",
    "LLMClient", "LLMResponse", "OpenAIClient", "OpenAIClientError",
    "ScriptedClient", "TemplateLibraryClient", "UsageLog",
    "default_client", "estimate_cost",
]
