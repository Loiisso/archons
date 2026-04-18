from __future__ import annotations

from archons.config import AgentsConfig
from archons.llm.base import PolicyEngine
from archons.llm.ollama import OllamaPolicy
from archons.llm.openai_policy import OpenAIPolicy


def build_policy_engine(agents: AgentsConfig) -> PolicyEngine | None:
    if agents.backend == "deterministic":
        return None
    if agents.backend == "ollama":
        return OllamaPolicy(
            config=agents.ollama,
            fallback_strategy=agents.fallback_strategy,
        )
    if agents.backend == "openai":
        return OpenAIPolicy(
            config=agents.openai,
            fallback_strategy=agents.fallback_strategy,
        )
    raise ValueError(f"Unsupported backend: {agents.backend}")
