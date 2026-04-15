import httpx
import pytest

from archons.config import OllamaConfig
from archons.core.models import AgentState, RecognitionSnapshot
from archons.llm.ollama import OllamaError, OllamaPolicy


def test_ollama_policy_uses_live_response_and_parses_wrapped_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b-instruct"}]})
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={
                    "message": {
                        "content": "Result follows: {\"action\": \"D\", \"confidence\": 0.81, \"reasoning_summary\": \"Opponent has a hostile prior record.\"}"
                    }
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    policy = OllamaPolicy(
        config=OllamaConfig(model="qwen2.5:3b-instruct"),
        fallback_strategy="tit_for_tat",
        transport=httpx.MockTransport(handler),
    )
    policy.ensure_available()

    result = policy.choose_action(
        agent=AgentState("agent_1", "lineage_1", "tit_for_tat"),
        opponent=AgentState("agent_2", "lineage_2", "always_defect"),
        prior_rounds=tuple(),
        side="left",
        recognition=RecognitionSnapshot(None, None, 0.0, "No prior memory."),
    )

    assert result.action == "D"
    assert result.backend == "ollama"
    assert result.used_fallback is False
    assert result.confidence == pytest.approx(0.81)
    assert "policy_prompt=" in result.prompt_text


def test_ollama_policy_raises_when_model_is_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "some-other-model"}]})

    policy = OllamaPolicy(
        config=OllamaConfig(model="qwen2.5:3b-instruct", require_model=True),
        fallback_strategy="tit_for_tat",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(OllamaError):
        policy.ensure_available()


def test_ollama_policy_can_fallback_when_enabled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": "not-json"}})
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b-instruct"}]})

    policy = OllamaPolicy(
        config=OllamaConfig(model="qwen2.5:3b-instruct", fallback_on_error=True),
        fallback_strategy="always_defect",
        transport=httpx.MockTransport(handler),
    )

    result = policy.choose_action(
        agent=AgentState("agent_1", "lineage_1", "tit_for_tat"),
        opponent=AgentState("agent_2", "lineage_2", "always_cooperate"),
        prior_rounds=tuple(),
        side="left",
        recognition=RecognitionSnapshot(None, None, 0.0, "No prior memory."),
    )

    assert result.action == "D"
    assert result.used_fallback is True
    assert result.backend == "deterministic-fallback"
    assert result.error_message is not None