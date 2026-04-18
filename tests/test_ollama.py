import asyncio

import httpx
import pytest

from archons.config import OllamaConfig
from archons.core.models import RoundRecord
from archons.core.models import AgentState, RecognitionSnapshot
from archons.llm.base import StructuredDecision
from archons.llm.ollama import OllamaError, OllamaPolicy


@pytest.mark.asyncio
async def test_ollama_policy_uses_live_response_and_parses_wrapped_json() -> None:
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
    await policy.ensure_available()

    result = await policy.choose_action(
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


@pytest.mark.asyncio
async def test_ollama_policy_raises_when_model_is_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "some-other-model"}]})

    policy = OllamaPolicy(
        config=OllamaConfig(model="qwen2.5:3b-instruct", require_model=True),
        fallback_strategy="tit_for_tat",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(OllamaError):
        await policy.ensure_available()


@pytest.mark.asyncio
async def test_ollama_policy_can_fallback_when_enabled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": "not-json"}})
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b-instruct"}]})

    policy = OllamaPolicy(
        config=OllamaConfig(model="qwen2.5:3b-instruct", fallback_on_error=True),
        fallback_strategy="always_defect",
        transport=httpx.MockTransport(handler),
    )

    result = await policy.choose_action(
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


@pytest.mark.asyncio
async def test_instruction_following_can_be_measured_against_strategy_seed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={
                    "message": {
                        "content": '{"action": "D", "confidence": 0.9, "reasoning_summary": "Mirroring the opponent after a defection."}'
                    }
                },
            )
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b-instruct"}]})

    policy = OllamaPolicy(
        config=OllamaConfig(model="qwen2.5:3b-instruct"),
        fallback_strategy="always_cooperate",
        transport=httpx.MockTransport(handler),
    )

    prior_rounds = (
        RoundRecord(round_index=1, left_action="C", right_action="D", left_payoff=0, right_payoff=5),
    )
    result = await policy.choose_action(
        agent=AgentState("agent_1", "lineage_1", "tit_for_tat"),
        opponent=AgentState("agent_2", "lineage_2", "always_defect"),
        prior_rounds=prior_rounds,
        side="left",
        recognition=RecognitionSnapshot("agent_2", "lineage_2", 1.0, "Known defector."),
    )

    assert result.action == "D"


@pytest.mark.asyncio
async def test_ollama_policy_respects_max_parallel_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = OllamaPolicy(
        config=OllamaConfig(model="qwen2.5:3b-instruct", max_parallel_calls=1),
        fallback_strategy="tit_for_tat",
    )
    in_flight = 0
    max_in_flight = 0

    async def fake_request_decision(*, system_prompt: str, user_prompt: str) -> StructuredDecision:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return StructuredDecision(
            action="C",
            confidence=0.5,
            reasoning_summary="ok",
        )

    monkeypatch.setattr(policy, "_request_decision", fake_request_decision)

    await asyncio.gather(
        policy.choose_action(
            agent=AgentState("agent_1", "lineage_1", "tit_for_tat"),
            opponent=AgentState("agent_2", "lineage_2", "always_cooperate"),
            prior_rounds=tuple(),
            side="left",
            recognition=RecognitionSnapshot(None, None, 0.0, "No prior memory."),
        ),
        policy.choose_action(
            agent=AgentState("agent_3", "lineage_3", "tit_for_tat"),
            opponent=AgentState("agent_4", "lineage_4", "always_cooperate"),
            prior_rounds=tuple(),
            side="left",
            recognition=RecognitionSnapshot(None, None, 0.0, "No prior memory."),
        ),
    )

    assert max_in_flight == 1
