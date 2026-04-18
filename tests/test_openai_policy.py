import asyncio
from types import SimpleNamespace

import pytest

from archons.config import OpenAIConfig
from archons.core.models import AgentState, RecognitionSnapshot, RoundRecord
from archons.llm.base import StructuredDecision
from archons.llm.openai_policy import OpenAIPolicy, OpenAIPolicyError


class FakeModelsAPI:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.retrieved_model: str | None = None

    async def retrieve(self, model: str) -> object:
        self.retrieved_model = model
        if self.error is not None:
            raise self.error
        return SimpleNamespace(id=model)


class FakeCompletionsAPI:
    def __init__(self, completion: object | None = None, error: Exception | None = None) -> None:
        self.completion = completion
        self.error = error
        self.last_kwargs: dict[str, object] | None = None
        self.last_create_kwargs: dict[str, object] | None = None

    async def parse(self, **kwargs: object) -> object:
        self.last_kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.completion

    async def create(self, **kwargs: object) -> object:
        self.last_create_kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.completion


class FakeOpenAIClient:
    def __init__(
        self,
        completion: object | None = None,
        model_error: Exception | None = None,
        parse_error: Exception | None = None,
    ) -> None:
        self.models = FakeModelsAPI(error=model_error)
        self.chat = SimpleNamespace(
            completions=FakeCompletionsAPI(completion=completion, error=parse_error)
        )


def _completion(
    *,
    parsed: StructuredDecision | None = None,
    refusal: str | None = None,
    content: str | None = None,
) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=parsed,
                    refusal=refusal,
                    content=content,
                )
            )
        ]
    )


@pytest.mark.asyncio
async def test_openai_policy_uses_structured_outputs() -> None:
    client = FakeOpenAIClient(
        completion=_completion(
            parsed=StructuredDecision(
                action="C",
                confidence=0.77,
                reasoning_summary="Opponent has been cooperative so far.",
            ),
            content='{"action":"C","confidence":0.77,"reasoning_summary":"Opponent has been cooperative so far."}',
        )
    )
    policy = OpenAIPolicy(
        config=OpenAIConfig(model="gpt-4o-mini"),
        fallback_strategy="always_defect",
        client=client,  # type: ignore[arg-type]
    )

    result = await policy.choose_action(
        agent=AgentState("agent_1", "lineage_1", "tit_for_tat"),
        opponent=AgentState("agent_2", "lineage_2", "always_cooperate"),
        prior_rounds=tuple(),
        side="left",
        recognition=RecognitionSnapshot(None, None, 0.0, "No prior memory."),
    )

    assert result.action == "C"
    assert result.backend == "openai"
    assert result.used_fallback is False
    assert client.chat.completions.last_kwargs is not None
    assert client.chat.completions.last_kwargs["response_format"] is StructuredDecision


@pytest.mark.asyncio
async def test_openai_policy_can_check_model_availability() -> None:
    client = FakeOpenAIClient(
        completion=_completion(
            content="test",
        )
    )
    policy = OpenAIPolicy(
        config=OpenAIConfig(model="gpt-4o-mini", require_model=True),
        fallback_strategy="tit_for_tat",
        client=client,  # type: ignore[arg-type]
    )

    await policy.ensure_available()

    assert client.chat.completions.last_create_kwargs is not None
    assert client.chat.completions.last_create_kwargs["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_openai_policy_raises_on_refusal_without_fallback() -> None:
    client = FakeOpenAIClient(
        completion=_completion(
            parsed=None,
            refusal="I can't help with that request.",
        )
    )
    policy = OpenAIPolicy(
        config=OpenAIConfig(model="gpt-4o-mini", fallback_on_error=False),
        fallback_strategy="always_defect",
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(OpenAIPolicyError):
        await policy.choose_action(
            agent=AgentState("agent_1", "lineage_1", "tit_for_tat"),
            opponent=AgentState("agent_2", "lineage_2", "always_defect"),
            prior_rounds=tuple(),
            side="left",
            recognition=RecognitionSnapshot(None, None, 0.0, "No prior memory."),
        )


@pytest.mark.asyncio
async def test_openai_policy_can_fallback_when_enabled() -> None:
    prior_rounds = (
        RoundRecord(round_index=1, left_action="C", right_action="D", left_payoff=0, right_payoff=5),
    )
    client = FakeOpenAIClient(
        completion=_completion(
            parsed=None,
            refusal="Refusing the request.",
        )
    )
    policy = OpenAIPolicy(
        config=OpenAIConfig(model="gpt-4o-mini", fallback_on_error=True),
        fallback_strategy="tit_for_tat",
        client=client,  # type: ignore[arg-type]
    )

    result = await policy.choose_action(
        agent=AgentState("agent_1", "lineage_1", "tit_for_tat"),
        opponent=AgentState("agent_2", "lineage_2", "always_defect"),
        prior_rounds=prior_rounds,
        side="left",
        recognition=RecognitionSnapshot("agent_2", "lineage_2", 1.0, "Known defector."),
    )

    assert result.action == "D"
    assert result.used_fallback is True
    assert result.backend == "deterministic-fallback"
    assert result.error_message is not None


@pytest.mark.asyncio
async def test_openai_policy_respects_max_parallel_calls() -> None:
    client = FakeOpenAIClient(
        completion=_completion(
            parsed=StructuredDecision(
                action="C",
                confidence=0.5,
                reasoning_summary="ok",
            ),
            content='{"action":"C","confidence":0.5,"reasoning_summary":"ok"}',
        )
    )
    policy = OpenAIPolicy(
        config=OpenAIConfig(model="gpt-4o-mini", max_parallel_calls=1),
        fallback_strategy="tit_for_tat",
        client=client,  # type: ignore[arg-type]
    )
    in_flight = 0
    max_in_flight = 0

    async def delayed_parse(**kwargs: object) -> object:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return client.chat.completions.completion

    client.chat.completions.parse = delayed_parse  # type: ignore[method-assign]

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
