from __future__ import annotations

import asyncio
import time

import typer
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage

from archons.config import OpenAIConfig
from archons.core.models import (
    AgentState,
    EncounterSide,
    RecognitionSnapshot,
    RoundRecord,
)
from archons.core.strategies import choose_deterministic_action
from archons.llm.base import DecisionResult, StructuredDecision, build_policy_prompt


class OpenAIPolicyError(RuntimeError):
    pass


class OpenAIPolicy:
    def __init__(
        self,
        config: OpenAIConfig,
        fallback_strategy: str,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._config = config
        self._fallback_strategy = fallback_strategy
        self._client = client
        self._semaphore = asyncio.Semaphore(config.max_parallel_calls)

    async def ensure_available(self) -> None:
        try:
            client = self._get_client()
            async with self._semaphore:
                r = await client.chat.completions.create(
                    model=self._config.model,
                    messages=[
                        {"role": "user", "content": "Say 'test'"},
                    ],
                    max_tokens=1,
                    **self._assemble_extra_generation_parameters(),
                )
            result = self.get_reasoning(r.choices[0].message) + str(
                r.choices[0].message.content
            )
            assert len(result) > 0
        except Exception as exc:
            raise OpenAIPolicyError(
                f"Unable to initialize OpenAI client: {exc}"
            ) from exc

    @staticmethod
    def get_reasoning(msg: ChatCompletionMessage) -> str:
        result = ""
        if hasattr(msg, "reasoning_content"):
            result = msg.reasoning_content
        elif hasattr(msg, "reasoning"):
            result = msg.reasoning
        else:
            pass
        return result

    def _assemble_extra_generation_parameters(self) -> dict:
        result: dict = {"extra_body": {}}
        if self._config.temperature:
            result["temperature"] = self._config.temperature
        if self._config.presence_penalty:
            result["presence_penalty"] = self._config.presence_penalty
        if self._config.top_p:
            result["top_p"] = self._config.top_p
        if self._config.top_k:
            result["extra_body"]["top_k"] = self._config.top_k
        if self._config.min_p:
            result["extra_body"]["min_p"] = self._config.min_p
        if self._config.repetition_penalty:
            result["extra_body"]["repetition_penalty"] = self._config.repetition_penalty

        return result

    async def choose_action(
        self,
        agent: AgentState,
        opponent: AgentState,
        prior_rounds: tuple[RoundRecord, ...],
        side: EncounterSide,
        recognition: RecognitionSnapshot,
    ) -> DecisionResult:
        system_prompt, user_prompt, prompt_text = build_policy_prompt(
            agent=agent,
            opponent=opponent,
            prior_rounds=prior_rounds,
            side=side,
            recognition=recognition,
        )
        started = time.perf_counter()
        try:
            async with self._semaphore:
                completion = await self._get_client().chat.completions.parse(
                    model=self._config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=StructuredDecision,
                    **(self._assemble_extra_generation_parameters()),
                )
            latency_ms = (time.perf_counter() - started) * 1000
            message = completion.choices[0].message
            if message.refusal:
                raise OpenAIPolicyError(f"Model refusal: {message.refusal}")
            decision = message.parsed
            if decision is None:
                raise OpenAIPolicyError("OpenAI returned no parsed decision.")
            response_text = message.content or decision.model_dump_json()
            return DecisionResult(
                action=decision.action,
                backend="openai",
                confidence=decision.confidence,
                reasoning_summary=decision.reasoning_summary,
                prompt_text=prompt_text,
                response_text=response_text,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            if not self._config.fallback_on_error:
                raise OpenAIPolicyError(f"OpenAI decision failed: {exc}") from exc

            # TODO: track how often it happens

            latency_ms = (time.perf_counter() - started) * 1000
            fallback_action = choose_deterministic_action(
                strategy=self._fallback_strategy,
                prior_rounds=prior_rounds,
                side=side,
            )
            return DecisionResult(
                action=fallback_action,
                backend="deterministic-fallback",
                confidence=0.0,
                reasoning_summary="Fallback strategy used after OpenAI error.",
                used_fallback=True,
                prompt_text=prompt_text,
                error_message=str(exc),
                latency_ms=latency_ms,
            )

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
                timeout=self._config.timeout_seconds,
                max_retries=self._config.max_retries,
            )
        return self._client
