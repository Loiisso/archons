from __future__ import annotations

import asyncio
import json
import time

import httpx

from archons.config import OllamaConfig
from archons.core.models import AgentState, EncounterSide, RecognitionSnapshot, RoundRecord
from archons.core.strategies import choose_deterministic_action
from archons.llm.base import DecisionResult, StructuredDecision, build_policy_prompt


class OllamaError(RuntimeError):
    pass


class OllamaPolicy:
    def __init__(
        self,
        config: OllamaConfig,
        fallback_strategy: str,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._fallback_strategy = fallback_strategy
        self._transport = transport
        self._semaphore = asyncio.Semaphore(config.max_parallel_calls)

    async def ensure_available(self) -> None:
        try:
            async with self._semaphore:
                async with httpx.AsyncClient(
                    timeout=self._config.timeout_seconds,
                    transport=self._transport,
                ) as client:
                    response = await client.get(f"{self._config.base_url}/api/tags")
                    response.raise_for_status()
                    body = response.json()
        except Exception as exc:
            raise OllamaError(f"Unable to reach Ollama at {self._config.base_url}: {exc}") from exc

        available_models = {model["name"] for model in body.get("models", []) if "name" in model}
        if self._config.require_model and self._config.model not in available_models:
            raise OllamaError(
                f"Model {self._config.model!r} is not available in local Ollama. "
                f"Available models: {sorted(available_models)}"
            )

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
                decision = await self._request_decision(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            latency_ms = (time.perf_counter() - started) * 1000
            return DecisionResult(
                action=decision.action,
                backend="ollama",
                confidence=decision.confidence,
                reasoning_summary=decision.reasoning_summary,
                prompt_text=prompt_text,
                response_text=decision.model_dump_json(),
                latency_ms=latency_ms,
            )
        except Exception as exc:
            if not self._config.fallback_on_error:
                raise OllamaError(f"Ollama decision failed: {exc}") from exc

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
                reasoning_summary="Fallback strategy used after Ollama error.",
                used_fallback=True,
                prompt_text=prompt_text,
                error_message=str(exc),
                latency_ms=latency_ms,
            )

    async def _request_decision(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> StructuredDecision:
        payload = {
            "model": self._config.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": self._config.temperature},
        }

        async with httpx.AsyncClient(
            timeout=self._config.timeout_seconds,
            transport=self._transport,
        ) as client:
            response = await client.post(f"{self._config.base_url}/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()

        content = body["message"]["content"]
        parsed = self._parse_json_response(content)
        return StructuredDecision.model_validate(parsed)

    def _parse_json_response(self, content: str) -> dict[str, object]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(content[start : end + 1])
