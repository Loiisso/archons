from __future__ import annotations

import json
import time

import httpx
from pydantic import BaseModel, Field

from archons.config import OllamaConfig
from archons.core.models import Action, AgentState, EncounterSide, RecognitionSnapshot, RoundRecord
from archons.core.strategies import choose_deterministic_action


class OllamaDecision(BaseModel):
    action: Action
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning_summary: str = Field(default="Fallback response.")


class OllamaMessage(BaseModel):
    message: str = Field(default="")
    intent: str = Field(default="neutral")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class DecisionResult(BaseModel):
    action: Action
    backend: str = "ollama"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning_summary: str = Field(default="")
    used_fallback: bool = False
    prompt_text: str = ""
    response_text: str = ""
    error_message: str | None = None
    latency_ms: float = 0.0


class MessageResult(BaseModel):
    message_text: str = ""
    intent: str = "neutral"
    backend: str = "ollama"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    used_fallback: bool = False
    prompt_text: str = ""
    response_text: str = ""
    error_message: str | None = None
    latency_ms: float = 0.0


class OllamaError(RuntimeError):
    pass


class OllamaPolicy:
    def __init__(
        self,
        config: OllamaConfig,
        fallback_strategy: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._fallback_strategy = fallback_strategy
        self._transport = transport

    def ensure_available(self) -> None:
        try:
            with httpx.Client(
                timeout=self._config.timeout_seconds,
                transport=self._transport,
            ) as client:
                response = client.get(f"{self._config.base_url}/api/tags")
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

    def choose_action(
        self,
        agent: AgentState,
        opponent: AgentState,
        prior_rounds: tuple[RoundRecord, ...],
        side: EncounterSide,
        recognition: RecognitionSnapshot,
        incoming_message: str | None = None,
        incoming_intent: str | None = None,
    ) -> DecisionResult:
        system_prompt = (
            "You are an agent in an iterated Prisoner's Dilemma. Return JSON only with keys "
            "action, confidence, and reasoning_summary. action must be C or D."
        )
        user_prompt = self._build_user_prompt(
            agent=agent,
            opponent=opponent,
            prior_rounds=prior_rounds,
            side=side,
            recognition=recognition,
            incoming_message=incoming_message,
            incoming_intent=incoming_intent,
        )
        prompt_text = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
        started = time.perf_counter()
        try:
            decision = self._request_decision(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            return DecisionResult(
                action=decision.action,
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

    def generate_message(
        self,
        agent: AgentState,
        opponent: AgentState,
        prior_rounds: tuple[RoundRecord, ...],
        side: EncounterSide,
        recognition: RecognitionSnapshot,
        max_chars: int,
    ) -> MessageResult:
        system_prompt = (
            "You are an agent in an iterated Prisoner's Dilemma. Before choosing an action, send one "
            "short message to the opponent. Return JSON only with keys message, intent, and confidence."
        )
        user_prompt = self._build_message_prompt(
            agent=agent,
            opponent=opponent,
            prior_rounds=prior_rounds,
            side=side,
            recognition=recognition,
            max_chars=max_chars,
        )
        prompt_text = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
        started = time.perf_counter()
        try:
            message = self._request_message(system_prompt=system_prompt, user_prompt=user_prompt)
            latency_ms = (time.perf_counter() - started) * 1000
            message_text = message.message.strip().replace("\n", " ")[:max_chars]
            return MessageResult(
                message_text=message_text,
                intent=message.intent,
                confidence=message.confidence,
                prompt_text=prompt_text,
                response_text=message.model_dump_json(),
                latency_ms=latency_ms,
            )
        except Exception as exc:
            if not self._config.fallback_on_error:
                raise OllamaError(f"Ollama message generation failed: {exc}") from exc

            latency_ms = (time.perf_counter() - started) * 1000
            return MessageResult(
                message_text="",
                intent="silent",
                backend="deterministic-fallback",
                confidence=0.0,
                used_fallback=True,
                prompt_text=prompt_text,
                error_message=str(exc),
                latency_ms=latency_ms,
            )

    def _build_user_prompt(
        self,
        agent: AgentState,
        opponent: AgentState,
        prior_rounds: tuple[RoundRecord, ...],
        side: EncounterSide,
        recognition: RecognitionSnapshot,
        incoming_message: str | None = None,
        incoming_intent: str | None = None,
    ) -> str:
        history_lines = [
            self._render_history_line(round_record=round_record, side=side)
            for round_record in prior_rounds
        ]
        return "\n".join(
            [
                f"self_agent={agent.agent_id}",
                f"opponent_agent={opponent.agent_id}",
                f"self_strategy_seed={agent.strategy}",
                f"opponent_strategy_seed={opponent.strategy}",
                f"base_prompt={agent.base_prompt}",
                f"policy_prompt={agent.policy_prompt}",
                f"recognition_confidence={recognition.confidence:.2f}",
                f"recognized_agent={recognition.matched_agent_id or 'unknown'}",
                f"recognized_lineage={recognition.matched_lineage_id or 'unknown'}",
                f"memory_summary={recognition.memory_summary}",
                f"incoming_message={incoming_message or 'none'}",
                f"incoming_intent={incoming_intent or 'none'}",
                "history:",
                *history_lines,
                "Choose your next action.",
            ]
        )

    def _build_message_prompt(
        self,
        agent: AgentState,
        opponent: AgentState,
        prior_rounds: tuple[RoundRecord, ...],
        side: EncounterSide,
        recognition: RecognitionSnapshot,
        max_chars: int,
    ) -> str:
        history_lines = [
            self._render_history_line(round_record=round_record, side=side)
            for round_record in prior_rounds
        ]
        return "\n".join(
            [
                f"self_agent={agent.agent_id}",
                f"opponent_agent={opponent.agent_id}",
                f"self_strategy_seed={agent.strategy}",
                f"opponent_strategy_seed={opponent.strategy}",
                f"base_prompt={agent.base_prompt}",
                f"policy_prompt={agent.policy_prompt}",
                f"recognition_confidence={recognition.confidence:.2f}",
                f"recognized_agent={recognition.matched_agent_id or 'unknown'}",
                f"recognized_lineage={recognition.matched_lineage_id or 'unknown'}",
                f"memory_summary={recognition.memory_summary}",
                "history:",
                *history_lines,
                f"Send one pre-encounter message to the opponent in at most {max_chars} characters.",
            ]
        )

    def _request_decision(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> OllamaDecision:
        payload = self._build_payload(system_prompt=system_prompt, user_prompt=user_prompt)

        with httpx.Client(
            timeout=self._config.timeout_seconds,
            transport=self._transport,
        ) as client:
            response = client.post(f"{self._config.base_url}/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()

        content = body["message"]["content"]
        parsed = self._parse_json_response(content)
        return OllamaDecision.model_validate(parsed)

    def _request_message(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> OllamaMessage:
        payload = self._build_payload(system_prompt=system_prompt, user_prompt=user_prompt)

        with httpx.Client(
            timeout=self._config.timeout_seconds,
            transport=self._transport,
        ) as client:
            response = client.post(f"{self._config.base_url}/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()

        content = body["message"]["content"]
        parsed = self._parse_json_response(content)
        return OllamaMessage.model_validate(parsed)

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, object]:
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
        return payload

    def _parse_json_response(self, content: str) -> dict[str, object]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(content[start : end + 1])

    def _render_history_line(self, round_record: RoundRecord, side: EncounterSide) -> str:
        if side == "left":
            self_action = round_record.left_action
            other_action = round_record.right_action
        else:
            self_action = round_record.right_action
            other_action = round_record.left_action
        return f"round={round_record.round_index} self={self_action} other={other_action}"
