from __future__ import annotations

import json

import httpx
from pydantic import BaseModel, Field

from archons.config import OllamaConfig
from archons.core.models import Action, AgentState, EncounterSide, RecognitionSnapshot, RoundRecord
from archons.core.strategies import choose_deterministic_action


class OllamaDecision(BaseModel):
    action: Action
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning_summary: str = Field(default="Fallback response.")


class OllamaPolicy:
    def __init__(self, config: OllamaConfig, fallback_strategy: str) -> None:
        self._config = config
        self._fallback_strategy = fallback_strategy

    def choose_action(
        self,
        agent: AgentState,
        opponent: AgentState,
        prior_rounds: tuple[RoundRecord, ...],
        side: EncounterSide,
        recognition: RecognitionSnapshot,
    ) -> Action:
        try:
            decision = self._request_decision(
                agent=agent,
                opponent=opponent,
                prior_rounds=prior_rounds,
                side=side,
                recognition=recognition,
            )
            return decision.action
        except Exception:
            return choose_deterministic_action(
                strategy=self._fallback_strategy,
                prior_rounds=prior_rounds,
                side=side,
            )

    def _request_decision(
        self,
        agent: AgentState,
        opponent: AgentState,
        prior_rounds: tuple[RoundRecord, ...],
        side: EncounterSide,
        recognition: RecognitionSnapshot,
    ) -> OllamaDecision:
        system_prompt = (
            "You are an agent in an iterated Prisoner's Dilemma. Return JSON only with keys "
            "action, confidence, and reasoning_summary. action must be C or D."
        )
        history_lines = [
            self._render_history_line(round_record=round_record, side=side)
            for round_record in prior_rounds
        ]
        user_prompt = "\n".join(
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
                "Choose your next action.",
            ]
        )

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

        with httpx.Client(timeout=self._config.timeout_seconds) as client:
            response = client.post(f"{self._config.base_url}/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()

        content = body["message"]["content"]
        parsed = json.loads(content)
        return OllamaDecision.model_validate(parsed)

    def _render_history_line(self, round_record: RoundRecord, side: EncounterSide) -> str:
        if side == "left":
            self_action = round_record.left_action
            other_action = round_record.right_action
        else:
            self_action = round_record.right_action
            other_action = round_record.left_action
        return f"round={round_record.round_index} self={self_action} other={other_action}"
