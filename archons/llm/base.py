from __future__ import annotations

import json
from typing import Protocol

from jinja2 import Template
from pydantic import BaseModel, Field

from archons.core.models import Action, AgentState, EncounterSide, RecognitionSnapshot, RoundRecord


class StructuredDecision(BaseModel):
    action: Action
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning_summary: str = Field(default="Fallback response.")


class DecisionResult(BaseModel):
    action: Action
    backend: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning_summary: str = Field(default="")
    used_fallback: bool = False
    prompt_text: str = ""
    response_text: str = ""
    error_message: str | None = None
    latency_ms: float = 0.0


class PolicyEngine(Protocol):
    async def ensure_available(self) -> None: ...

    async def choose_action(
        self,
        agent: AgentState,
        opponent: AgentState,
        prior_rounds: tuple[RoundRecord, ...],
        side: EncounterSide,
        recognition: RecognitionSnapshot,
    ) -> DecisionResult: ...


USER_PROMPT_TEMPLATE = Template(
    """self_agent={{ agent.agent_id }}
opponent_agent={{ opponent.agent_id }}
self_strategy_seed={{ agent.strategy }}
opponent_strategy_seed={{ opponent.strategy }}
base_prompt={{ agent.base_prompt }}
policy_prompt={{ agent.policy_prompt }}
recognition_confidence={{ recognition.confidence_formatted }}
recognized_agent={{ recognition.matched_agent_id or 'unknown' }}
recognized_lineage={{ recognition.matched_lineage_id or 'unknown' }}
memory_summary={{ recognition.memory_summary }}
expected_output_format:
Return exactly one JSON object matching this schema.
Output JSON only.
Do not wrap it in markdown, code fences, prose, labels, prefixes, suffixes, or explanations.
Your first character must be '{' and your final character must be '}'.
{{ expected_output_schema }}
history:
{% for line in history_lines %}
{{ line }}
{% else %}
none
{% endfor %}
Choose your next action and reply with the raw JSON object only."""
)


def build_policy_prompt(
    agent: AgentState,
    opponent: AgentState,
    prior_rounds: tuple[RoundRecord, ...],
    side: EncounterSide,
    recognition: RecognitionSnapshot,
) -> tuple[str, str, str]:
    system_prompt = (
        "You are an agent in an iterated Prisoner's Dilemma. "
        "Choose the next action while staying consistent with the provided policy seed. "
        "Return raw JSON only with no markdown, no wrappers, and no extra text. "
        "Your response must begin with '{'."
    )
    expected_output_schema = json.dumps(StructuredDecision.model_json_schema(), indent=2, sort_keys=True)
    history_lines = [
        _render_history_line(round_record=round_record, side=side)
        for round_record in prior_rounds
    ]
    user_prompt = USER_PROMPT_TEMPLATE.render(
        agent=agent,
        opponent=opponent,
        recognition={
            "confidence_formatted": f"{recognition.confidence:.2f}",
            "matched_agent_id": recognition.matched_agent_id,
            "matched_lineage_id": recognition.matched_lineage_id,
            "memory_summary": recognition.memory_summary,
        },
        expected_output_schema=expected_output_schema,
        history_lines=history_lines,
    )
    prompt_text = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
    return system_prompt, user_prompt, prompt_text


def _render_history_line(round_record: RoundRecord, side: EncounterSide) -> str:
    if side == "left":
        self_action = round_record.left_action
        other_action = round_record.right_action
    else:
        self_action = round_record.right_action
        other_action = round_record.left_action
    return f"round={round_record.round_index} self={self_action} other={other_action}"
