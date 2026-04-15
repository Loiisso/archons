from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Action = Literal["C", "D"]
StrategyName = Literal[
    "always_cooperate",
    "always_defect",
    "tit_for_tat",
    "grim_trigger",
]
BackendName = Literal["deterministic", "ollama"]
EncounterSide = Literal["left", "right"]


@dataclass(frozen=True, order=True, slots=True)
class Position:
    x: int
    y: int


@dataclass(slots=True)
class AgentState:
    agent_id: str
    lineage_id: str
    strategy: StrategyName
    backend: BackendName = "deterministic"
    base_prompt: str = "Maintain your lineage identity while playing the iterated Prisoner's Dilemma."
    policy_prompt: str = "Choose one action using the current state."
    prompt_traits: "PromptTraits" = field(default_factory=lambda: PromptTraits())
    total_score: int = 0
    generation_score: int = 0
    cooperation_count: int = 0
    defection_count: int = 0
    opponent_memories: dict[str, "OpponentMemory"] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PromptTraits:
    cooperation_bias: float = 0.5
    retaliation_bias: float = 0.5
    forgiveness_bias: float = 0.5
    memory_weight: float = 0.5
    prompt_style: str = "balanced"


@dataclass(frozen=True, slots=True)
class OpponentMemory:
    opponent_agent_id: str
    opponent_lineage_id: str
    encounters: int
    rounds_played: int
    self_cooperations: int
    self_defections: int
    opponent_cooperations: int
    opponent_defections: int
    cumulative_score_delta: int
    last_generation: int
    last_self_action: Action | None
    last_opponent_action: Action | None

    def summary(self) -> str:
        return (
            f"Known agent {self.opponent_agent_id} from lineage {self.opponent_lineage_id}; "
            f"encounters={self.encounters}, rounds={self.rounds_played}, "
            f"opponent_c={self.opponent_cooperations}, opponent_d={self.opponent_defections}, "
            f"score_delta={self.cumulative_score_delta}, last_seen_generation={self.last_generation}."
        )


@dataclass(frozen=True, slots=True)
class RecognitionSnapshot:
    matched_agent_id: str | None
    matched_lineage_id: str | None
    confidence: float
    memory_summary: str


@dataclass(frozen=True, slots=True)
class RoundRecord:
    round_index: int
    left_action: Action
    right_action: Action
    left_payoff: int
    right_payoff: int


@dataclass(frozen=True, slots=True)
class EncounterRecord:
    generation: int
    left_agent_id: str
    right_agent_id: str
    left_position: Position
    right_position: Position
    left_recognition: RecognitionSnapshot
    right_recognition: RecognitionSnapshot
    rounds: tuple[RoundRecord, ...]


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    generation: int
    live_cells: int
    births: int
    deaths: int
    survivors: int
    cooperation_rate: float
    average_score: float
    total_encounters: int
    total_rounds: int
