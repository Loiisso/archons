from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class WorldConfig(BaseModel):
    width: int = Field(default=20, ge=3)
    height: int = Field(default=20, ge=3)
    toroidal: bool = True
    initial_alive_probability: float = Field(default=0.35, gt=0.0, lt=1.0)


class PayoffConfig(BaseModel):
    reward: int = 3
    sucker: int = 0
    temptation: int = 5
    punishment: int = 1


class GameConfig(BaseModel):
    rounds_per_encounter: int = Field(default=3, ge=1)
    payoff: PayoffConfig = Field(default_factory=PayoffConfig)
    communication: "CommunicationConfig" = Field(default_factory=lambda: CommunicationConfig())


class CommunicationConfig(BaseModel):
    enabled: bool = False
    mode: Literal["pre_encounter"] = "pre_encounter"
    message_max_chars: int = Field(default=160, ge=1, le=500)


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b-instruct"
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    require_model: bool = True
    fallback_on_error: bool = False


class PromptEvolutionConfig(BaseModel):
    mutation_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    mutation_step: float = Field(default=0.12, gt=0.0, le=1.0)
    style_options: list[str] = Field(
        default_factory=lambda: ["balanced", "cautious", "opportunistic", "vengeful"]
    )


class AgentsConfig(BaseModel):
    backend: Literal["deterministic", "ollama"] = "deterministic"
    strategy_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "always_cooperate": 0.30,
            "always_defect": 0.25,
            "tit_for_tat": 0.25,
            "grim_trigger": 0.20,
        }
    )
    fallback_strategy: Literal[
        "always_cooperate", "always_defect", "tit_for_tat", "grim_trigger"
    ] = "tit_for_tat"
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    prompt_evolution: PromptEvolutionConfig = Field(default_factory=PromptEvolutionConfig)

    @model_validator(mode="after")
    def validate_strategy_weights(self) -> "AgentsConfig":
        total = sum(self.strategy_weights.values())
        if total <= 0:
            raise ValueError("strategy_weights must sum to a value greater than zero")
        return self


class VisualizationConfig(BaseModel):
    enabled: bool = True
    interval: int = Field(default=5, ge=1)
    output_dir_name: str = "viz"
    render_ascii: bool = True


class OutputConfig(BaseModel):
    root_dir: str = "artifacts"


class SimulationConfig(BaseModel):
    generations: int = Field(default=20, ge=1)
    seed: int = 7
    print_progress: bool = True
    progress_interval: int = Field(default=1, ge=1)
    progress_log_name: str = "progress.log"


class ExperimentConfig(BaseModel):
    name: str = "baseline"
    world: WorldConfig = Field(default_factory=WorldConfig)
    game: GameConfig = Field(default_factory=GameConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return ExperimentConfig.model_validate(data)
