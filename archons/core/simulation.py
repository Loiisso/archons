from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import random
import shutil
from typing import Iterable

import yaml

from archons.agents.memory import build_recognition_snapshot, update_agent_memory
from archons.agents.prompting import (
    base_prompt_for_strategy,
    mutate_prompt_traits,
    render_policy_prompt,
    seed_prompt_traits,
)
from archons.analysis.viz import PeriodicVisualizer
from archons.config import ExperimentConfig
from archons.core.models import (
    Action,
    AgentState,
    EncounterRecord,
    EncounterSide,
    GenerationMetrics,
    Position,
    PromptTraits,
    RecognitionSnapshot,
    RoundRecord,
    StrategyName,
)
from archons.core.strategies import choose_deterministic_action
from archons.llm.ollama import OllamaPolicy
from archons.storage.sqlite_store import RunStore


class SimulationSummary:
    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        final_generation: int,
        final_live_cells: int,
        total_agents_created: int,
    ) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.final_generation = final_generation
        self.final_live_cells = final_live_cells
        self.total_agents_created = total_agents_created


class SimulationRunner:
    def __init__(
        self,
        experiment: ExperimentConfig,
        run_dir: Path,
        store: RunStore,
        visualizer: PeriodicVisualizer,
    ) -> None:
        self.experiment = experiment
        self.run_dir = run_dir
        self.store = store
        self.visualizer = visualizer
        self.random = random.Random(experiment.simulation.seed)
        self.agent_counter = 0
        self.ollama_policy = None
        if experiment.agents.backend == "ollama":
            self.ollama_policy = OllamaPolicy(
                config=experiment.agents.ollama,
                fallback_strategy=experiment.agents.fallback_strategy,
            )

    @classmethod
    def from_config(cls, experiment: ExperimentConfig, config_path: Path) -> "SimulationRunner":
        artifact_root = Path(experiment.output.root_dir)

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}-{experiment.name}"
        run_dir = artifact_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        effective_config_path = run_dir / "effective_config.yaml"
        shutil.copyfile(config_path, effective_config_path)

        store = RunStore(run_dir / "simulation.sqlite3")
        visualizer = PeriodicVisualizer(
            output_dir=run_dir / experiment.visualization.output_dir_name,
            config=experiment.visualization,
            world=experiment.world,
        )
        return cls(experiment=experiment, run_dir=run_dir, store=store, visualizer=visualizer)

    def run(self) -> SimulationSummary:
        config_yaml = yaml.safe_dump(self.experiment.model_dump(mode="json"), sort_keys=False)
        started_at = datetime.now(tz=timezone.utc).isoformat()
        run_id = self.run_dir.name
        self.store.create_run(
            run_id=run_id,
            experiment_name=self.experiment.name,
            config_yaml=config_yaml,
            started_at=started_at,
        )

        world = self._seed_world()
        initial_metrics = GenerationMetrics(
            generation=0,
            live_cells=len(world),
            births=0,
            deaths=0,
            survivors=len(world),
            cooperation_rate=0.0,
            average_score=0.0,
            total_encounters=0,
            total_rounds=0,
        )
        self.store.record_generation(run_id=run_id, metrics=initial_metrics, world=world, encounters=tuple())
        self.visualizer.maybe_render(generation=0, world=world, metrics=initial_metrics)

        last_metrics = initial_metrics
        for generation in range(1, self.experiment.simulation.generations + 1):
            encounters = self._resolve_encounters(world=world, generation=generation)
            next_world, metrics = self._advance_world(
                current_world=world,
                generation=generation,
                encounters=encounters,
            )
            self.store.record_generation(
                run_id=run_id,
                metrics=metrics,
                world=next_world,
                encounters=encounters,
            )
            self.visualizer.maybe_render(generation=generation, world=next_world, metrics=metrics)
            world = next_world
            last_metrics = metrics

        self.store.close()
        return SimulationSummary(
            run_id=run_id,
            run_dir=self.run_dir,
            final_generation=last_metrics.generation,
            final_live_cells=last_metrics.live_cells,
            total_agents_created=self.agent_counter,
        )

    def _seed_world(self) -> dict[Position, AgentState]:
        world: dict[Position, AgentState] = {}
        for y in range(self.experiment.world.height):
            for x in range(self.experiment.world.width):
                if self.random.random() <= self.experiment.world.initial_alive_probability:
                    strategy = self._sample_strategy()
                    agent = self._new_agent(strategy=strategy)
                    world[Position(x=x, y=y)] = agent
        return world

    def _sample_strategy(self) -> StrategyName:
        names = list(self.experiment.agents.strategy_weights.keys())
        weights = list(self.experiment.agents.strategy_weights.values())
        chosen = self.random.choices(names, weights=weights, k=1)[0]
        return chosen  # type: ignore[return-value]

    def _new_agent(
        self,
        strategy: StrategyName,
        lineage_id: str | None = None,
        parent: AgentState | None = None,
    ) -> AgentState:
        self.agent_counter += 1
        agent_id = f"agent_{self.agent_counter:05d}"
        if parent is None:
            base_prompt = base_prompt_for_strategy(strategy)
            prompt_traits = seed_prompt_traits(strategy)
        else:
            base_prompt = parent.base_prompt
            prompt_traits = mutate_prompt_traits(
                parent_traits=parent.prompt_traits,
                random=self.random,
                config=self.experiment.agents.prompt_evolution,
            )
        policy_prompt = render_policy_prompt(
            strategy=strategy,
            traits=prompt_traits,
            base_prompt=base_prompt,
        )
        return AgentState(
            agent_id=agent_id,
            lineage_id=lineage_id or agent_id,
            strategy=strategy,
            backend=self.experiment.agents.backend,
            base_prompt=base_prompt,
            policy_prompt=policy_prompt,
            prompt_traits=prompt_traits,
        )

    def _resolve_encounters(
        self,
        world: dict[Position, AgentState],
        generation: int,
    ) -> tuple[EncounterRecord, ...]:
        encounters: list[EncounterRecord] = []
        seen_pairs: set[tuple[Position, Position]] = set()

        for position, agent in world.items():
            for neighbor_position in self._live_neighbor_positions(world=world, position=position):
                pair = tuple(sorted((position, neighbor_position)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                opponent = world[neighbor_position]
                encounter = self._play_encounter(
                    generation=generation,
                    left_position=pair[0],
                    right_position=pair[1],
                    left_agent=world[pair[0]],
                    right_agent=world[pair[1]],
                )
                encounters.append(encounter)
        return tuple(encounters)

    def _play_encounter(
        self,
        generation: int,
        left_position: Position,
        right_position: Position,
        left_agent: AgentState,
        right_agent: AgentState,
    ) -> EncounterRecord:
        rounds: list[RoundRecord] = []
        left_recognition = build_recognition_snapshot(agent=left_agent, opponent=right_agent)
        right_recognition = build_recognition_snapshot(agent=right_agent, opponent=left_agent)
        for round_index in range(1, self.experiment.game.rounds_per_encounter + 1):
            prior_rounds = tuple(rounds)
            left_action = self._choose_action(
                agent=left_agent,
                opponent=right_agent,
                prior_rounds=prior_rounds,
                side="left",
                recognition=left_recognition,
            )
            right_action = self._choose_action(
                agent=right_agent,
                opponent=left_agent,
                prior_rounds=prior_rounds,
                side="right",
                recognition=right_recognition,
            )
            left_payoff, right_payoff = self._score_actions(left_action=left_action, right_action=right_action)

            left_agent.total_score += left_payoff
            left_agent.generation_score += left_payoff
            right_agent.total_score += right_payoff
            right_agent.generation_score += right_payoff

            left_agent.cooperation_count += int(left_action == "C")
            left_agent.defection_count += int(left_action == "D")
            right_agent.cooperation_count += int(right_action == "C")
            right_agent.defection_count += int(right_action == "D")

            rounds.append(
                RoundRecord(
                    round_index=round_index,
                    left_action=left_action,
                    right_action=right_action,
                    left_payoff=left_payoff,
                    right_payoff=right_payoff,
                )
            )

        encounter = EncounterRecord(
            generation=generation,
            left_agent_id=left_agent.agent_id,
            right_agent_id=right_agent.agent_id,
            left_position=left_position,
            right_position=right_position,
            left_recognition=left_recognition,
            right_recognition=right_recognition,
            rounds=tuple(rounds),
        )
        update_agent_memory(agent=left_agent, opponent=right_agent, encounter=encounter, side="left")
        update_agent_memory(agent=right_agent, opponent=left_agent, encounter=encounter, side="right")
        return encounter

    def _choose_action(
        self,
        agent: AgentState,
        opponent: AgentState,
        prior_rounds: tuple[RoundRecord, ...],
        side: EncounterSide,
        recognition: RecognitionSnapshot,
    ) -> Action:
        if self.ollama_policy is not None:
            return self.ollama_policy.choose_action(
                agent=agent,
                opponent=opponent,
                prior_rounds=prior_rounds,
                side=side,
                recognition=recognition,
            )
        return choose_deterministic_action(strategy=agent.strategy, prior_rounds=prior_rounds, side=side)

    def _score_actions(self, left_action: Action, right_action: Action) -> tuple[int, int]:
        payoff = self.experiment.game.payoff
        if left_action == "C" and right_action == "C":
            return payoff.reward, payoff.reward
        if left_action == "C" and right_action == "D":
            return payoff.sucker, payoff.temptation
        if left_action == "D" and right_action == "C":
            return payoff.temptation, payoff.sucker
        return payoff.punishment, payoff.punishment

    def _advance_world(
        self,
        current_world: dict[Position, AgentState],
        generation: int,
        encounters: tuple[EncounterRecord, ...],
    ) -> tuple[dict[Position, AgentState], GenerationMetrics]:
        next_world: dict[Position, AgentState] = {}
        births = 0
        deaths = 0
        survivors = 0

        for y in range(self.experiment.world.height):
            for x in range(self.experiment.world.width):
                position = Position(x=x, y=y)
                live_neighbors = self._live_neighbor_positions(world=current_world, position=position)
                neighbor_count = len(live_neighbors)
                current_agent = current_world.get(position)

                if current_agent is not None and neighbor_count in (2, 3):
                    next_world[position] = replace(current_agent, generation_score=0)
                    survivors += 1
                    continue

                if current_agent is not None:
                    deaths += 1
                    continue

                if neighbor_count == 3:
                    parent_position = max(
                        live_neighbors,
                        key=lambda neighbor: (
                            current_world[neighbor].generation_score,
                            current_world[neighbor].total_score,
                            current_world[neighbor].agent_id,
                        ),
                    )
                    parent = current_world[parent_position]
                    next_world[position] = self._new_agent(
                        strategy=parent.strategy,
                        lineage_id=parent.lineage_id,
                        parent=parent,
                    )
                    births += 1

        current_scores = [agent.generation_score for agent in current_world.values()]
        average_score = sum(current_scores) / len(current_scores) if current_scores else 0.0
        total_rounds = sum(len(encounter.rounds) for encounter in encounters)
        total_cooperations = sum(
            int(round_record.left_action == "C") + int(round_record.right_action == "C")
            for encounter in encounters
            for round_record in encounter.rounds
        )
        total_actions = total_rounds * 2
        cooperation_rate = total_cooperations / total_actions if total_actions else 0.0

        metrics = GenerationMetrics(
            generation=generation,
            live_cells=len(next_world),
            births=births,
            deaths=deaths,
            survivors=survivors,
            cooperation_rate=cooperation_rate,
            average_score=average_score,
            total_encounters=len(encounters),
            total_rounds=total_rounds,
        )
        return next_world, metrics

    def _live_neighbor_positions(
        self,
        world: dict[Position, AgentState],
        position: Position,
    ) -> list[Position]:
        return [neighbor for neighbor in self._neighbor_positions(position) if neighbor in world]

    def _neighbor_positions(self, position: Position) -> Iterable[Position]:
        width = self.experiment.world.width
        height = self.experiment.world.height
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                x = position.x + dx
                y = position.y + dy
                if self.experiment.world.toroidal:
                    yield Position(x=x % width, y=y % height)
                    continue
                if 0 <= x < width and 0 <= y < height:
                    yield Position(x=x, y=y)
