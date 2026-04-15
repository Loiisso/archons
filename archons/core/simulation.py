from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import random
import shutil
import time
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
    DecisionTrace,
    EncounterRecord,
    EncounterSide,
    GenerationProfile,
    GenerationMetrics,
    Position,
    PromptTraits,
    RecognitionSnapshot,
    RoundRecord,
    StrategyName,
)
from archons.core.strategies import choose_deterministic_action
from archons.llm.ollama import DecisionResult, OllamaPolicy
from archons.storage.sqlite_store import RunStore, export_profiles_csv


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


class ProgressReporter:
    def __init__(self, run_dir: Path, enabled: bool, interval: int, log_name: str) -> None:
        self._enabled = enabled
        self._interval = interval
        self._log_path = run_dir / log_name
        self._started = time.perf_counter()

    @property
    def log_path(self) -> Path:
        return self._log_path

    def emit(self, message: str, force: bool = False) -> None:
        if not self._enabled and not force:
            return
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def maybe_emit_generation(
        self,
        generation: int,
        total_generations: int,
        metrics: GenerationMetrics,
        profile: GenerationProfile,
    ) -> None:
        if generation % self._interval != 0 and generation != total_generations:
            return

        elapsed_seconds = time.perf_counter() - self._started
        self.emit(
            " | ".join(
                [
                    f"generation={generation}/{total_generations}",
                    f"elapsed={elapsed_seconds:.1f}s",
                    f"gen_time={profile.total_seconds:.2f}s",
                    f"resolve={profile.resolve_encounters_seconds:.2f}s",
                    f"advance={profile.advance_world_seconds:.2f}s",
                    f"persist={profile.persist_seconds:.2f}s",
                    f"viz={profile.visualize_seconds:.2f}s",
                    f"model={profile.ollama_latency_seconds:.2f}s",
                    f"live={metrics.live_cells}",
                    f"births={metrics.births}",
                    f"deaths={metrics.deaths}",
                    f"coop_rate={metrics.cooperation_rate:.2f}",
                    f"encounters={metrics.total_encounters}",
                    f"decision_traces={profile.decision_trace_count}",
                    f"ollama={profile.ollama_decisions}",
                    f"fallback={profile.fallback_decisions}",
                ]
            )
        )


class SimulationRunner:
    def __init__(
        self,
        experiment: ExperimentConfig,
        run_dir: Path,
        store: RunStore,
        visualizer: PeriodicVisualizer,
        progress_reporter: ProgressReporter | None = None,
    ) -> None:
        self.experiment = experiment
        self.run_dir = run_dir
        self.store = store
        self.visualizer = visualizer
        self.progress_reporter = progress_reporter or ProgressReporter(
            run_dir=run_dir,
            enabled=experiment.simulation.print_progress,
            interval=experiment.simulation.progress_interval,
            log_name=experiment.simulation.progress_log_name,
        )
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
        progress_reporter = ProgressReporter(
            run_dir=run_dir,
            enabled=experiment.simulation.print_progress,
            interval=experiment.simulation.progress_interval,
            log_name=experiment.simulation.progress_log_name,
        )
        return cls(
            experiment=experiment,
            run_dir=run_dir,
            store=store,
            visualizer=visualizer,
            progress_reporter=progress_reporter,
        )

    def run(self) -> SimulationSummary:
        if self.ollama_policy is not None:
            self.ollama_policy.ensure_available()

        config_yaml = yaml.safe_dump(self.experiment.model_dump(mode="json"), sort_keys=False)
        started_at = datetime.now(tz=timezone.utc).isoformat()
        run_id = self.run_dir.name
        self.store.create_run(
            run_id=run_id,
            experiment_name=self.experiment.name,
            config_yaml=config_yaml,
            started_at=started_at,
        )
        self.progress_reporter.emit(
            f"run_started | run_id={run_id} | backend={self.experiment.agents.backend} | generations={self.experiment.simulation.generations}",
            force=True,
        )

        initial_generation_started = time.perf_counter()
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
        initial_persist_started = time.perf_counter()
        self.store.record_generation(run_id=run_id, metrics=initial_metrics, world=world, encounters=tuple())
        initial_persist_finished = time.perf_counter()
        self.visualizer.maybe_render(generation=0, world=world, metrics=initial_metrics)
        initial_visualize_finished = time.perf_counter()
        initial_profile = self._build_generation_profile(
            generation=0,
            encounters=tuple(),
            total_seconds=initial_visualize_finished - initial_generation_started,
            resolve_encounters_seconds=0.0,
            advance_world_seconds=0.0,
            persist_seconds=initial_persist_finished - initial_persist_started,
            visualize_seconds=initial_visualize_finished - initial_persist_finished,
        )
        self.store.record_generation_profile(run_id=run_id, profile=initial_profile)
        self.progress_reporter.emit(
            f"generation=0/{self.experiment.simulation.generations} | live={initial_metrics.live_cells} | seeded_agents={len(world)}",
            force=True,
        )

        last_metrics = initial_metrics
        for generation in range(1, self.experiment.simulation.generations + 1):
            generation_started = time.perf_counter()
            encounters = self._resolve_encounters(world=world, generation=generation)
            resolve_finished = time.perf_counter()
            next_world, metrics = self._advance_world(
                current_world=world,
                generation=generation,
                encounters=encounters,
            )
            advance_finished = time.perf_counter()
            self.store.record_generation(
                run_id=run_id,
                metrics=metrics,
                world=next_world,
                encounters=encounters,
            )
            persist_finished = time.perf_counter()
            self.visualizer.maybe_render(generation=generation, world=next_world, metrics=metrics)
            visualize_finished = time.perf_counter()
            profile = self._build_generation_profile(
                generation=generation,
                encounters=encounters,
                total_seconds=visualize_finished - generation_started,
                resolve_encounters_seconds=resolve_finished - generation_started,
                advance_world_seconds=advance_finished - resolve_finished,
                persist_seconds=persist_finished - advance_finished,
                visualize_seconds=visualize_finished - persist_finished,
            )
            self.store.record_generation_profile(run_id=run_id, profile=profile)
            self.progress_reporter.maybe_emit_generation(
                generation=generation,
                total_generations=self.experiment.simulation.generations,
                metrics=metrics,
                profile=profile,
            )
            world = next_world
            last_metrics = metrics

        self.store.close()
        profiles_csv_path = export_profiles_csv(
            db_path=self.store.db_path,
            output_path=self.run_dir / "profiles.csv",
        )
        self.progress_reporter.emit(
            " | ".join(
                [
                    "run_complete",
                    f"run_id={run_id}",
                    f"final_generation={last_metrics.generation}",
                    f"final_live={last_metrics.live_cells}",
                    f"agents_created={self.agent_counter}",
                    f"progress_log={self.progress_reporter.log_path}",
                    f"profiles_csv={profiles_csv_path}",
                ]
            ),
            force=True,
        )
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
        decision_traces: list[DecisionTrace] = []
        left_recognition = build_recognition_snapshot(agent=left_agent, opponent=right_agent)
        right_recognition = build_recognition_snapshot(agent=right_agent, opponent=left_agent)
        for round_index in range(1, self.experiment.game.rounds_per_encounter + 1):
            prior_rounds = tuple(rounds)
            left_result = self._choose_action(
                agent=left_agent,
                opponent=right_agent,
                prior_rounds=prior_rounds,
                side="left",
                recognition=left_recognition,
            )
            right_result = self._choose_action(
                agent=right_agent,
                opponent=left_agent,
                prior_rounds=prior_rounds,
                side="right",
                recognition=right_recognition,
            )
            left_action = left_result.action
            right_action = right_result.action
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
            decision_traces.extend(
                [
                    DecisionTrace(
                        round_index=round_index,
                        side="left",
                        agent_id=left_agent.agent_id,
                        opponent_id=right_agent.agent_id,
                        backend=left_result.backend,
                        action=left_result.action,
                        confidence=left_result.confidence,
                        reasoning_summary=left_result.reasoning_summary,
                        used_fallback=left_result.used_fallback,
                        error_message=left_result.error_message,
                        latency_ms=left_result.latency_ms,
                        prompt_text=left_result.prompt_text,
                        response_text=left_result.response_text,
                    ),
                    DecisionTrace(
                        round_index=round_index,
                        side="right",
                        agent_id=right_agent.agent_id,
                        opponent_id=left_agent.agent_id,
                        backend=right_result.backend,
                        action=right_result.action,
                        confidence=right_result.confidence,
                        reasoning_summary=right_result.reasoning_summary,
                        used_fallback=right_result.used_fallback,
                        error_message=right_result.error_message,
                        latency_ms=right_result.latency_ms,
                        prompt_text=right_result.prompt_text,
                        response_text=right_result.response_text,
                    ),
                ]
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
            decision_traces=tuple(decision_traces),
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
    ) -> DecisionResult:
        if self.ollama_policy is not None:
            return self.ollama_policy.choose_action(
                agent=agent,
                opponent=opponent,
                prior_rounds=prior_rounds,
                side=side,
                recognition=recognition,
            )
        action = choose_deterministic_action(strategy=agent.strategy, prior_rounds=prior_rounds, side=side)
        return DecisionResult(
            action=action,
            backend="deterministic",
            confidence=1.0,
            reasoning_summary=f"Deterministic policy {agent.strategy} selected the action.",
        )

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

    def _build_generation_profile(
        self,
        generation: int,
        encounters: tuple[EncounterRecord, ...],
        total_seconds: float,
        resolve_encounters_seconds: float,
        advance_world_seconds: float,
        persist_seconds: float,
        visualize_seconds: float,
    ) -> GenerationProfile:
        decision_traces = [
            decision_trace
            for encounter in encounters
            for decision_trace in encounter.decision_traces
        ]
        ollama_decision_latencies_ms = [
            trace.latency_ms for trace in decision_traces if trace.backend == "ollama"
        ]
        ollama_latency_seconds = sum(ollama_decision_latencies_ms) / 1000.0
        mean_decision_latency_ms = (
            sum(ollama_decision_latencies_ms) / len(ollama_decision_latencies_ms)
            if ollama_decision_latencies_ms
            else 0.0
        )
        max_decision_latency_ms = max(ollama_decision_latencies_ms, default=0.0)
        overhead_seconds = max(
            0.0,
            total_seconds
            - resolve_encounters_seconds
            - advance_world_seconds
            - persist_seconds
            - visualize_seconds,
        )
        return GenerationProfile(
            generation=generation,
            total_seconds=total_seconds,
            resolve_encounters_seconds=resolve_encounters_seconds,
            advance_world_seconds=advance_world_seconds,
            persist_seconds=persist_seconds,
            visualize_seconds=visualize_seconds,
            overhead_seconds=overhead_seconds,
            decision_trace_count=len(decision_traces),
            ollama_decisions=sum(1 for trace in decision_traces if trace.backend == "ollama"),
            fallback_decisions=sum(1 for trace in decision_traces if trace.used_fallback),
            ollama_latency_seconds=ollama_latency_seconds,
            mean_decision_latency_ms=mean_decision_latency_ms,
            max_decision_latency_ms=max_decision_latency_ms,
        )

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
