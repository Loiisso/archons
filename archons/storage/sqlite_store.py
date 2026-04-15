from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from archons.core.models import AgentState, EncounterRecord, GenerationMetrics, Position


class RunStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_name TEXT NOT NULL,
                config_yaml TEXT NOT NULL,
                started_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS generations (
                run_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                live_cells INTEGER NOT NULL,
                births INTEGER NOT NULL,
                deaths INTEGER NOT NULL,
                survivors INTEGER NOT NULL,
                cooperation_rate REAL NOT NULL,
                average_score REAL NOT NULL,
                total_encounters INTEGER NOT NULL,
                total_rounds INTEGER NOT NULL,
                PRIMARY KEY (run_id, generation)
            );

            CREATE TABLE IF NOT EXISTS cells (
                run_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                lineage_id TEXT NOT NULL,
                strategy TEXT NOT NULL,
                base_prompt TEXT NOT NULL,
                policy_prompt TEXT NOT NULL,
                cooperation_bias REAL NOT NULL,
                retaliation_bias REAL NOT NULL,
                forgiveness_bias REAL NOT NULL,
                memory_weight REAL NOT NULL,
                prompt_style TEXT NOT NULL,
                total_score INTEGER NOT NULL,
                PRIMARY KEY (run_id, generation, x, y)
            );

            CREATE TABLE IF NOT EXISTS encounters (
                encounter_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                left_agent_id TEXT NOT NULL,
                right_agent_id TEXT NOT NULL,
                left_recognized_agent_id TEXT,
                left_recognized_lineage_id TEXT,
                left_recognition_confidence REAL NOT NULL,
                left_memory_summary TEXT NOT NULL,
                right_recognized_agent_id TEXT,
                right_recognized_lineage_id TEXT,
                right_recognition_confidence REAL NOT NULL,
                right_memory_summary TEXT NOT NULL,
                left_x INTEGER NOT NULL,
                left_y INTEGER NOT NULL,
                right_x INTEGER NOT NULL,
                right_y INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memories (
                run_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                opponent_agent_id TEXT NOT NULL,
                opponent_lineage_id TEXT NOT NULL,
                encounters INTEGER NOT NULL,
                rounds_played INTEGER NOT NULL,
                self_cooperations INTEGER NOT NULL,
                self_defections INTEGER NOT NULL,
                opponent_cooperations INTEGER NOT NULL,
                opponent_defections INTEGER NOT NULL,
                cumulative_score_delta INTEGER NOT NULL,
                last_generation INTEGER NOT NULL,
                last_self_action TEXT,
                last_opponent_action TEXT,
                PRIMARY KEY (run_id, generation, agent_id, opponent_agent_id)
            );

            CREATE TABLE IF NOT EXISTS rounds (
                encounter_id TEXT NOT NULL,
                round_index INTEGER NOT NULL,
                left_action TEXT NOT NULL,
                right_action TEXT NOT NULL,
                left_payoff INTEGER NOT NULL,
                right_payoff INTEGER NOT NULL,
                PRIMARY KEY (encounter_id, round_index)
            );
            """
        )
        self.connection.commit()

    def create_run(self, run_id: str, experiment_name: str, config_yaml: str, started_at: str) -> None:
        self.connection.execute(
            "INSERT INTO runs (run_id, experiment_name, config_yaml, started_at) VALUES (?, ?, ?, ?)",
            (run_id, experiment_name, config_yaml, started_at),
        )
        self.connection.commit()

    def record_generation(
        self,
        run_id: str,
        metrics: GenerationMetrics,
        world: dict[Position, AgentState],
        encounters: tuple[EncounterRecord, ...],
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO generations (
                run_id, generation, live_cells, births, deaths, survivors,
                cooperation_rate, average_score, total_encounters, total_rounds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                metrics.generation,
                metrics.live_cells,
                metrics.births,
                metrics.deaths,
                metrics.survivors,
                metrics.cooperation_rate,
                metrics.average_score,
                metrics.total_encounters,
                metrics.total_rounds,
            ),
        )

        self.connection.execute(
            "DELETE FROM cells WHERE run_id = ? AND generation = ?",
            (run_id, metrics.generation),
        )
        self.connection.execute(
            "DELETE FROM memories WHERE run_id = ? AND generation = ?",
            (run_id, metrics.generation),
        )
        for position, agent in world.items():
            self.connection.execute(
                """
                INSERT INTO cells (
                    run_id, generation, x, y, agent_id, lineage_id, strategy,
                    base_prompt, policy_prompt, cooperation_bias, retaliation_bias,
                    forgiveness_bias, memory_weight, prompt_style, total_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    metrics.generation,
                    position.x,
                    position.y,
                    agent.agent_id,
                    agent.lineage_id,
                    agent.strategy,
                    agent.base_prompt,
                    agent.policy_prompt,
                    agent.prompt_traits.cooperation_bias,
                    agent.prompt_traits.retaliation_bias,
                    agent.prompt_traits.forgiveness_bias,
                    agent.prompt_traits.memory_weight,
                    agent.prompt_traits.prompt_style,
                    agent.total_score,
                ),
            )
            for memory in agent.opponent_memories.values():
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO memories (
                        run_id, generation, agent_id, opponent_agent_id, opponent_lineage_id,
                        encounters, rounds_played, self_cooperations, self_defections,
                        opponent_cooperations, opponent_defections, cumulative_score_delta,
                        last_generation, last_self_action, last_opponent_action
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        metrics.generation,
                        agent.agent_id,
                        memory.opponent_agent_id,
                        memory.opponent_lineage_id,
                        memory.encounters,
                        memory.rounds_played,
                        memory.self_cooperations,
                        memory.self_defections,
                        memory.opponent_cooperations,
                        memory.opponent_defections,
                        memory.cumulative_score_delta,
                        memory.last_generation,
                        memory.last_self_action,
                        memory.last_opponent_action,
                    ),
                )

        for encounter_index, encounter in enumerate(encounters):
            encounter_id = f"{run_id}:g{metrics.generation}:e{encounter_index:05d}"
            self.connection.execute(
                """
                INSERT OR REPLACE INTO encounters (
                    encounter_id, run_id, generation, left_agent_id, right_agent_id,
                    left_recognized_agent_id, left_recognized_lineage_id,
                    left_recognition_confidence, left_memory_summary,
                    right_recognized_agent_id, right_recognized_lineage_id,
                    right_recognition_confidence, right_memory_summary,
                    left_x, left_y, right_x, right_y
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    encounter_id,
                    run_id,
                    metrics.generation,
                    encounter.left_agent_id,
                    encounter.right_agent_id,
                    encounter.left_recognition.matched_agent_id,
                    encounter.left_recognition.matched_lineage_id,
                    encounter.left_recognition.confidence,
                    encounter.left_recognition.memory_summary,
                    encounter.right_recognition.matched_agent_id,
                    encounter.right_recognition.matched_lineage_id,
                    encounter.right_recognition.confidence,
                    encounter.right_recognition.memory_summary,
                    encounter.left_position.x,
                    encounter.left_position.y,
                    encounter.right_position.x,
                    encounter.right_position.y,
                ),
            )
            for round_record in encounter.rounds:
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO rounds (
                        encounter_id, round_index, left_action, right_action, left_payoff, right_payoff
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        encounter_id,
                        round_record.round_index,
                        round_record.left_action,
                        round_record.right_action,
                        round_record.left_payoff,
                        round_record.right_payoff,
                    ),
                )

        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def export_metrics_csv(db_path: Path, output_path: Path | None = None) -> Path:
    resolved_output = output_path or db_path.with_name("metrics.csv")
    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        """
        SELECT generation, live_cells, births, deaths, survivors,
               cooperation_rate, average_score, total_encounters, total_rounds
        FROM generations
        ORDER BY generation ASC
        """
    ).fetchall()
    connection.close()

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "generation",
                "live_cells",
                "births",
                "deaths",
                "survivors",
                "cooperation_rate",
                "average_score",
                "total_encounters",
                "total_rounds",
            ]
        )
        writer.writerows(rows)
    return resolved_output
