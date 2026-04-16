import sqlite3

from archons.config import ExperimentConfig, OutputConfig, SimulationConfig, VisualizationConfig, WorldConfig
from archons.core.simulation import SimulationRunner


def test_generation_profiles_are_persisted_and_exported(tmp_path) -> None:
    config_path = tmp_path / "profiling.yaml"
    config_path.write_text("name: profiling-test\n", encoding="utf-8")

    experiment = ExperimentConfig(
        name="profiling-test",
        world=WorldConfig(width=5, height=5, toroidal=False, initial_alive_probability=0.6),
        simulation=SimulationConfig(generations=2, seed=5, print_progress=False),
        visualization=VisualizationConfig(enabled=False, interval=1, render_ascii=False),
        output=OutputConfig(root_dir=str(tmp_path / "artifacts")),
    )
    runner = SimulationRunner.from_config(experiment=experiment, config_path=config_path)

    summary = runner.run()
    db_path = summary.run_dir / "simulation.sqlite3"
    profiles_csv_path = summary.run_dir / "profiles.csv"

    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        """
        SELECT generation, total_seconds, resolve_encounters_seconds,
               advance_world_seconds, persist_seconds, visualize_seconds,
               overhead_seconds, decision_trace_count, instruction_followed_count,
               instruction_following_rate, ollama_latency_seconds
        FROM generation_profiles
        ORDER BY generation ASC
        """
    ).fetchall()
    connection.close()

    assert len(rows) == 3
    assert rows[0][0] == 0
    assert rows[-1][0] == 2
    assert all(row[1] >= 0.0 for row in rows)
    assert all(row[2] >= 0.0 for row in rows)
    assert all(row[3] >= 0.0 for row in rows)
    assert all(row[4] >= 0.0 for row in rows)
    assert all(row[5] >= 0.0 for row in rows)
    assert all(row[6] >= 0.0 for row in rows)
    assert all(row[7] >= 0 for row in rows)
    assert all(row[8] >= 0 for row in rows)
    assert all(0.0 <= row[9] <= 1.0 for row in rows)
    assert all(row[10] == 0.0 for row in rows)

    csv_text = profiles_csv_path.read_text(encoding="utf-8")
    assert "generation,total_seconds,resolve_encounters_seconds" in csv_text
    assert "instruction_followed_count,instruction_following_rate" in csv_text
    assert "\n2," in csv_text