from archons.analysis.viz import PeriodicVisualizer
from archons.config import ExperimentConfig, OutputConfig, SimulationConfig, VisualizationConfig, WorldConfig
from archons.core.models import AgentState, GenerationMetrics, Position
from archons.core.simulation import SimulationRunner
from archons.storage.sqlite_store import RunStore


def test_exact_id_memory_and_recognition_are_persisted(tmp_path) -> None:
    experiment = ExperimentConfig(
        world=WorldConfig(width=4, height=4, toroidal=False, initial_alive_probability=0.1),
        simulation=SimulationConfig(generations=1, seed=3),
        visualization=VisualizationConfig(enabled=False, interval=10, render_ascii=False),
        output=OutputConfig(root_dir=str(tmp_path / "artifacts")),
    )
    run_dir = tmp_path / "memory"
    store = RunStore(run_dir / "simulation.sqlite3")
    visualizer = PeriodicVisualizer(run_dir / "viz", experiment.visualization, experiment.world)
    runner = SimulationRunner(experiment=experiment, run_dir=run_dir, store=store, visualizer=visualizer)

    left_agent = AgentState("agent_left", "lineage_left", "always_cooperate")
    right_agent = AgentState("agent_right", "lineage_right", "always_defect")

    first = runner._play_encounter(
        generation=1,
        left_position=Position(1, 1),
        right_position=Position(1, 2),
        left_agent=left_agent,
        right_agent=right_agent,
    )
    assert first.left_recognition.confidence == 0.0
    assert first.left_recognition.matched_agent_id is None
    assert left_agent.opponent_memories[right_agent.agent_id].opponent_defections == 3

    second = runner._play_encounter(
        generation=2,
        left_position=Position(1, 1),
        right_position=Position(1, 2),
        left_agent=left_agent,
        right_agent=right_agent,
    )
    assert second.left_recognition.confidence == 1.0
    assert second.left_recognition.matched_agent_id == right_agent.agent_id
    assert right_agent.agent_id in second.left_recognition.memory_summary

    world = {
        Position(1, 1): left_agent,
        Position(1, 2): right_agent,
    }
    store.create_run("memory-test", "memory-test", "name: memory-test\n", "2026-04-15T00:00:00Z")
    store.record_generation(
        run_id="memory-test",
        metrics=GenerationMetrics(
            generation=2,
            live_cells=2,
            births=0,
            deaths=0,
            survivors=2,
            cooperation_rate=0.5,
            average_score=0.0,
            total_encounters=1,
            total_rounds=3,
        ),
        world=world,
        encounters=(second,),
    )

    memory_row = store.connection.execute(
        "SELECT opponent_agent_id, encounters, opponent_defections FROM memories WHERE agent_id = ?",
        (left_agent.agent_id,),
    ).fetchone()
    encounter_row = store.connection.execute(
        "SELECT left_recognized_agent_id, left_recognition_confidence FROM encounters WHERE encounter_id = ?",
        ("memory-test:g2:e00000",),
    ).fetchone()
    store.close()

    assert memory_row == (right_agent.agent_id, 2, 6)
    assert encounter_row == (right_agent.agent_id, 1.0)