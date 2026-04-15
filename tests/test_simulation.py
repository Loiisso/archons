from pathlib import Path

from archons.config import ExperimentConfig, GameConfig, OutputConfig, SimulationConfig, VisualizationConfig, WorldConfig
from archons.core.models import AgentState, Position
from archons.core.simulation import SimulationRunner
from archons.storage.sqlite_store import RunStore
from archons.analysis.viz import PeriodicVisualizer


def test_blinker_transitions_under_life_rules() -> None:
    experiment = ExperimentConfig(
        world=WorldConfig(width=5, height=5, toroidal=False, initial_alive_probability=0.1),
        simulation=SimulationConfig(generations=1, seed=3),
        visualization=VisualizationConfig(enabled=False, interval=10, render_ascii=False),
        output=OutputConfig(root_dir="artifacts/tests"),
    )
    run_dir = Path("artifacts/tests/blinker")
    store = RunStore(run_dir / "simulation.sqlite3")
    visualizer = PeriodicVisualizer(run_dir / "viz", experiment.visualization, experiment.world)
    runner = SimulationRunner(experiment=experiment, run_dir=run_dir, store=store, visualizer=visualizer)

    world = {
        Position(2, 1): AgentState("agent_1", "lineage_1", "always_cooperate"),
        Position(2, 2): AgentState("agent_2", "lineage_2", "always_cooperate"),
        Position(2, 3): AgentState("agent_3", "lineage_3", "always_cooperate"),
    }
    encounters = runner._resolve_encounters(world=world, generation=1)
    next_world, metrics = runner._advance_world(current_world=world, generation=1, encounters=encounters)

    assert metrics.live_cells == 3
    assert set(next_world) == {Position(1, 2), Position(2, 2), Position(3, 2)}
