from pathlib import Path

from archons.analysis.viz import PeriodicVisualizer
from archons.config import ExperimentConfig, OutputConfig, VisualizationConfig, WorldConfig
from archons.core.simulation import SimulationRunner
from archons.storage.sqlite_store import RunStore


def test_payoff_matrix_matches_prisoners_dilemma_defaults() -> None:
    experiment = ExperimentConfig(
        world=WorldConfig(width=4, height=4, toroidal=False, initial_alive_probability=0.1),
        visualization=VisualizationConfig(enabled=False, interval=10, render_ascii=False),
        output=OutputConfig(root_dir="artifacts/tests"),
    )
    run_dir = Path("artifacts/tests/payoffs")
    runner = SimulationRunner(
        experiment=experiment,
        run_dir=run_dir,
        store=RunStore(run_dir / "simulation.sqlite3"),
        visualizer=PeriodicVisualizer(run_dir / "viz", experiment.visualization, experiment.world),
    )

    assert runner._score_actions("C", "C") == (3, 3)
    assert runner._score_actions("C", "D") == (0, 5)
    assert runner._score_actions("D", "C") == (5, 0)
    assert runner._score_actions("D", "D") == (1, 1)
