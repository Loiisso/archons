from pathlib import Path

from archons.analysis.viz import PeriodicVisualizer
from archons.config import (
    AgentsConfig,
    ExperimentConfig,
    OutputConfig,
    PromptEvolutionConfig,
    SimulationConfig,
    VisualizationConfig,
    WorldConfig,
)
from archons.core.simulation import SimulationRunner
from archons.storage.sqlite_store import RunStore


def test_offspring_mutates_prompt_traits_within_bounds() -> None:
    experiment = ExperimentConfig(
        world=WorldConfig(width=4, height=4, toroidal=False, initial_alive_probability=0.1),
        simulation=SimulationConfig(generations=1, seed=5),
        visualization=VisualizationConfig(enabled=False, interval=10, render_ascii=False),
        output=OutputConfig(root_dir="artifacts/tests"),
        agents=AgentsConfig(
            prompt_evolution=PromptEvolutionConfig(
                mutation_rate=1.0,
                mutation_step=0.2,
                style_options=["balanced", "cautious", "opportunistic", "vengeful"],
            )
        ),
    )
    run_dir = Path("artifacts/tests/prompt-evolution")
    store = RunStore(run_dir / "simulation.sqlite3")
    visualizer = PeriodicVisualizer(run_dir / "viz", experiment.visualization, experiment.world)
    runner = SimulationRunner(experiment=experiment, run_dir=run_dir, store=store, visualizer=visualizer)

    parent = runner._new_agent(strategy="tit_for_tat")
    child = runner._new_agent(strategy=parent.strategy, lineage_id=parent.lineage_id, parent=parent)
    store.close()

    assert child.lineage_id == parent.lineage_id
    assert child.agent_id != parent.agent_id
    assert child.base_prompt == parent.base_prompt
    assert child.policy_prompt != parent.policy_prompt
    assert child.prompt_traits != parent.prompt_traits
    assert 0.0 <= child.prompt_traits.cooperation_bias <= 1.0
    assert 0.0 <= child.prompt_traits.retaliation_bias <= 1.0
    assert 0.0 <= child.prompt_traits.forgiveness_bias <= 1.0
    assert 0.0 <= child.prompt_traits.memory_weight <= 1.0