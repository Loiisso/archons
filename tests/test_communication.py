from pathlib import Path

from archons.analysis.viz import PeriodicVisualizer
from archons.config import (
    AgentsConfig,
    CommunicationConfig,
    ExperimentConfig,
    GameConfig,
    OutputConfig,
    SimulationConfig,
    VisualizationConfig,
    WorldConfig,
)
from archons.core.models import AgentState, DecisionTrace, GenerationMetrics, MessageRecord, Position, RecognitionSnapshot
from archons.core.simulation import SimulationRunner
from archons.llm.ollama import DecisionResult, MessageResult
from archons.storage.sqlite_store import RunStore


class RecordingPolicy:
    def __init__(self) -> None:
        self.received_messages: list[tuple[str, str | None, str | None]] = []

    def ensure_available(self) -> None:
        return None

    def generate_message(
        self,
        agent: AgentState,
        opponent: AgentState,
        prior_rounds,
        side,
        recognition: RecognitionSnapshot,
        max_chars: int,
    ) -> MessageResult:
        return MessageResult(
            message_text=f"signal-from-{agent.agent_id}"[:max_chars],
            intent="promise",
            backend="ollama",
            confidence=0.9,
            prompt_text=f"prompt-for-{agent.agent_id}",
            response_text='{"message":"ok","intent":"promise","confidence":0.9}',
        )

    def choose_action(
        self,
        agent: AgentState,
        opponent: AgentState,
        prior_rounds,
        side,
        recognition: RecognitionSnapshot,
        incoming_message: str | None = None,
        incoming_intent: str | None = None,
    ) -> DecisionResult:
        self.received_messages.append((agent.agent_id, incoming_message, incoming_intent))
        return DecisionResult(
            action="C",
            backend="ollama",
            confidence=1.0,
            reasoning_summary="Accepted the pre-encounter signal.",
            prompt_text=f"incoming_message={incoming_message};incoming_intent={incoming_intent}",
            response_text='{"action":"C","confidence":1.0,"reasoning_summary":"Accepted the pre-encounter signal."}',
        )


def test_pre_encounter_messages_are_injected_and_persisted(tmp_path) -> None:
    experiment = ExperimentConfig(
        world=WorldConfig(width=4, height=4, toroidal=False, initial_alive_probability=0.1),
        game=GameConfig(rounds_per_encounter=1, communication=CommunicationConfig(enabled=True, message_max_chars=64)),
        simulation=SimulationConfig(generations=1, seed=3, print_progress=False),
        visualization=VisualizationConfig(enabled=False, interval=10, render_ascii=False),
        output=OutputConfig(root_dir=str(tmp_path / "artifacts")),
        agents=AgentsConfig(backend="ollama"),
    )
    run_dir = tmp_path / "communication"
    store = RunStore(run_dir / "simulation.sqlite3")
    visualizer = PeriodicVisualizer(run_dir / "viz", experiment.visualization, experiment.world)
    runner = SimulationRunner(experiment=experiment, run_dir=run_dir, store=store, visualizer=visualizer)
    runner.ollama_policy = RecordingPolicy()

    left_agent = AgentState("agent_left", "lineage_left", "tit_for_tat")
    right_agent = AgentState("agent_right", "lineage_right", "always_defect")
    encounter = runner._play_encounter(
        generation=1,
        left_position=Position(0, 0),
        right_position=Position(1, 0),
        left_agent=left_agent,
        right_agent=right_agent,
    )

    assert len(encounter.messages) == 2
    assert encounter.messages[0].message_text == "signal-from-agent_left"
    assert encounter.messages[1].message_text == "signal-from-agent_right"
    assert runner.ollama_policy.received_messages == [
        ("agent_left", "signal-from-agent_right", "promise"),
        ("agent_right", "signal-from-agent_left", "promise"),
    ]

    store.create_run("run-1", "communication-test", "name: communication-test\n", "2026-04-16T00:00:00Z")
    metrics = GenerationMetrics(
        generation=1,
        live_cells=2,
        births=0,
        deaths=0,
        survivors=2,
        cooperation_rate=1.0,
        average_score=0.0,
        total_encounters=1,
        total_rounds=1,
    )
    world = {Position(0, 0): left_agent, Position(1, 0): right_agent}
    store.record_generation(run_id="run-1", metrics=metrics, world=world, encounters=(encounter,))
    message_rows = store.connection.execute(
        "SELECT sender_agent_id, recipient_agent_id, message_text, intent FROM messages ORDER BY sender_side ASC"
    ).fetchall()

    assert message_rows == [
        ("agent_left", "agent_right", "signal-from-agent_left", "promise"),
        ("agent_right", "agent_left", "signal-from-agent_right", "promise"),
    ]
    store.close()