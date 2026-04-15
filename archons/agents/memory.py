from __future__ import annotations

from archons.core.models import AgentState, EncounterRecord, EncounterSide, OpponentMemory, RecognitionSnapshot


def build_recognition_snapshot(agent: AgentState, opponent: AgentState) -> RecognitionSnapshot:
    memory = agent.opponent_memories.get(opponent.agent_id)
    if memory is None:
        return RecognitionSnapshot(
            matched_agent_id=None,
            matched_lineage_id=None,
            confidence=0.0,
            memory_summary="No prior memory.",
        )
    return RecognitionSnapshot(
        matched_agent_id=memory.opponent_agent_id,
        matched_lineage_id=memory.opponent_lineage_id,
        confidence=1.0,
        memory_summary=memory.summary(),
    )


def update_agent_memory(
    agent: AgentState,
    opponent: AgentState,
    encounter: EncounterRecord,
    side: EncounterSide,
) -> None:
    prior = agent.opponent_memories.get(opponent.agent_id)
    rounds_played = len(encounter.rounds)
    self_cooperations = 0
    self_defections = 0
    opponent_cooperations = 0
    opponent_defections = 0
    score_delta = 0
    last_self_action = None
    last_opponent_action = None

    for round_record in encounter.rounds:
        if side == "left":
            self_action = round_record.left_action
            opponent_action = round_record.right_action
            score_delta += round_record.left_payoff - round_record.right_payoff
        else:
            self_action = round_record.right_action
            opponent_action = round_record.left_action
            score_delta += round_record.right_payoff - round_record.left_payoff

        self_cooperations += int(self_action == "C")
        self_defections += int(self_action == "D")
        opponent_cooperations += int(opponent_action == "C")
        opponent_defections += int(opponent_action == "D")
        last_self_action = self_action
        last_opponent_action = opponent_action

    agent.opponent_memories[opponent.agent_id] = OpponentMemory(
        opponent_agent_id=opponent.agent_id,
        opponent_lineage_id=opponent.lineage_id,
        encounters=(prior.encounters if prior else 0) + 1,
        rounds_played=(prior.rounds_played if prior else 0) + rounds_played,
        self_cooperations=(prior.self_cooperations if prior else 0) + self_cooperations,
        self_defections=(prior.self_defections if prior else 0) + self_defections,
        opponent_cooperations=(prior.opponent_cooperations if prior else 0) + opponent_cooperations,
        opponent_defections=(prior.opponent_defections if prior else 0) + opponent_defections,
        cumulative_score_delta=(prior.cumulative_score_delta if prior else 0) + score_delta,
        last_generation=encounter.generation,
        last_self_action=last_self_action,
        last_opponent_action=last_opponent_action,
    )
