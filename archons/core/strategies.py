from __future__ import annotations

from archons.core.models import Action, EncounterSide, RoundRecord, StrategyName


def choose_deterministic_action(
    strategy: StrategyName,
    prior_rounds: tuple[RoundRecord, ...],
    side: EncounterSide,
) -> Action:
    opponent_actions = tuple(_opponent_actions(prior_rounds=prior_rounds, side=side))

    if strategy == "always_cooperate":
        return "C"
    if strategy == "always_defect":
        return "D"
    if strategy == "tit_for_tat":
        return opponent_actions[-1] if opponent_actions else "C"
    if strategy == "grim_trigger":
        return "D" if "D" in opponent_actions else "C"
    raise ValueError(f"Unsupported strategy: {strategy}")


def _opponent_actions(
    prior_rounds: tuple[RoundRecord, ...],
    side: EncounterSide,
) -> list[Action]:
    if side == "left":
        return [round_record.right_action for round_record in prior_rounds]
    return [round_record.left_action for round_record in prior_rounds]
