from __future__ import annotations

from random import Random

from archons.config import PromptEvolutionConfig
from archons.core.models import PromptTraits, StrategyName

DEFAULT_BASE_PROMPTS: dict[StrategyName, str] = {
    "always_cooperate": "You are a civility-first agent. Default toward cooperation unless strong evidence says otherwise.",
    "always_defect": "You are a maximizer of short-term advantage. Default toward exploitation unless there is a clear reason not to.",
    "tit_for_tat": "You are a reciprocal agent. Begin generously, then mirror the opponent's demonstrated behavior.",
    "grim_trigger": "You are a guardrail-heavy agent. Cooperate until betrayed, then protect yourself aggressively.",
}

STRATEGY_TRAIT_SEEDS: dict[StrategyName, PromptTraits] = {
    "always_cooperate": PromptTraits(0.9, 0.2, 0.9, 0.4, "cautious"),
    "always_defect": PromptTraits(0.1, 0.9, 0.1, 0.3, "opportunistic"),
    "tit_for_tat": PromptTraits(0.7, 0.7, 0.6, 0.8, "balanced"),
    "grim_trigger": PromptTraits(0.6, 0.95, 0.15, 0.85, "vengeful"),
}


def seed_prompt_traits(strategy: StrategyName) -> PromptTraits:
    return STRATEGY_TRAIT_SEEDS[strategy]


def base_prompt_for_strategy(strategy: StrategyName) -> str:
    return DEFAULT_BASE_PROMPTS[strategy]


def mutate_prompt_traits(
    parent_traits: PromptTraits,
    random: Random,
    config: PromptEvolutionConfig,
) -> PromptTraits:
    cooperation_bias = _mutate_scalar(parent_traits.cooperation_bias, random, config)
    retaliation_bias = _mutate_scalar(parent_traits.retaliation_bias, random, config)
    forgiveness_bias = _mutate_scalar(parent_traits.forgiveness_bias, random, config)
    memory_weight = _mutate_scalar(parent_traits.memory_weight, random, config)
    prompt_style = parent_traits.prompt_style
    if random.random() < config.mutation_rate:
        prompt_style = random.choice(config.style_options)
    return PromptTraits(
        cooperation_bias=cooperation_bias,
        retaliation_bias=retaliation_bias,
        forgiveness_bias=forgiveness_bias,
        memory_weight=memory_weight,
        prompt_style=prompt_style,
    )


def render_policy_prompt(strategy: StrategyName, traits: PromptTraits, base_prompt: str) -> str:
    style_instruction = {
        "balanced": "Balance generosity with caution and update quickly from evidence.",
        "cautious": "Prefer stability, avoid overreacting, and value long-term cooperation.",
        "opportunistic": "Look for profitable defections when retaliation risk seems low.",
        "vengeful": "Treat betrayals as major signals and retaliate decisively once trust is broken.",
    }.get(traits.prompt_style, "Adapt to the opponent using your memory and the local game state.")
    return (
        f"{base_prompt}\n"
        f"Policy seed: {strategy}.\n"
        f"Prompt traits: cooperation_bias={traits.cooperation_bias:.2f}, "
        f"retaliation_bias={traits.retaliation_bias:.2f}, "
        f"forgiveness_bias={traits.forgiveness_bias:.2f}, "
        f"memory_weight={traits.memory_weight:.2f}, prompt_style={traits.prompt_style}.\n"
        f"Guidance: {style_instruction}\n"
        "Use recognition and retrieved memory when available, then choose exactly one action."
    )


def _mutate_scalar(value: float, random: Random, config: PromptEvolutionConfig) -> float:
    if random.random() >= config.mutation_rate:
        return value
    mutated = value + random.uniform(-config.mutation_step, config.mutation_step)
    return max(0.0, min(1.0, mutated))
