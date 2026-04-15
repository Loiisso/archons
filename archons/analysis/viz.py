from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from archons.config import VisualizationConfig, WorldConfig
from archons.core.models import AgentState, GenerationMetrics, Position

STRATEGY_COLORS = {
    "dead": "#f4efe6",
    "always_cooperate": "#5b8def",
    "always_defect": "#db5a42",
    "tit_for_tat": "#2a9d8f",
    "grim_trigger": "#8f5cc1",
}

STRATEGY_INDEX = {
    "dead": 0,
    "always_cooperate": 1,
    "always_defect": 2,
    "tit_for_tat": 3,
    "grim_trigger": 4,
}


class PeriodicVisualizer:
    def __init__(self, output_dir: Path, config: VisualizationConfig, world: WorldConfig) -> None:
        self.output_dir = output_dir
        self.config = config
        self.world = world
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._colormap = ListedColormap([STRATEGY_COLORS[name] for name in STRATEGY_INDEX])

    def maybe_render(
        self,
        generation: int,
        world: dict[Position, AgentState],
        metrics: GenerationMetrics,
    ) -> None:
        if not self.config.enabled:
            return
        if generation != 0 and generation % self.config.interval != 0:
            return

        self._render_png(generation=generation, world=world, metrics=metrics)
        if self.config.render_ascii:
            self._render_ascii(generation=generation, world=world)

    def _render_png(
        self,
        generation: int,
        world: dict[Position, AgentState],
        metrics: GenerationMetrics,
    ) -> None:
        grid = [
            [STRATEGY_INDEX["dead"] for _ in range(self.world.width)]
            for _ in range(self.world.height)
        ]
        for position, agent in world.items():
            grid[position.y][position.x] = STRATEGY_INDEX[agent.strategy]

        figure, axis = plt.subplots(
            figsize=(max(self.world.width * 0.35, 6.0), max(self.world.height * 0.35, 6.0))
        )
        axis.imshow(grid, cmap=self._colormap, vmin=0, vmax=len(STRATEGY_INDEX) - 1)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            " | ".join(
                [
                    f"generation={generation}",
                    f"live={metrics.live_cells}",
                    f"births={metrics.births}",
                    f"deaths={metrics.deaths}",
                    f"coop_rate={metrics.cooperation_rate:.2f}",
                ]
            )
        )

        handles = [
            Patch(color=STRATEGY_COLORS[name], label=name)
            for name in STRATEGY_INDEX
            if name != "dead"
        ]
        axis.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

        output_path = self.output_dir / f"generation_{generation:04d}.png"
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)

    def _render_ascii(self, generation: int, world: dict[Position, AgentState]) -> None:
        glyphs = {
            "always_cooperate": "C",
            "always_defect": "D",
            "tit_for_tat": "T",
            "grim_trigger": "G",
        }
        rows: list[str] = []
        for y in range(self.world.height):
            row = []
            for x in range(self.world.width):
                agent = world.get(Position(x=x, y=y))
                row.append(glyphs.get(agent.strategy, ".") if agent else ".")
            rows.append("".join(row))

        output_path = self.output_dir / f"generation_{generation:04d}.txt"
        output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
