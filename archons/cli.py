from __future__ import annotations

from pathlib import Path

import typer

from archons.analysis.reporting import export_metrics_csv
from archons.config import load_config
from archons.core.simulation import SimulationRunner

app = typer.Typer(add_completion=False, help="Run Archons simulation experiments.")


@app.command()
def run(config: Path = typer.Option(..., exists=True, dir_okay=False, readable=True)) -> None:
    experiment = load_config(config)
    runner = SimulationRunner.from_config(experiment=experiment, config_path=config)
    summary = runner.run()
    typer.echo(f"Run complete: {summary.run_id}")
    typer.echo(f"Artifacts: {summary.run_dir}")
    typer.echo(
        "Final state: "
        f"generation={summary.final_generation} "
        f"live_cells={summary.final_live_cells} "
        f"agents_created={summary.total_agents_created}"
    )


@app.command("export-metrics")
def export_metrics(
    db: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    output: Path | None = typer.Option(None, dir_okay=False),
) -> None:
    output_path = export_metrics_csv(db_path=db, output_path=output)
    typer.echo(f"Metrics exported to {output_path}")
