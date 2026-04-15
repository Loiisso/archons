from __future__ import annotations

from pathlib import Path

import typer

from archons.analysis.reporting import export_metrics_csv, export_profiles_csv
from archons.config import load_config
from archons.core.simulation import SimulationRunner
from archons.llm.ollama import OllamaPolicy

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


@app.command("export-profiles")
def export_profiles(
    db: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    output: Path | None = typer.Option(None, dir_okay=False),
) -> None:
    output_path = export_profiles_csv(db_path=db, output_path=output)
    typer.echo(f"Profiles exported to {output_path}")


@app.command("ollama-check")
def ollama_check(config: Path = typer.Option(..., exists=True, dir_okay=False, readable=True)) -> None:
    experiment = load_config(config)
    policy = OllamaPolicy(
        config=experiment.agents.ollama,
        fallback_strategy=experiment.agents.fallback_strategy,
    )
    policy.ensure_available()
    typer.echo(
        "Ollama is reachable and model is available: "
        f"{experiment.agents.ollama.model} at {experiment.agents.ollama.base_url}"
    )
