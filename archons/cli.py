from __future__ import annotations

import asyncio
from functools import wraps
from pathlib import Path

import typer
from dotenv import load_dotenv

from archons.analysis.reporting import export_metrics_csv, export_profiles_csv
from archons.config import load_config
from archons.core.simulation import SimulationRunner
from archons.llm.factory import build_policy_engine

load_dotenv(override=True)

app = typer.Typer(add_completion=False, help="Run Archons simulation experiments.")

syncify = lambda f: wraps(f)(lambda *args, **kwargs: asyncio.run(f(*args, **kwargs)))


@app.command()
@syncify
async def run(
    config: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
) -> None:
    experiment = load_config(config)
    runner = SimulationRunner.from_config(experiment=experiment, config_path=config)
    summary = await runner.run()
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


@app.command("policy-check")
@syncify
async def policy_check(
    config: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
) -> None:
    experiment = load_config(config)
    typer.echo(f"Policy backend: {experiment.agents.backend}")
    policy = build_policy_engine(experiment.agents)
    if policy is None:
        typer.echo("Policy engine: deterministic")
        typer.echo("No external policy service configured, so liveliness is implicit.")
        return
    await policy.ensure_available()
    typer.echo(_policy_liveliness_message(experiment))


def _policy_liveliness_message(experiment) -> str:
    if experiment.agents.backend == "ollama":
        config_obj = experiment.agents.ollama

    elif experiment.agents.backend == "openai":
        config_obj = experiment.agents.openai
    else:
        # should not happen
        config_obj = None
        typer.echo(f"Unexpected policy engine: {experiment.agents.backend}")

    return (
        "Policy engine is reachable: "
        f"{experiment.agents.backend} model={config_obj.model} "
        f"base_url={config_obj.base_url}"
    )
