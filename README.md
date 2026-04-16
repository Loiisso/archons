# Archons

Archons is a local-first simulation that combines repeated Prisoner's Dilemma with Conway's Game of Life. The current baseline implements a deterministic simulation core, SQLite run logging, and periodic visualization snapshots. The Ollama integration surface is included so the next step can swap deterministic policies for a small local Qwen model.

## Environment

- Local `.venv` in the workspace root
- Dependencies managed with `uv`
- Local Ollama expected at `http://localhost:11434`
- Baseline local model targets: `qwen2.5:0.5b`, `qwen2.5:1.5b`, or `qwen2.5:3b-instruct`

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

## Run The Baseline

```bash
uv run archons run --config experiments/baseline.yaml
```

## Run The Ollama Agent

Start Ollama locally and ensure the small Qwen model is present:

```bash
ollama serve
ollama pull qwen2.5:0.5b
uv run archons ollama-check --config experiments/ollama-qwen-smoke.yaml
uv run archons run --config experiments/ollama-qwen-smoke.yaml
```

Longer local runs are preconfigured for different model sizes:

```bash
uv run archons run --config experiments/ollama-qwen-0.5b-10x10-20.yaml
uv run archons run --config experiments/ollama-qwen-1.5b-10x10-20.yaml
uv run archons run --config experiments/ollama-qwen-10x10-20.yaml
```

Artifacts are written under `artifacts/` by default:

- `simulation.sqlite3` with run data
- `viz/` with PNG and ASCII snapshots every configured interval
- `effective_config.yaml` with the config used for the run
- `progress.log` with run start, per-generation progress, and completion lines for long simulations
- `profiles.csv` with per-generation timing breakdowns for encounter resolution, world advance, persistence, visualization, model latency, and instruction-following rate
- `decision_traces` table in SQLite showing prompt/response pairs, expected seed-consistent action, and whether each decision followed the strategy seed

Progress reporting is configurable in the experiment YAML under `simulation`:

- `print_progress`: enable or disable live console and file progress
- `progress_interval`: emit every `n` generations
- `progress_log_name`: filename written inside the run artifact directory

Live progress lines now include `instruction_rate`, which measures how often the chosen actions match the deterministic strategy seed implied by the agent prompt.

## Export Metrics

```bash
uv run archons export-metrics --db artifacts/<run-id>/simulation.sqlite3
uv run archons export-profiles --db artifacts/<run-id>/simulation.sqlite3
```
