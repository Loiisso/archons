# Archons

Archons is a local-first simulation that combines repeated Prisoner's Dilemma with Conway's Game of Life. The current baseline implements a deterministic simulation core, SQLite run logging, and periodic visualization snapshots. The Ollama integration surface is included so the next step can swap deterministic policies for a small local Qwen model.

## Environment

- Local `.venv` in the workspace root
- Dependencies managed with `uv`
- Local Ollama expected at `http://localhost:11434`
- Baseline model target: `qwen2.5:3b-instruct`

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

Artifacts are written under `artifacts/` by default:

- `simulation.sqlite3` with run data
- `viz/` with PNG and ASCII snapshots every configured interval
- `effective_config.yaml` with the config used for the run

## Export Metrics

```bash
uv run archons export-metrics --db artifacts/<run-id>/simulation.sqlite3
```
