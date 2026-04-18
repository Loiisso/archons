# Archons

Archons is a local-first simulation that combines repeated Prisoner's Dilemma with Conway's Game of Life. The current baseline implements a deterministic simulation core, SQLite run logging, and periodic visualization snapshots. Policy engines currently include deterministic strategies, a local Ollama backend, and an OpenAI SDK backend with structured outputs.

## Environment

- Local `.venv` in the workspace root
- Dependencies managed with `uv`
- Local Ollama expected at `http://localhost:11434`
- Baseline local model targets: `qwen2.5:0.5b`, `qwen2.5:1.5b`, or `qwen2.5:3b-instruct`

## Setup

```bash
uv sync
cp template.env .env
```

`template.env` is the checked-in starter file. Copy it to `.env` once at the beginning. The CLI loads `.env` automatically, and `.env` is gitignored so local secrets do not end up in the repo.

The default `OPENAI_API_KEY=sk-none` placeholder is intentional. The OpenAI Python SDK requires some API key value to be present before it will initialize a client, even if you point `base_url` at a local OpenAI-compatible server. For a local server that does not validate bearer tokens, the placeholder is enough.

You only need to change `OPENAI_API_KEY` when one of these is true:

- you are calling the real OpenAI API
- you are calling another OpenAI-compatible server that actually validates the bearer token
- you want to use a different key than the one currently in `.env`

## Run The Baseline

```bash
uv run archons run --config experiments/baseline.yaml
```

## Run The Ollama Agent

Start Ollama locally and ensure the small Qwen model is present:

```bash
ollama serve
ollama pull qwen2.5:0.5b
uv run archons policy-check --config experiments/ollama-qwen-smoke.yaml
uv run archons run --config experiments/ollama-qwen-smoke.yaml
```

Longer local runs are preconfigured for different model sizes:

```bash
uv run archons run --config experiments/ollama-qwen-0.5b-10x10-20.yaml
uv run archons run --config experiments/ollama-qwen-1.5b-10x10-20.yaml
uv run archons run --config experiments/ollama-qwen-10x10-20.yaml
```

## Run The OpenAI Agent

Use `agents.backend: openai` and configure `agents.openai`. For the hosted OpenAI API, replace the placeholder `OPENAI_API_KEY` in `.env` with a real key first.

Minimal config shape:

```yaml
agents:
  backend: openai
  fallback_strategy: tit_for_tat
  openai:
    model: gpt-4o-mini
    temperature: 0.1
    require_model: false
    fallback_on_error: false
```

Then verify and run:

```bash
uv run archons policy-check --config <your-openai-config>.yaml
uv run archons run --config <your-openai-config>.yaml
```

## Run A Local OpenAI-Compatible Model

If you want a local OpenAI-compatible server instead of the hosted API, you can run an MLX model and point `agents.openai.base_url` at it.

In this setup, keeping `OPENAI_API_KEY=sk-none` in `.env` is usually sufficient. It exists to satisfy the OpenAI SDK client initialization, not because the local MLX server needs a real OpenAI credential.

Download a model somewhere with enough free disk space:

```bash
uvx --from huggingface_hub hf download mlx-community/Qwen3.6-35B-A3B-mxfp4 --local-dir ./mlx-community/Qwen3.6-35B-A3B-mxfp4
```

Start the OpenAI-compatible server:

```bash
uvx -w mlx-lm python -m mlx_lm server --max-tokens 131072 --model ./mlx-community/Qwen3.6-35B-A3B-mxfp4
```

Confirm the model id exposed by the server:

```bash
curl -s http://localhost:8080/v1/models | jq -r '.data[0].id'
```

Use that value as `agents.openai.model` in your YAML config and set the base URL to the local server:

```yaml
agents:
  backend: openai
  fallback_strategy: tit_for_tat
  openai:
    base_url: http://localhost:8080/v1
    model: <model-id-from-v1-models>
    temperature: 0.1
    require_model: false
    fallback_on_error: false
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
