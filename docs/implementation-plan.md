# Implementation Plan

## 1. Delivery Strategy

Build this in narrow phases so the deterministic core is correct before the LLM layer becomes complicated.

Implementation order:

1. local Python project setup with workspace-local `.venv`
2. deterministic grid and Prisoner's Dilemma engine
3. persistence and replay
4. Ollama-backed agent decisions
5. memory and recognition
6. prompt evolution
7. experiment runner and analysis outputs

## 2. Environment Baseline

Use a local `.venv` in the project root and manage dependencies with `uv`.

Suggested setup:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --upgrade pip
```

Initial runtime target:

- Python 3.11+
- Ollama running locally
- Small Qwen instruct model available in Ollama, preferably `qwen2.5:3b-instruct`
- SQLite for storage

Recommended first dependencies:

- `pydantic` for schemas and config models
- `pyyaml` for experiment configs
- `sqlalchemy` or `sqlite3` for persistence
- `typer` for CLI
- `pytest` for tests
- `httpx` for Ollama HTTP calls
- `pandas` for metrics export if needed

## 3. Proposed Repository Layout

```text
archons/
  .venv/
  docs/
  src/
    archons/
      core/
      agents/
      llm/
      evolution/
      storage/
      analysis/
      experiments/
      cli/
  tests/
  pyproject.toml
```

## 4. Phase 0: Project Skeleton

Objective: create a stable local development base.

Tasks:

1. Create `.venv` locally in the workspace root.
2. Add `pyproject.toml` with runtime and dev dependencies.
3. Add a minimal package under `src/archons`.
4. Add `pytest` configuration and a first smoke test.
5. Add `README.md` with local setup and Ollama assumptions.
6. Add a periodic visualization output path for baseline runs.

Exit criteria:

- project installs in editable mode inside `.venv`
- tests run locally
- `python -m archons --help` or equivalent CLI entry point works

## 5. Phase 1: Deterministic Simulation Core

Objective: implement the hybrid world without any LLM dependency.

Tasks:

1. Define core models:
   - `Position`
   - `Cell`
   - `WorldState`
   - `AgentInstance`
   - `EncounterResult`
   - `GenerationResult`
2. Implement neighborhood lookup and occupancy updates.
3. Implement canonical B3/S23 birth and survival logic.
4. Implement repeated Prisoner's Dilemma payoff engine.
5. Add deterministic baseline strategies:
   - always cooperate
   - always defect
   - tit-for-tat
   - grim trigger
6. Implement one generation loop that:
   - finds live neighbors
   - resolves encounters
   - computes payoff
   - updates births and deaths

Exit criteria:

- deterministic tests cover payoff matrix, neighborhood rules, and generation transitions
- a small seeded world can run for many generations without LLM calls

## 6. Phase 2: Persistence And Replay

Objective: make every run inspectable.

Tasks:

1. Define storage schema for runs, agents, generations, rounds, and memories.
2. Persist world snapshots per generation.
3. Persist encounter and round data.
4. Add run metadata and experiment config snapshot storage.
5. Build a replay loader for one run.

Exit criteria:

- one completed run can be reloaded from SQLite
- analysis can be performed without rerunning the simulation

## 7. Phase 3: Ollama Integration

Objective: replace baseline policies with model-backed decisions behind an adapter.

Tasks:

1. Implement Ollama client adapter using HTTP.
2. Define request and response schemas.
3. Create prompt builder that combines:
   - base prompt
   - policy traits
   - opponent summary
   - current round state
   - retrieved memories
4. Parse model output into strict structured actions.
5. Add retries, timeouts, and fallback policies.
6. Add logging for prompt, response, latency, and parse failures.

Exit criteria:

- one agent can choose `C` or `D` via local Ollama with schema validation
- invalid responses degrade gracefully to a fallback strategy

## 8. Phase 4: Memory And Recognition

Objective: make agents stateful across repeated encounters and generations.

Tasks:

1. Implement episodic memory write path after each encounter.
2. Implement semantic summary generation with a deterministic first pass.
3. Implement exact ID lookup for known opponents.
4. Add lightweight recognition signatures using structured features such as:
   - recent move ratios
   - betrayal count
   - response latency bucket
   - style markers if exposed
5. Add top-k retrieval API for current decision context.
6. Inject retrieved summaries into prompts.

Exit criteria:

- agents retrieve relevant prior encounters with the same opponent
- decision traces show which memories affected each action

## 9. Phase 5: Prompt Traits And Evolution

Objective: allow behavior to evolve without losing inspectability.

Tasks:

1. Represent policy prompts as structured trait templates.
2. Define inheritable traits such as:
   - cooperation prior
   - retaliation threshold
   - forgiveness rate
   - exploration rate
   - memory reliance
3. Implement parent selection for births.
4. Implement rule-based mutation over trait values.
5. Rebuild child prompt text from trait templates.
6. Persist prompt versions and diffs.

Exit criteria:

- offspring prompts differ in controlled, explainable ways
- lineage history is queryable after a run

## 10. Phase 6: Experiment Configuration

Objective: make runs reproducible and easy to compare.

Tasks:

1. Define YAML config schema for world, agents, game, LLM, and storage.
2. Add validation at startup.
3. Add experiment IDs and output directories.
4. Support seed control for deterministic parts of the simulation.
5. Store effective config with each run.

Exit criteria:

- the same config reproduces the same deterministic behavior outside model variance
- runs are attributable to config files, not shell history

## 11. Phase 7: Analysis And Reporting

Objective: answer the actual research questions.

Tasks:

1. Export per-generation metrics.
2. Export lineage summaries.
3. Compare memory-enabled vs memory-disabled runs.
4. Compare recognition-enabled vs recognition-disabled runs.
5. Add a simple text or notebook report generator.

Exit criteria:

- one command produces enough artifacts to compare experiments offline

## 12. Test Strategy

Separate tests by confidence level.

Unit tests:

- payoff engine
- Life update rules
- parent selection
- mutation rules
- memory retrieval ranking
- Ollama response parsing

Integration tests:

- deterministic non-LLM simulation run
- LLM-backed decision with mocked Ollama responses
- SQLite persistence and replay

Golden tests:

- fixed config and mocked responses produce stable outputs

## 13. Initial Milestone Sequence

If you want the shortest path to a usable prototype, implement in this order:

1. project scaffold plus `.venv`
2. deterministic simulation with non-LLM strategies
3. SQLite persistence
4. single Ollama-backed strategy agent
5. exact-ID memory retrieval
6. rule-based prompt trait mutation
7. metrics export

That sequence gets you a research-capable baseline before recognition signatures and richer evolution.

## 14. Recommended First Sprint

Sprint goal: prove the hybrid loop works without depending on complex model behavior.

Build in the first sprint:

1. project scaffold
2. world grid
3. canonical Life update
4. iterated Prisoner's Dilemma engine
5. two deterministic policies
6. SQLite run logging
7. one CLI command to run 20 generations from a YAML config
8. periodic visualization snapshots every configurable `n` generations

Do not include in the first sprint:

- prompt evolution
- semantic memory summarization by LLM
- visualization UI beyond periodic snapshot export
- heterogeneous models

## 15. Example CLI Surface

Target commands:

```bash
python -m archons run --config experiments/baseline.yaml
python -m archons replay --run-id <id>
python -m archons export-metrics --run-id <id>
```

## 16. Immediate Next Build Steps

Concrete next actions for implementation:

1. create the Python project in this workspace using a local `.venv`
2. add the deterministic simulation core and tests
3. add SQLite persistence
4. add the Ollama adapter only after the deterministic test suite is stable

That order keeps the hardest-to-debug stochastic component from contaminating the core model too early.