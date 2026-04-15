# Hybrid LLM Society Simulator Design

## 1. Goal

Build a local-first simulation that combines:

- repeated Prisoner's Dilemma as the strategic interaction layer
- Conway's Game of Life as the spatial population dynamics layer
- LLM-driven agents whose behavior is controlled by prompts that can evolve over time
- persistent agent memory, including recognition of previously encountered agents

The system should run against a local Ollama instance using open-source models and should be structured so that the expensive LLM loop is isolated from the deterministic simulation core.

## 2. Product Outcome

The first usable version should let you:

- create a 2D world populated by LLM agents
- run discrete generations of interaction and spatial updates
- persist world state, agent memory, and interaction history
- inspect why an agent cooperated or defected
- compare different prompt lineages and memory strategies
- replay runs deterministically except for the LLM decision boundary

## 3. Assumptions

- Runtime is local Python in a workspace-local `.venv`, with dependencies managed by `uv`.
- Model serving is local Ollama over HTTP.
- Models are open-source, small enough to run locally, and can be swapped per experiment.
- The simulation is research-oriented, not latency-critical in the first iteration.
- Agent memory must outlive a single game round and should survive across generations when configured.

Recommended local baseline model:

- Ollama `qwen2.5:3b-instruct` or the closest small Qwen instruct variant available locally

## 4. Core Design Choice

Use Conway's Game of Life to govern where agents exist, and use repeated Prisoner's Dilemma to govern how successful those agents are.

Default hybrid rule set:

1. Each live cell contains one agent.
2. Each generation, a live agent plays repeated Prisoner's Dilemma against each live neighbor.
3. The agent accumulates payoff, memory entries, and recognition updates from those interactions.
4. After interactions resolve, the grid advances using a Life-derived birth/survival rule.
5. When a new cell is born, it receives a newly instantiated agent derived from local parents or neighborhood statistics.
6. Birth inheritance carries prompt lineage, some memory traits, and optional prompt mutation.

This keeps the spatial update interpretable while allowing strategy success to matter through reproduction and inheritance.

## 5. Simulation Model

### 5.1 World

- Grid: 2D matrix, finite by default, optional toroidal wrap.
- Cell states:
  - dead: empty
  - alive: contains an `AgentInstance`
- Neighborhood: Moore neighborhood by default (8 surrounding cells).
- Tick hierarchy:
  - round: one Prisoner's Dilemma choice against one opponent
  - encounter: multiple rounds between a pair of neighboring agents
  - generation: all encounters plus one world update

### 5.2 Life Rule Integration

Preserve the recognizable Life rule shape, but attach semantics to birth/survival:

- Survival: live cells survive based on neighborhood occupancy under B3/S23-style rules.
- Birth: new agents spawn in dead cells with exactly three live neighbors by default.
- Parent selection for births:
  - choose the three neighbors that caused birth
  - rank them by recent cumulative payoff
  - either select one parent or synthesize from multiple parents
- Death: dead cells remove the active agent instance, but archived memory and lineage remain in persistence.

This approach keeps the world visually and conceptually close to Game of Life while making strategic success affect what gets born next.

### 5.3 Prisoner's Dilemma Layer

Each neighboring live pair plays an iterated game for `n_rounds` per generation.

Default payoff matrix:

| Self / Other | Cooperate | Defect |
| --- | --- | --- |
| Cooperate | R=3 | S=0 |
| Defect | T=5 | P=1 |

Configurable parameters:

- rounds per encounter
- simultaneous vs turn-aware interpretation
- whether agents see cumulative history for the current opponent only or summary statistics for all opponents
- whether communication is allowed before choosing

## 6. Agent Model

Each agent should be represented as a structured object, not just a prompt string.

Suggested fields:

- `agent_id`: immutable unique identifier for this instantiated agent
- `lineage_id`: shared identifier across descendants
- `generation_born`
- `base_prompt`: stable identity or ideology prompt
- `policy_prompt`: current decision prompt after mutation/evolution
- `memory_profile`: config for what the agent remembers
- `recognition_profile`: config for how the agent identifies others
- `strategy_state`: non-natural-language state such as scores, trust values, or counters
- `fitness_state`: cumulative and recent payoff metrics
- `model_spec`: Ollama model name and decoding parameters

The important design constraint is that natural-language prompting and structured state should coexist. Do not force all memory into prompt text.

## 7. Memory And Recognition

### 7.1 Memory Tiers

Use three memory layers.

1. Episodic memory
   Stores concrete encounters with specific agents.
   Example: "Agent X defected after two cooperative rounds in generation 14."

2. Semantic memory
   Stores compressed beliefs and summaries.
   Example: "Neighbors with terse language tend to defect."

3. Lineage memory
   Stores inherited priors from parent prompts or prior generations.
   Example: "Our lineage prefers forgiving retaliation."

### 7.2 Recognition Model

Recognition should not rely only on exact IDs because future experiments may hide or mutate them.

Default recognition stack:

- hard identity: exact `agent_id` when visible
- lineage identity: exact `lineage_id` when visible
- signature identity: hashed or embedded behavioral profile based on prior actions, style markers, or prompt metadata
- confidence score: how sure the agent is that this opponent matches a known prior actor

Recognition pipeline:

1. Observe opponent features exposed by the environment.
2. Match exact IDs if available.
3. Otherwise compare against stored recognition signatures.
4. Retrieve top-k relevant memories.
5. Provide those memories and confidence to the decision prompt.

### 7.3 Memory Retrieval Strategy

Use retrieval rather than dumping the full history into context.

Recommended retrieval inputs:

- current opponent identity or signature
- last few moves in the current encounter
- global stress signals such as low fitness or crowded neighborhood
- lineage priors

Recommended outputs to prompt:

- short opponent summary
- 1 to 5 top episodic memories
- current trust or retaliation score
- recent self-performance summary

## 8. Prompt Evolution

Prompt evolution should be explicit and auditable.

There are two reasonable mechanisms:

### Option A: Rule-based mutation

- mutate prompt fragments from a constrained library
- adjust aggression, forgiveness, curiosity, memory usage, and self-description
- easy to reproduce and compare

### Option B: LLM-authored mutation

- use a separate mutation prompt to rewrite the policy prompt of offspring
- more creative, less predictable
- requires strong constraints and validation

Recommended first implementation:

- store prompts as structured templates with named traits
- mutate trait values first
- regenerate the concrete prompt from those traits

This is safer than directly letting the model rewrite its own prompt from scratch.

## 9. LLM Decision Loop

### 9.1 Ollama Integration

Use Ollama as a pluggable inference backend behind a narrow adapter.

Adapter responsibilities:

- submit generation requests
- enforce structured output schema
- track latency, token counts, and failures
- support per-agent model settings
- support retries and fallbacks

Recommended local-first defaults:

- start with one decision model for all agents
- allow later experiments with heterogeneous models
- require JSON output for action selection

Example response schema:

```json
{
  "action": "C",
  "confidence": 0.74,
  "reasoning_summary": "Opponent cooperated previously and trust remains positive.",
  "memory_ids_used": ["mem_142", "mem_807"],
  "recognition": {
    "matched_agent_id": "agent_12",
    "matched_lineage_id": "lineage_3",
    "confidence": 0.92
  }
}
```

Do not depend on hidden chain-of-thought. Require short public summaries only.

### 9.2 Cost Control

LLMs are the bottleneck. The architecture should support:

- caching exact decision contexts
- skipping LLM calls for simple deterministic strategies in baseline runs
- limiting context window via retrieval summaries
- batch scheduling where possible
- asynchronous inference queueing

## 10. Deterministic Core Vs Stochastic Boundary

Separate the system into:

- deterministic simulation core
- stochastic inference boundary

The deterministic core handles:

- grid updates
- payoff calculations
- parent selection
- persistence
- metrics
- experiment configuration

The stochastic boundary handles:

- action choice via Ollama
- prompt mutation if LLM-authored
- semantic memory summarization if LLM-assisted

This separation is necessary for debugging and reproducibility.

## 11. Persistence Model

Use SQLite first. It is sufficient for local research runs and easy to inspect.

Suggested tables:

- `runs`
- `generations`
- `cells`
- `agents`
- `lineages`
- `encounters`
- `rounds`
- `memories`
- `recognition_signatures`
- `prompt_versions`
- `events`

Store enough data to replay, analyze, and visualize a run without re-querying the LLM.

## 12. Configuration Model

Use file-based experiment configs rather than ad hoc CLI overrides.

Config domains:

- world: grid size, topology, birth/survival rules
- game: payoff matrix, rounds per encounter
- agents: seed prompts, mutation rules, memory settings
- llm: model name, temperature, max tokens, retry policy
- persistence: database path, artifact paths
- analysis: metrics to compute and export

Recommended format: YAML.

## 13. Observability And Tooling

Need visibility into both behavior and mechanics.

Minimum observability features:

- structured logs for each generation
- per-agent decision trace with prompt inputs and parsed outputs
- replayable artifacts for one encounter
- aggregate metrics export to CSV or Parquet
- periodic visualization snapshots of the grid and lineage distribution during a run
- optional simple UI later, but not required for v1

Recommended first visualization mode:

- periodic static image export every `n` generations
- optional ASCII snapshot in CLI output for tiny runs
- color cells by strategy or lineage and annotate aggregate cooperation rate

## 14. Metrics

Track both game-theoretic and evolutionary metrics.

Core metrics:

- cooperation rate
- defection rate
- average payoff by lineage
- lineage survival time
- memory retrieval frequency
- recognition accuracy
- prompt mutation frequency
- spatial clustering of cooperators and defectors
- model latency and token usage

Important derived questions:

- Do memory-enabled agents outperform stateless agents?
- Does recognition improve cooperation stability?
- Which prompt traits survive longest under different densities?
- Do local neighborhoods converge to stable cultural strategies?

## 15. Recommended Software Architecture

Use a modular Python application with these packages:

- `core/`: grid, rules, simulation loop, payoff engine
- `agents/`: agent state, prompt builder, memory manager, recognition manager
- `llm/`: Ollama adapter, schemas, response parsing
- `evolution/`: prompt traits, mutation, inheritance
- `storage/`: SQLite models and repositories
- `analysis/`: metrics and post-run reports
- `experiments/`: YAML configs and launch helpers
- `tests/`: deterministic unit and integration tests

Entry points:

- CLI for running experiments
- CLI for replaying a run
- CLI for exporting metrics

## 16. Risks And Design Constraints

### Main risks

- LLM outputs drift from schema or prompt intent.
- Memory grows faster than context budget.
- Game of Life semantics become muddy if payoff effects override spatial rules too heavily.
- Local inference throughput may be too slow for large worlds.
- Prompt evolution may produce unreadable or degenerate agents.

### Guardrails

- validate all model outputs against JSON schema
- cap retrieved memory count and summarize aggressively
- preserve a default canonical Life-like rule set
- start with small grids and low rounds per encounter
- keep prompt evolution structured and diffable

## 17. Recommended V1 Scope

Do not build the full research platform first.

V1 should include:

- one local Ollama model
- small Qwen instruct model served by local Ollama
- one grid world
- one canonical Prisoner's Dilemma payoff matrix
- persistent agent memory in SQLite
- exact-ID recognition plus simple signature matching
- rule-based prompt trait mutation
- CLI-based experiment execution
- periodic visualization snapshot export
- offline metrics export

Defer until later:

- multi-model tournaments
- rich visualization UI
- distributed execution
- embedding-heavy long-term memory systems
- free-form self-rewriting prompts

## 18. Open Decisions

These choices should be made before implementation starts in earnest:

1. Whether new births inherit memory directly, indirectly via summaries, or not at all.
2. Whether agents can communicate natural language to each other before choosing C or D.
3. Whether agent identity is transparent, partially hidden, or adversarially spoofable.
4. Whether Game of Life stays canonical or becomes payoff-weighted.
5. Which local model is the baseline for all experiments.

## 19. Recommended Baseline

Start with this baseline because it is simple and inspectable:

- finite 20x20 toroidal grid
- standard Moore neighborhood
- B3/S23 life rule for occupancy
- 3 rounds per neighboring encounter per generation
- exact agent IDs visible
- SQLite persistence
- local Ollama `qwen2.5:3b-instruct` shared across all agents
- structured prompt templates with rule-based trait mutation
- Python project using a workspace-local `.venv` with `uv` for dependency management

That baseline is sufficient to answer whether memory and recognition materially change cooperation dynamics.