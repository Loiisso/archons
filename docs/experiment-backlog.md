# Experiment Backlog

This backlog tracks experiment ideas that are close enough to the current architecture to be implemented without rethinking the simulator from scratch. Each item is written as an executable experiment, not just a feature request.

## 1. Enable Agent Communication

### Status

Not started.

### Why This Matters

Right now agents only observe:

- their own fixed base prompt and current policy prompt
- recognition and memory summaries about the opponent
- the local encounter history

They cannot send or receive messages before choosing `C` or `D`. That means the simulation currently models memory, recognition, and evolved prompt policy, but not negotiation, signaling, bluffing, apology, or reputation repair through language.

### Experiment Question

Does allowing short pre-action communication increase cooperation stability, or does it mostly create deceptive cheap talk?

### Hypothesis

- In sparse or moderately dense worlds, short bilateral communication will increase cooperation among agents with memory and recognition.
- In crowded worlds, communication may initially help but then collapse into manipulation unless memories track message honesty.

### Current Gap

- No communication phase exists in the encounter loop.
- No message artifacts are persisted.
- No metrics distinguish truthful signaling from deceptive signaling.

### Proposed Scope

Add an optional communication phase before action selection.

Recommended first implementation:

1. Add `game.communication.enabled` to config.
2. Add a single short message exchange before each round or before each encounter.
3. Limit message format to one short text field, optionally with a structured intent field.
4. Inject the received message into the next decision prompt.
5. Persist all messages and message metadata in SQLite.

Recommended message schema:

```json
{
  "message": "I will cooperate if you do.",
  "intent": "promise"
}
```

### Design Constraints

- Keep message length hard-capped to avoid exploding inference cost.
- Treat communication as optional and fully config-driven.
- Do not let communication replace the existing decision call at first; add it as a separate phase.
- Track whether communication is pre-encounter or pre-round.

### Minimum Implementation Tasks

1. Add communication config fields and validation.
2. Extend the encounter model with `MessageRecord` storage.
3. Add an Ollama message-generation prompt and response schema.
4. Inject the opponent's last message into the decision prompt.
5. Persist messages in a `messages` table keyed by run, generation, encounter, and round.
6. Export communication metrics.

### Suggested Metrics

- cooperation rate with communication enabled vs disabled
- message frequency
- promise rate
- promise-keeping rate
- apology frequency after defection
- deception rate: cooperative language followed by defection
- long-term cooperation after prior betrayal and apology

### Risks

- Cost may rise sharply if every round adds another model call.
- Messages may be repetitive and low-signal without strong prompt constraints.
- Agents may exploit natural language for empty promises unless honesty is scored.

### Success Criteria

- Communication can be toggled on and off from YAML.
- Runs persist a full message trace.
- The analysis layer can compare communication-enabled and communication-disabled runs.
- At least one report answers whether communication improved cooperation or mostly increased deception.

### Recommended First Experiment Matrix

1. communication off, memory on, recognition on
2. communication on, memory on, recognition on
3. communication on, memory off, recognition off
4. communication on, honesty scoring enabled

## 2. Enable Prompt Mutation As A First-Class Experiment

### Status

Partially implemented in code, not yet formalized as an experiment track.

### Why This Matters

The simulator already mutates structured prompt traits on birth and regenerates child policy prompts. That means prompt mutation exists mechanically, but it is not yet exposed as a clear experiment family with explicit controls, mutation logs, and comparison reports.

### Experiment Question

Do mutated prompt traits create lineages that meaningfully diverge in strategy and survival, or are current mutations too weak and too opaque to study?

### Hypothesis

- Small structured mutations will create stable strategic families under some densities and payoff settings.
- Without lineage-level logging and ablations, it will be difficult to tell whether observed changes came from prompt mutation or from random spatial dynamics.

### Current State

Already present:

- seeded prompt traits by strategy
- rule-based scalar mutation on birth
- regenerated `policy_prompt`
- persistence of prompt traits and prompt text on cells

Still missing as an experiment layer:

- explicit mutation on vs off comparisons
- mutation event logs
- parent-child prompt diffs
- lineage diversity metrics
- reports that connect prompt drift to survival outcomes

### Proposed Scope

Turn prompt mutation into an explicit experiment axis instead of an internal mechanism.

Recommended first implementation:

1. Add a top-level experiment toggle for mutation enabled vs disabled.
2. Log every mutation event with parent traits, child traits, and textual prompt diff summary.
3. Add lineage-level reports showing which traits spread or disappear.
4. Support mutation schedules and strengths from config.

### Minimum Implementation Tasks

1. Add `agents.prompt_evolution.enabled`.
2. Add a `mutation_events` table in SQLite.
3. Record parent and child trait values for every birth.
4. Persist a compact prompt diff summary.
5. Export lineage diversity and trait drift metrics.
6. Add experiment configs for mutation disabled, mild mutation, and aggressive mutation.

### Suggested Metrics

- mutation frequency per generation
- lineage count over time
- trait variance over time
- dominant prompt style by generation
- average payoff by prompt style
- lineage survival time by mutation regime
- cooperation rate by prompt style and lineage

### Risks

- If mutations are too small, lineages may not diverge enough to study.
- If mutations are too large, prompt behavior may become noisy and uninterpretable.
- Prompt text diffs may be hard to understand if the rendered prompt changes in many places at once.

### Success Criteria

- Mutation can be turned off cleanly for control runs.
- Every birth optionally emits a mutation event artifact.
- Reports can answer which mutated traits persisted, which died out, and whether mutation improved fitness.
- A reader can inspect parent-child prompt drift without reverse-engineering internal state.

### Recommended First Experiment Matrix

1. mutation off
2. mutation on with low step size
3. mutation on with medium step size
4. mutation on with high step size

### Recommended Follow-Up

Once rule-based mutation is fully observable, consider a second phase:

- LLM-authored mutation under a strict schema
- mutation conditioned on parent fitness
- mutation conditioned on local neighborhood stress

Do not start there. Make the structured mutation path measurable first.