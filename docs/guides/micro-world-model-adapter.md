# Connecting a model to Micro-World

Micro-World and the formal-language transformer are intentionally independent. The
simulator owns interactive world truth; the language interpreter owns generated
dataset truth. A model adapter is an experiment component, not a hidden part of
either baseline.

## Minimal adapter contract

1. Read only the agent's permitted typed observation, not the inspector's complete
   world view.
2. Serialize that observation and relevant in-episode history into vocabulary
   tokens within the 128-token context limit.
3. Ask for either a controlled answer or an action using an explicitly versioned
   prompt grammar.
4. Decode generated tokens into a typed candidate action.
5. Let Micro-World validate and execute the action; never mutate simulator state
   directly from model output.
6. Record observation, prompt tokens, checkpoint/adapter hashes, generated tokens,
   decoded action, simulator result, and replay identity.
7. Reset all model-side episode state at the declared boundary unless persistent
   state is the experimental intervention.

The bundled checkpoint can answer prompts expressed in its existing controlled
language. It was not trained to emit a complete Micro-World action policy. New
action syntax or observation formats therefore require task-specific training or
LoRA adaptation and separate evaluation.

Useful measurements include valid-action rate, task completion, steps to completion,
observation perturbation sensitivity, deterministic replay, and retention of the
original language suites. Do not describe success on a textual state query as
successful embodied control.

## V1 reference implementation

The bounded implementation is split into:

- `micro_world.agent_adapter`: a versioned observation/action convention using
  only the existing 128-token vocabulary;
- `micro_world.benchmark`: six goal-directed scenario families, stopping rules,
  replay, feasibility and behavioural controls, and report generation;
- `benchmarks/v1/manifest.json`: split seeds, metrics, controls, and claim limits;
- `scripts/run_microworld_benchmark.py`: a non-overwriting runner.

The adapter omits hidden world state and world coordinates from the model prompt.
The released language checkpoint was not trained on this convention, so parser or
task failure is expected to remain visible. A scripted controller establishes
scenario feasibility only. Wait, uniform-random, and reactive-no-history policies
are behavioural controls, not model variants.
