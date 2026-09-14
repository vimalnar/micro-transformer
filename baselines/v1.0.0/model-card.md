# Model card: vanilla Micro-Transformer v1.0.0

## Model

The reference model is a five-layer, decoder-only causal transformer with width
128, four attention heads, feed-forward width 512, context length 128, tied output
weights, and 1,024,512 parameters. The output vocabulary contains 128 formal
language tokens; padding is input-only ID 128.

The bundled checkpoint is seed 42 at the fixed 31,250-step endpoint. It contains
weights needed for inference but omits optimizer and random-state data. Its source
architecture hash and source full-checkpoint hash are recorded in `manifest.json`.

## Intended use

- Inspect a complete small transformer without a large-model framework.
- Fine-tune a frozen reference checkpoint with LoRA.
- Compare a modified architecture trained under the fixed baseline protocol.
- Study controlled data, evaluation, counterfactual, and shortcut interventions.
- Serve as the language-model component of a separately specified Micro-World
  adapter experiment.

## Measured results

| Suite | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|
| Standard exact | 91.44% | 90.73% | 88.54% |
| Challenge exact | 78.02% | 78.65% | 75.97% |
| Shortcut exact | 8.90% | 22.93% | 11.93% |

The shortcut suite is deliberately difficult and narrow. Its weak results are
baseline measurements, not cases to omit. The complete release retains per-record
predictions and the published breakdown.

## Limits

The model is valid only for the fixed vocabulary, grammar, context window, and
measured data distributions. It is not a general-purpose language model and is not
evidence of general reasoning, persistent memory, online learning, selfhood, or
sentience. Prompts are self-contained and independent. The model is not directly
trained as a Micro-World action policy.

The historical Full Test result was `partially_demonstrated`. A post-hoc correction
to the replay comparison, with no changed model, data, predictions, or metrics,
produced `demonstrated_with_controls`. Both records are retained in the complete
release and must not be silently conflated.
