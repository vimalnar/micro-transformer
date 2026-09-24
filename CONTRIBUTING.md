# Contributing

Micro-Transformer values small, controlled and reproducible experiments.

Before proposing a model change:

1. State the hypothesis and expected endpoint.
2. Identify the frozen baseline, data, seeds, controls and evaluation suites.
3. Keep the intervention isolated from unrelated refactoring.
4. Retain negative and inconclusive results.
5. Include source/configuration identities, raw outputs and reproduction commands.
6. State what the result does not establish, especially about scale transfer,
   general reasoning, agency or consciousness.

Run the relevant unit tests and `scripts/verify_v1_suite.py`. Training changes
should also pass the full interoperability smoke path. Do not commit generated
datasets, large checkpoints or report directories unless they are intentionally
curated release artifacts.
