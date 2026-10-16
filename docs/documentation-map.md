# Find your way around the documentation

Start with the short route that matches what you want to do. The full technical
references are available in the documentation navigation when you need them.

## New to Micro-Transformer

1. Read [R&D suite artifacts and orientation](artifacts.md) for the components,
   their boundaries, and the current evidence.
2. Follow [Install and setup](guides/setup.md) to prepare the environment.
3. Use the [V1 quickstart](guides/v1-quickstart.md) to run the bundled model.

## Learn or try the model

- [Frozen-model harness](guides/harness.md) explains prompts, interactive
  commands, and known limits.
- [Worked examples](guides/examples.md) walks through successful cases and
  retained failures.
- [Model specifications](specifications/model-specifications.md) and the
  [transformer architecture](specifications/transformer-architecture.md) explain
  the checkpoint and its implementation.
- The [language reference](reference/micro-transformer-language.md) defines the
  fixed vocabulary, grammar, and meanings used by the model.

## Build or evaluate an experiment

- [Data generation v4](specifications/data-generation-v4.md) describes the
  procedural corpus and its validation.
- [Training and inference](guides/training-and-inference.md) covers data
  preparation, training, checkpoint selection, and evaluation.
- The [experiment guide](guides/experiment-guide.md) and
  [report specification](specifications/experiment-report-v1.md) cover controls,
  endpoints, and reporting.
- [Reproducibility](guides/reproducibility.md) explains how to identify inputs,
  runs, outputs, and reports retained in external storage.
- [Results](results/v1-readiness-audit.md) summarizes what is complete and what
  remains preliminary.

## Explore or integrate Micro-World

- [Using Micro-World](guides/microworld-usage.md) explains the simulator,
  inspector, and implemented HTTP/WebSocket routes.
- [Gridworld v1](specifications/gridworld-v1.md) defines its state, actions, and
  observations.
- [Connecting a model](guides/micro-world-model-adapter.md) describes the
  experiment-specific adapter boundary. The bundled language checkpoint is not
  a trained world controller.
- [System architecture](specifications/system-architecture.md) shows how the
  independent components fit together.

## Contribute and understand the limits

- [Contributing](guides/contributing.md) covers changes and evidence expectations.
- [Release status](guides/release-status.md) describes available artifacts and
  preliminary work.
- [Historical experiments](results/historical-experiments.md) identifies shelved
  agent and living-world work separately from the supported baseline.
- [Limitations](guides/limitations.md) explains what results do and do not
  establish.

The repository's [README](https://github.com/vimalnar/micro-transformer) is the
GitHub entry point. This guide and the documentation navigation are the routes
into the longer explanations.
