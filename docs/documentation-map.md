# Documentation map

This table is the recommended information architecture for the project. It separates
practical onboarding from technical specifications and from evidence about results.

| Area | What it should answer | Primary audience | Recommended contents | Current source or status |
| --- | --- | --- | --- | --- |
| Setup | How do I install, verify, and run the repository? | Everyone | Python version, environments, dependencies, checkout, smoke tests, troubleshooting | [`Install and setup`](guides/setup.md) |
| Quickstart | What is the shortest successful first session? | New users | Install, run one example, inspect one output, run one test | [`V1 quickstart`](guides/v1-quickstart.md) |
| Project orientation | What is this project, and what is it not? | Everyone | Purpose, claim limits, repository map, terminology, evidence boundaries | [`System architecture`](specifications/system-architecture.md) and the [home page](index.md) |
| Model specifications | What model variants exist, with which parameters and checkpoints? | Researchers, engineers | Configuration, parameter counts, tokenizer/vocabulary, context, checkpoints, provenance, compatibility | [`Model specifications`](specifications/model-specifications.md) |
| Transformer architecture | How is the transformer implemented? | Engineers, researchers | Embeddings, attention, MLP, normalization, positional handling, causal mask, forward pass, extension points | [`Architecture guide`](specifications/transformer-architecture.md) |
| Language and semantics | What does the formal language mean? | Researchers, contributors | Vocabulary, grammar, interpreter semantics, episode format, invalid cases | [`Language reference`](reference/micro-transformer-language.md) |
| Data and generators | How is data produced, validated, split, and identified? | Researchers, engineers | Generator commands, schemas, manifests, seeds, partitions, paired/counterfactual data, leakage controls | [`Data generation v4`](specifications/data-generation-v4.md) |
| Examples | What can I run or modify immediately? | Students, hobbyists, evaluators | Small scripts, expected output, progressively harder examples, failure interpretation | [`Worked examples`](guides/examples.md); companion case collection is linked there |
| Harness | How do I interact with a frozen model? | Users, evaluators | CLI/API usage, prompts and outputs, checkpoint selection, deterministic settings, limitations | [`Frozen-model harness`](guides/harness.md) |
| Training and inference | How are models trained and evaluated? | Engineers, researchers | Environment separation, commands, checkpoints, metrics, inference, resource expectations | [`Training and inference workflow`](guides/training-and-inference.md) |
| Evaluation and benchmarks | What is measured, and what does it establish? | Researchers, reviewers | Tasks, controls, held-out evaluation, baselines, validity, negative results, claim boundaries | [`V1 readiness audit`](results/v1-readiness-audit.md) and [`Experiment report specification`](specifications/experiment-report-v1.md) |
| Micro-World | How does the gridworld work, and how can a model connect to it? | Researchers, developers | Observation/action contract, lifecycle, adapter, replay, visual/audio boundaries, known limitations | [`Using Micro-World`](guides/microworld-usage.md), [`Gridworld v1`](specifications/gridworld-v1.md) and [`Model adapter`](guides/micro-world-model-adapter.md) |
| Experiments | How do I design, run, and report a controlled experiment? | Researchers | Registration, identity, controls, artifacts, review states, stopping rules, reporting template | [`Experiment guide`](guides/experiment-guide.md) and [`Experiment report specification`](specifications/experiment-report-v1.md) |
| Results and evidence | What has actually been observed? | Everyone, reviewers | Exact runs, metrics, artifacts, interpretation, alternatives, unresolved limitations | [`Results`](results/baseline.md) |
| API and extension points | How do I integrate another model or tool? | Developers | Stable interfaces, adapters, schemas, examples, compatibility policy | Partly covered by [`System architecture`](specifications/system-architecture.md); needs API reference pages |
| Reproducibility | Can I reproduce a result and identify the exact inputs? | Researchers, reviewers | Commits, manifests, seeds, environments, data/checkpoint identities, verification commands | [`Reproducibility and reporting`](guides/reproducibility.md) |
| Contributing | How should changes be proposed and validated? | Contributors | Code style, tests, docs, artifact policy, pull requests, claim review | [`Contributing`](guides/contributing.md) and repository [`CONTRIBUTING.md`](https://github.com/vimalnar/micro-transformer/blob/main/CONTRIBUTING.md) |
| Release and versioning | What is stable, preliminary, or experimental? | Users, downstream adopters | Version policy, compatibility, release checklist, model cards, deprecation policy | [`Release status`](guides/release-status.md) |
| Limitations and ethics | What should users not infer from this project? | Everyone | Capability limits, consciousness non-claims, privacy, responsible use, known failure modes | [`Limitations and claim boundaries`](guides/limitations.md) |

## Recommended build order

1. Setup and quickstart.
2. Model specifications and transformer architecture.
3. Examples and harness.
4. Data, training, inference, and evaluation.
5. Micro-World and adapter interfaces.
6. Reproducibility, contributing, release, and limitations.

This order follows the dependency chain a reader experiences: get the project
running, understand what is running, use it, evaluate it, extend it, and then assess
whether a result is reproducible and what claims it supports.
