# Contributing

Contributions are most useful when they make the system easier to inspect and
results easier to interpret. Keep the intervention small, identify what changed,
and include evidence that supports the change.

## Before changing a model or benchmark

Write down:

- the question or hypothesis and the endpoint that could answer it;
- the baseline, exact intervention and comparison;
- the data, initialization seeds, controls and evaluation suites;
- the stopping rule, compute budget and likely confounds;
- what a positive, negative or inconclusive result would mean.

Keep architecture changes separate from unrelated refactoring. Preserve failed and
inconclusive runs. Do not change labels to match model predictions or use a test
suite for development and later describe it as untouched evaluation.

## Code and documentation changes

- Keep formal-language truth in its reference interpreter and Micro-World rules in
  the simulator.
- Keep new model/world translation in an explicit adapter with a versioned
  contract.
- Update specifications when an implemented interface changes.
- Describe code that exists separately from planned or proposed work.
- Include the exact command and relevant environment needed to reproduce a result.

## Checks

Run the relevant tests and the versioned suite verifier for code changes:

```bash
.venv/bin/python scripts/verify_v1_suite.py
.venv-training/bin/python scripts/verify_v1_suite.py --full
```

The full command runs the bounded generate-to-evaluate smoke path. Select the
relevant unit-test groups for the changed area. Training, checkpoint, generator,
benchmark and report changes may require additional checks described in the
repository's CI workflow.

## Artifacts and claims

Do not commit generated corpora, large checkpoints or report directories unless
they are intentionally curated release artifacts. Preserve data/model hashes and
raw outputs for a reported experiment. State the claim boundary, especially for
generalization, scale transfer, agency and consciousness.

The repository-level [`CONTRIBUTING.md`](https://github.com/vimalnar/micro-transformer/blob/main/CONTRIBUTING.md)
is the concise contributor policy. The detailed experiment contract is the
[experiment report specification](../specifications/experiment-report-v1.md).
