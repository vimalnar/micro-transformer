# Larger v1-language reference model: training stage

This is a from-scratch **preliminary reference**, not a claim that results or abilities transfer
automatically from the 1.0M-parameter v1 baseline. The sole planned architecture
change is scale: 8 layers, width 320, 8 heads, feed-forward width 1280, context
128, and the unchanged ordered 128-token v1 language. This is 9,946,560
parameters (9.71 times the baseline's 1,024,512). Seed 42 has completed; the
remaining seed panel and challenge/shortcut evaluation are pending.

## Data and protocol

`data-binding.json` binds training, validation, and standard-test tensors to the
published baseline v1.0.0 release. The raw JSONL copy in `artifacts/datasets` is
not proven identical to the archived source and is **not** a training input.
`scripts/verify_large_model_data_binding.py` checks the 12 packed files against
their binding and the release manifest before every launch.

The matched training target is one deterministic shuffled pass over 1,000,000
records at batch 32 (31,250 optimizer steps), independently for seeds 42, 43,
44, and a fresh seed-42 replay. Runs use CPU float32, four threads, AdamW
at learning rate 0.0003, weight decay 0.01, epsilon 1e-8, `foreach=False`,
gradient clip 1.0, no scheduler, and ordinary unweighted next-token loss.
The complete 5,000-record validation split is scored at steps 1,000 through
31,000 and at 31,250. `latest.pt` at 31,250 is the fixed primary endpoint;
`best.pt` selected by validation answer-macro accuracy is secondary only.
The standard, challenge, and shortcut test suites are never used for training
or checkpoint selection.

The released baseline was trained by an archived Research Lab wrapper around
the native architecture and data/metric components; these new runs use the
repository's native `training.train` entry point. Data, sampler, model source,
objective, optimizer settings, and validation protocol are matched, but this
is not asserted to be byte-for-byte the historical trainer or environment.
The final comparison must report that implementation limit and independently
assess exact run identities, seeds, and held-out metrics before making claims.

## Current evidence

The completed seed-42 endpoint is recorded in `manifest.json` and
`comparison-report.md`. It scored 4,578/4,800 questions (95.38%) on the
standard held-out suite. The harness also confirmed frozen weights and
repeatable greedy output. This is diagnostic preliminary evidence only; it is
not yet a final multi-seed scaling result.

## Commands

From the repository root, with `.venv-training` installed:

```sh
.venv-training/bin/python scripts/run_large_v1_training.py seed-42 --dry-run
.venv-training/bin/python scripts/run_large_v1_training.py seed-42
.venv-training/bin/python scripts/run_large_v1_training.py seed-43
.venv-training/bin/python scripts/run_large_v1_training.py seed-44
.venv-training/bin/python scripts/run_large_v1_training.py seed-42-replay
```

For an interrupted run only, add `--resume`; it requires that run's `latest.pt`
and refuses changed source or configuration. Run outputs live under
`artifacts/runs/large-v1-<run-id>/` and are intentionally ignored by Git.

The completed seed-42 checkpoint currently remains in the local ignored run
directory. Before public reproduction, publish an inference-only checkpoint and
the full training/data bundle as immutable release assets, then replace the
null URL in `data-binding.json`.
