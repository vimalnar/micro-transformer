# Reference baseline v1.0.0

This directory is the small, version-controlled entry point for the trained vanilla
Micro-Transformer baseline. It contains one inference-only seed-42 checkpoint,
machine-readable identity and results, representative data records, and links to
the optional complete reproducibility bundle.

The checkpoint is a frozen 1,024,512-parameter causal transformer. It was trained
from random initialization on the corrected one-million-record corpus. It has no
retrieval, persistent memory, simulator connection, pretrained state, active LoRA,
or rule-based answer path.

## Try it

Install the training dependencies, then run from the repository root:

```bash
.venv-training/bin/python -m training.infer \
  --checkpoint baselines/v1.0.0/checkpoints/seed42-inference.pt \
  --prompt 'hal move flag six to river. hal tell gia flag six at river. where gia knows flag six?'
```

The expected completion is `river |`. This checks that the model loads and runs; it
is not a substitute for the frozen evaluation suites.

## Choose the artifact level

- **Included core:** source, generator, tests, documentation, sample records, and
  the 3.9 MB seed-42 inference checkpoint. This is sufficient to explore the
  language, run predictions, fine-tune with LoRA, and start a new experiment.
- **Complete release:** all three canonical training checkpoints, the independent
  seed-42 replay, packed train/validation/standard data, held-out cases, archived
  predictions, reports, evidence, and clean-install verification. Install it with
  `scripts/baseline_artifacts.py` once the release directory or archive is
  available.

```bash
python scripts/baseline_artifacts.py install --source /path/to/micro-transformer-baseline-v1.0.0
python scripts/baseline_artifacts.py verify
```

Large artifacts are installed under `artifacts/releases/` and deliberately remain
outside ordinary Git history. The catalog's `download_url` remains null until the
prepared release is uploaded to an immutable public location.

## Comparisons

Use the same packed data, seeds, step budget, evaluator, and frozen suites when an
experiment is intended to isolate an architecture change. A LoRA result is a
fine-tuned derivative and should be labelled separately from a from-scratch
architecture comparison.

See [model-card.md](model-card.md), [dataset-card.md](dataset-card.md), and
[`docs/guides/experiment-guide.md`](../../docs/guides/experiment-guide.md).
