# Experiment guide

Micro-Transformer supports two distinct experiment paths.

## Architecture comparison

Train both the vanilla reference and the modified architecture from fresh random
initialization. Hold the dataset bytes, initialization seeds, optimizer, batch
size, step budget, checkpoint rule, and evaluation suites fixed. Report all three
seeds and every task breakdown. This is the appropriate path for attention,
normalization, positional encoding, recurrence, width, depth, or other architectural
changes.

The canonical baseline uses one million records, batch size 32, 31,250 updates,
seeds 42/43/44, AdamW at 0.0003 with weight decay 0.01, gradient clipping 1.0,
CPU float32, and ordinary next-token cross-entropy. The complete artifact release
contains the exact packed arrays and evaluation cases.

## LoRA adaptation

Start from the bundled seed-42 checkpoint, freeze its native parameters, attach
LoRA modules, and train only the adapters. This is appropriate for narrow task
adaptation, alternative data mixtures, or a Micro-World interface experiment. It
does not replace a from-scratch architecture comparison.

```bash
.venv-training/bin/python -m training.train \
  --init-from baselines/v1.0.0/checkpoints/seed42-inference.pt \
  --train-data artifacts/datasets/prepared/my-train \
  --validation-data artifacts/datasets/prepared/my-validation \
  --run-dir artifacts/runs/my-lora-experiment \
  --steps 1000 --lora-rank 8 --lora-alpha 16
```

Evaluate the adapted model on its new task and on the original standard, challenge,
and shortcut suites. Distribute the adapter separately from the unchanged base
checkpoint and record the base checkpoint hash.

## Reporting rules

- State whether the run is from scratch, full fine-tuning, or LoRA.
- Publish raw results for every seed rather than only the best run.
- Keep validation separate from final evaluation.
- Treat the now-public v1 final cases as regression tests; use fresh held-out cases
  for claims made after tuning against them.
- Retain genuine failures and define exactly what changed from the baseline.
- Do not infer memory or general reasoning from self-contained input-to-prediction
  accuracy.
