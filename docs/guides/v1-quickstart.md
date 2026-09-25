# V1 quick start

Micro-Transformer is a small research workbench for testing model and architecture
ideas before committing to expensive scale-up. The current suite includes a
preliminary single-seed larger reference model; the remaining seed panel,
challenge/shortcut evaluation, and public artifact release are still pending.

## Choose a path

### Hobbyist or student: inspect and change a complete model

1. Install the training requirements.
2. Run the integrated harness self-test.
3. Generate a small dataset.
4. Train a small run or copy `models/transformer.py` and change one component.
5. Compare the resulting report with the baseline and retain failures.

```bash
python3.12 -m venv .venv-training
.venv-training/bin/python -m pip install -r training/requirements.txt
.venv-training/bin/python harness/talk_to_micro_transformer.py --self-test
```

The complete generate, prepare, train, reload, and evaluate path can be checked
without a long training run:

```bash
.venv-training/bin/python scripts/verify_v1_suite.py --full
```

### Academic researcher: run a controlled architecture experiment

1. State the hypothesis and endpoint using
   `docs/specifications/experiment-report-v1.md`.
2. Copy the architecture file and change one declared component.
3. Hold dataset bytes, seeds, optimizer, step budget, checkpoint rule, and
   evaluation suites fixed.
4. Train baseline and candidate from fresh initialization across the declared
   seeds.
5. Run the standard, challenge, shortcut-control, and relevant Micro-World suites.
6. Run ablations and report every seed, failure, compute cost, and limitation.

The included seed-42 checkpoint is useful for inference and LoRA. A from-scratch
architecture claim requires matched from-scratch controls rather than comparison
with a fine-tuned derivative.

### Product R&D team: reduce scale-up uncertainty

Use the suite for model-level or embodied-system questions: memory mechanisms,
attention changes, compact routing, state representations, planning interfaces,
edge-model behaviour, and evaluation design. Define what should transfer, what is
model-specific, and what must be retested at the next scale.

A suitable workflow is:

```text
define task and failure cost
→ implement the smallest controlled intervention
→ measure against matched controls
→ identify failure conditions and compute trade-offs
→ decide whether a larger experiment is justified
→ rerun the same endpoints at larger scale
```

Micro-Transformer is not intended to replace a production model platform, hosted
agent service, RAG system, or deployment/governance stack.

## Micro-World reference experiment

```bash
.venv-training/bin/python scripts/run_microworld_benchmark.py \
  --split validation --device cpu \
  --report-dir artifacts/reports/my-micro-world-run
```

The output contains `summary.json` and `decisions.jsonl`. It distinguishes
technical completion, conformance, validity, and scientific outcome. The frozen
baseline currently provides a negative diagnostic under the untrained adapter;
that result is retained as evidence and as a starting point for a separately
specified model-development experiment.

## Release verification

```bash
.venv/bin/python scripts/verify_v1_suite.py
.venv-training/bin/python -m unittest discover -s tests
```

The final V1 scaling claim remains unavailable until the larger seed panel is
complete, the model is evaluated on the same named suites, and its public
artifacts are added to the release manifest.
