# Reference experiment: frozen baseline in Micro-World

Experiment: `REF-MW-001`; benchmark: `micro-world-benchmark-v1`; split: final,
seeds 1001 and 1002; model: frozen Micro-Transformer v1.0.0, seed 42, step 31,250.

## Question

Can the unmodified language checkpoint produce executable and useful actions when
given a new, versioned encoding of the permitted 9x9 Micro-World observation?

This is a bounded diagnostic. The model was not trained on the adapter syntax, and
failure is a valid result.

## Result

| Policy | Completed | Complete-command parsing | Physical failure rate |
|---|---:|---:|---:|
| Scripted feasibility | 12/12 | 100% | 0% |
| Wait | 0/12 | 100% | 0% |
| Uniform random | 0/12 | 100% | 41.33% |
| Reactive, no history | 4/12 | 100% | 20.21% |
| Frozen language model | 0/12 | 0% | 0% |

Scientific outcome: **negative**. The frozen language model emitted no complete
action command accepted by the adapter and completed no scenario. It did not
outperform the behavioural controls. This does not show that Micro-Transformer
cannot support an action policy after a separately controlled training or action-head
experiment.

All scenario and policy replays matched exactly. The model state hash was
`1e438459ce129032046faef7b1d43c1b1287b397d63a2d3464794aa9c962f3be`
before and after evaluation, with zero trainable parameters. The raw decision-file
SHA-256 was
`a2d0f695cc9ea792531501e74f10d1f115f583038f8f4bf7e36d4b0a27e2c0ee`.
The exact decision log is retained at
`benchmarks/v1/reference-decisions.jsonl` and is checked against that hash by the
suite verifier.

## Reproduce

```bash
.venv-training/bin/python scripts/run_microworld_benchmark.py \
  --split final --device cpu \
  --report-dir artifacts/reports/my-final-regression-run
```

## Limits

- The observation/action grammar is new and untrained.
- Two placement seeds per scenario do not estimate broad task variability.
- Placement variation is not structural generalization.
- The scripted policy proves feasibility, not learnability or optimality.
- The result is not evidence about general agency, planning, a learned world
  model, subjective experience, or consciousness.
