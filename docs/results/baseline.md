# Micro-Transformer Trial 01

8 September 2026

## Result

The complete data → training → saved checkpoint → frozen inference → scoring
pipeline works. The trial is not evidence that the model reliably understands
every valid Micro-Transformer sentence.

The selected checkpoint answered **905 of 950 held-out questions correctly
(95.26%)**. However, it failed **two of four manual diagnostic prompts**, including
an ordinary two-move location update. Both results matter: the generated benchmark
is substantially easier for this model than some simple variations of the language.

## Training

| Item | Value |
|---|---|
| Architecture | `models/transformer.py`, unchanged vanilla causal transformer |
| Parameters | 1,024,512 |
| Configuration | 5 layers, width 128, 4 heads, feed-forward width 512, context 128 |
| Vocabulary | 128 output classes; input-only padding ID 128 |
| Initialization | Fresh random weights, seed 42; no pretrained checkpoint or LoRA |
| Optimizer | AdamW, learning rate 0.0003, weight decay 0.01, clipping 1.0 |
| Trial duration | 3,200 updates, batch size 32 |
| Training examples consumed | 102,400 distinct episodes from the shuffled 500,000-episode corpus |
| Training target tokens | 5,113,543, excluding padding and each episode's initial token |
| Measured training loop | 154.37 seconds on Apple MPS; excludes setup and final answer evaluation |
| Validation | All 1,000 validation episodes, every 400 updates |
| Initial validation loss | 4.86406 |
| Lowest validation loss | 1.24829 at step 2,800 |
| Final validation loss | 1.27302 at step 3,200 |
| Final checkpoint validation QA | 913/950 (96.11%); not used to choose the test checkpoint |

The run completed normally. `best.pt` is the step-2,800 checkpoint selected by
minimum validation token loss; `latest.pt` is step 3,200. The selection was checked
against the complete validation log. The test split was not used for checkpoint
selection or further training. The trained model was not changed after testing.

```bash
.venv-training/bin/python -m training.train \
  --architecture models/transformer.py \
  --model-config training/configs/baseline.json \
  --train-data artifacts/datasets/prepared/train-500k \
  --validation-data artifacts/datasets/prepared/validation-1k \
  --run-dir artifacts/runs/transformer-trial-01 \
  --steps 3200 --batch-size 32 --eval-episodes 1000 \
  --eval-every 400 --log-every 200 --device mps
```

This is the executed command. Use a new run directory to reproduce it without
overwriting the existing trial. Runtime versions and exact model/data identities
were recorded in the generated run's `run.json`.

## Held-out test

The test contains 1,000 episodes: 950 questions and 50 statement-only episodes.
Question accuracy requires the complete generated answer and `|` to match exactly.
All 950 generated answers ended with `|`. Statement-only episodes contribute to
token metrics but are not counted as automatically correct answers.

| Task family | Correct / questions | Accuracy |
|---|---:|---:|
| State tracking | 150 / 150 | 100% |
| Delayed recall | 150 / 150 | 100% |
| Ownership | 100 / 100 | 100% |
| Property | 76 / 100 | 76% |
| Conditional | 98 / 100 | 98% |
| Belief | 100 / 100 | 100% |
| Communication | 40 / 40 | 100% |
| Spatial | 80 / 80 | 100% |
| Quantifier | 50 / 50 | 100% |
| Temporal | 42 / 50 | 84% |
| Ability | 19 / 30 | 63.33% |
| Total | 905 / 950 | 95.26% |

Next-token loss is 1.25922, perplexity 3.52266, and token accuracy 64.41%.
Next-token accuracy is not question accuracy: it also includes prediction of
generated statements whose randomly chosen contents are not always predictable.

The corpus partitions have 205 training, 26 validation, and 25 test object
kind/identifier combinations, with no pairwise overlap. This is held-out identity
composition within the generator's existing task templates, not an independent
language-wide generalization test.

The new harness on MPS and the existing evaluator on CPU agree on exact answer
accuracy and every task's correct count. Their loss difference is below 1e-7.
Every prediction, not just selected successes, was retained in the generated raw
evaluation artifacts. Those large reproducible files are not committed.

## Manual diagnostic cases

These four spot checks were recorded during interactive testing, then saved as
reusable cases. They are diagnostics, not a randomly sampled benchmark. Expected
answers are independently checked against the existing reference interpreter.

| Case | Expected | Generated | Result |
|---|---|---|---|
| Move cube two to garden; ask where it is | `garden \|` | `garden \|` | Correct |
| Move it to garden, then to hall; ask where it is | `hall \|` | `garden \|` | Incorrect |
| Move it to garden, then Ben takes it; ask who has it | `ben \|` | `yes \|` | Incorrect |
| Paint it red; ask its colour | `red \|` | `red \|` | Correct |

The two-move prompt is:

```text
ava move cube two to garden. ava move cube two to hall. where cube two?
```

The state-tracking generator currently places the queried object once, then
protects it from distractor changes. Therefore, 100% on that template does not
test whether the model follows repeated updates to the same target. These results
are consistent with coverage gaps and/or incomplete learning; this single trial
does not isolate their relative contributions. Task names such as "belief" and
"delayed recall" are benchmark labels, not evidence of capabilities beyond the
specific evaluated tasks.

The [diagnostic case definitions](../../training/examples/trial_cases.jsonl) are
version-controlled. Their generated summaries and predictions are local artifacts.

## Inference harness

Start interactive inference from the project root:

```bash
.venv-training/bin/python -m training.infer \
  --checkpoint artifacts/runs/transformer-trial-01/best.pt --interactive
```

Enter a complete prompt on one line, without `|`. Each line is independent;
previous lines are not remembered. Enter `:quit` to exit. `--json` exposes generated
token IDs, timing, and whether generation stopped at `|` or a limit.

To repeat the scored tests, choose new report directories:

```bash
.venv-training/bin/python -m training.infer \
  --checkpoint artifacts/runs/transformer-trial-01/best.pt \
  --data artifacts/datasets/prepared/test-1k \
  --report-dir artifacts/reports/trial-01-test-repeat

.venv-training/bin/python -m training.infer \
  --checkpoint artifacts/runs/transformer-trial-01/best.pt \
  --cases training/examples/trial_cases.jsonl \
  --report-dir artifacts/reports/trial-01-cases-repeat
```

The harness loads weights once and does not use the interpreter to answer queries.
The stored answer is removed before generation and only used for scoring. All
model parameters are frozen. SHA-256 of the full model state, including buffers,
was identical before and after the entire held-out test; greedy output remained
identical when the first prompt was repeated after all other prompts.

Automated tests cover checkpoint reload, repeatability, frozen state, answer
separation, unknown/empty/oversized input, explicit context limits, interactive
error recovery, custom cases, interpreter agreement for case labels, and agreement
with the existing evaluator. Existing model/training and simulator tests also pass.

The original 500k dataset, generator, transformer architecture, and gridworld were
not modified. Only inference tooling, its tests, documentation, and new trial
artifacts were added or changed.

## Interpretation

This achieves end-to-end execution and supplies an inspectable, reloadable trial
model. It does not establish a fully competent Micro-Transformer baseline. Before using
this checkpoint to attribute improvements to a changed architecture, extend the
evaluation/training coverage for ordinary state updates and other demonstrated
gaps, then rerun comparisons under the same protocol. No such dataset or
architecture changes were made during this trial.
