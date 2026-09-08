# Micro-Transformer v4: broader data and unchanged-transformer retraining

8 September 2026

## Outcome

The generator now supports **13 registered task families and 51 variants** using
the unchanged 128-token vocabulary. New capability modules can be added through
an explicit registration interface and selected with a JSON task profile.

A fresh, unchanged 1,024,512-parameter transformer was trained for one full epoch
on a new 500,000-episode corpus. On the **same broader test set**, the old checkpoint
scores **28.25%** and the new full-epoch checkpoint scores **95.00%**. The original
repeated-move and simple-ownership failures are corrected in the full-epoch model.

This is not complete language mastery. Colour queries and longer event chains
remain weak, and overall language-model loss is a poor checkpoint selector for
question-answering accuracy in this run. These limits are retained below.

## Generator changes

- Repeated target moves, take/drop transitions, swaps and containment changes.
- Ownership with no transfers, multiple transfers, dropping, retaking and relocation.
- Balanced yes/no outcomes within every declared boolean variant, not just globally.
- Positive, negated, unknown and updated properties and information.
- Current, stale, updated and unknown observation/communication cases.
- Order-sensitive same-object events and mixed action sequences.
- Distractors before, between and after relevant events, including non-first query targets.
- An explicit longer-chain challenge distribution.
- Versioned task contracts, trusted extension modules, custom percentage profiles,
  source/output hashes and measured coverage reports.

The default CLI uses the registered families. The legacy direct-import helpers are
retained for compatibility but are not the default generation path.

The [v4 specification and extension guide](../specifications/data-generation-v4.md)
contains the full family list, schema, constraints, commands and working extension
example. The example was tested through generation, validation, preparation and
training without loading task code into the trainer.

## Data produced and checked

| Split | Episodes | Serialized tokens | Questions |
|---|---:|---:|---:|
| Train | 500,000 | 22,521,484 | 480,000 |
| Validation | 5,000 | 220,482 | 4,800 |
| Test | 5,000 | 219,604 | 4,800 |
| Challenge | 3,000 | 163,338 | 2,880 |

Files are named `artifacts/datasets/micro-transformer-v4-*.jsonl`; prepared data lives in
`artifacts/datasets/prepared/v4-*`. All four corpora passed post-generation replay,
uniqueness, full 128-token coverage and declared capability-coverage checks. Their
file hashes match their manifests. Training, validation and test entity sets remain
pairwise disjoint; challenge shares the test identity partition but excludes exact
sequences in the standard test corpus.

The training manifest records **68,992 questions with repeated direct target
moves** and **237,040 questions targeting an entity other than the first mentioned
entity**. All 51 variants appear and every declared boolean variant is balanced
to within one example. The 500k generation rejected 513 over-length candidates and
published no exact duplicate sequences.

These checks are not a guarantee of exhaustive semantics. Generation and replay
validation use the same reference-interpreter implementation; separate handwritten
golden cases check important semantics. Initial v4 manifests had a documented
wording correction in v4.0.1 to state this accurately. The actual episode bytes,
original producer hashes and source snapshots were preserved.

## Training protocol

The architecture, 128-token mapping, model dimensions, AdamW learning rate 0.0003,
weight decay 0.01, clipping 1.0 and seed 42 were unchanged. The architecture was
not modified, and initialization was fresh.

```bash
.venv-training/bin/python -m training.train \
  --architecture models/transformer.py \
  --model-config training/configs/baseline.json \
  --train-data artifacts/datasets/prepared/v4-train-500k \
  --validation-data artifacts/datasets/prepared/v4-validation-5k \
  --run-dir artifacts/runs/transformer-v4-01 \
  --epochs 1 --batch-size 32 --eval-episodes 5000 \
  --eval-every 1000 --log-every 1000 --save-steps 3200 --device mps
```

Training completed 15,625 updates, consuming all 500,000 episodes once and
22,021,484 target tokens (serialized tokens minus each episode's first token).
The measured loop took **709.41 seconds (11m 49s)**, including periodic validation
and checkpointing but excluding initialization and final answer evaluation.

Three endpoints were retained: a preselected step-3,200 snapshot, `best.pt` selected
by minimum validation token loss, and the fixed full-epoch `latest.pt`. The lowest
validation loss was **1.17299 at step 6,000**; final loss was **1.18728**. Final
validation question accuracy was **4,549/4,800 (94.77%)**. Test results did not
trigger more training, dataset changes, or a rename/replacement of `best.pt`.

## Same-benchmark comparison

| Checkpoint | Training endpoint | Correct / test questions | Accuracy |
|---|---|---:|---:|
| Old v3 trial, selected checkpoint | Step 2,800 | 1,356 / 4,800 | 28.25% |
| New v4, early snapshot | Step 3,200 | 2,200 / 4,800 | 45.83% |
| New v4, loss-selected `best.pt` | Step 6,000 | 3,649 / 4,800 | 76.02% |
| New v4, fixed-epoch `latest.pt` | Step 15,625 | 4,560 / 4,800 | 95.00% |

The old selected checkpoint consumed approximately 4.473 million target tokens;
the new step-3,200 snapshot consumed 4,512,572. They are roughly comparable token
budgets, not an exact controlled causal experiment. The full-epoch model received
substantially more training. One seed and changed data/training duration do not
isolate every cause of improvement.

The original old-test 95.3% score is not directly comparable to these percentages:
that was a different, much narrower benchmark.

Raw generated reports contained every prediction and frozen-state check for the
old model, early snapshot, loss-selected model, and full-epoch model. They are not
committed; this document retains their measured totals and interpretation.

## Full-epoch task results

| Family | Correct / questions |
|---|---:|
| State tracking | 700 / 700 |
| Ownership | 600 / 600 |
| Delayed recall | 483 / 500 |
| Property | 454 / 600 |
| Conditional | 400 / 400 |
| Belief | 350 / 350 |
| Spatial | 300 / 300 |
| Quantifier | 339 / 350 |
| Temporal | 296 / 300 |
| Communication | 250 / 250 |
| Ability | 250 / 250 |
| Mixed composition | 138 / 200 |

Per-variant reporting matters: direct colour questions scored only **32/120
(26.67%)**, same-colour comparisons **62/120 (51.67%)**, and colour questions after
mixed actions **25/66 (37.88%)**. Aggregating all property variants conceals this.

## Original diagnostic prompts

| Case | Old model | Full-epoch v4 model | Expected |
|---|---|---|---|
| Move to garden; ask where | garden | garden | garden |
| Move to garden, then hall; ask where | garden | hall | hall |
| Ben takes the object; ask who has it | yes | ben | ben |
| Paint red; ask colour | red | blue | red |

The two originally failing cases now pass, but one previously passing colour case
regressed. Total is **3/4**, not 4/4. These four diagnostics are not a statistical
benchmark. The generated diagnostic predictions included the regression.

## Longer-chain challenge

The fixed full-epoch model scored **2,437/2,880 (84.62%)** across the complete
challenge set; the loss-selected checkpoint scored **1,980/2,880 (68.75%)**.

However, the core longer-move variants are substantially harder: full-epoch direct
state tracking scored **38/84 (45.24%)**, and delayed move recall **45/100 (45%)**.
Those variants use five to seven target moves rather than the standard mixed
training distribution's one to four. The aggregate challenge also includes
variants not length-held-out, so it must not be presented as 84.6% on wholly unseen
structures. Counting is additionally limited by the held-out entity pool, which
allows only one or two identifiers per kind.

Raw full-epoch and loss-selected challenge reports are reproducible local artifacts.

## Using the resulting model

For the fixed full-epoch checkpoint evaluated above:

```bash
.venv-training/bin/python -m training.infer \
  --checkpoint artifacts/runs/transformer-v4-01/latest.pt --interactive
```

`best.pt` still means lowest overall validation **token loss**, not highest QA
accuracy. It was not relabelled after inspecting test results. For future runs,
an answer-focused validation selection criterion is worth evaluating before
testing; otherwise the model that best predicts random scene descriptions need
not be the one that answers questions best.

The model architecture and original v3 corpus hashes are unchanged. Prepared v1
data remains readable. Regression tests cover capability balance, semantic golden
cases, disjointness, duplicates, bounded episodes, resume, sharding, extension
generation and training, checkpoint snapshots, and per-variant inference.

Final regression checks passed all **59 distinct tests** across the existing app
and training environments: 39 in `.venv`, plus the 20 PyTorch-dependent model and
learning tests in `.venv-training`. The 16 data tests also passed in the training
environment. Checkpoint hashes, prediction counts and totals, common test-dataset
identity, and frozen-weight checks were reverified against the saved reports.

## Remaining work

The generator is broader and extensible, and the requested retraining/evaluation
is complete. It is not an all-purpose future-capability dataset. Remaining model
work includes colour grounding, mixed-action generalization and longer chains.
Answer-aware checkpoint selection, multiple seeds and controlled training-budget
comparisons would strengthen subsequent experiments. New vocabulary, semantics,
multimodal observations or cross-episode persistence require explicitly versioned
extensions rather than being assumed from the current registry.
