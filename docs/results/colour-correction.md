# Micro-Transformer colour correction

Status: complete. Basic colour failure corrected and independently retested on
fresh held-out examples. More complex distractor and length-generalization errors
remain explicitly reported below.

## Scope

Correct basic colour failures in the existing vanilla transformer. The model
architecture, parameter count (1,024,512), 128-token mapping and context limit
remain unchanged. The correction does not alter the architecture, and inference
continues to use fixed parameters. Original checkpoints and datasets are preserved.

The correction changes offline training and task sampling. The existing interpreter
still supplies world semantics during data generation, never during model inference.

## What was found

The previous full-epoch model frequently answered `blue` regardless of the paint
colour. On the earlier inspected broad test it predicted blue on 103/120 direct
colour questions. Training colour labels were already approximately balanced,
so a simple colour-frequency imbalance did not explain the failure.

Independent validation-identity diagnostics exposed a second problem: minimal
valid episodes, without an initial placement statement, were poorly understood.
The old model scored 2/200 on single paint, 50/200 on repaint, 50/200 on conflicting
distractors, 49/200 on mixed actions, 101/200 on same-colour comparisons and 0/200
on unknown colour with a painted distractor: 252/1,200 overall.

This is evidence of inadequate colour grounding and template dependence, not a
proof of a single internal neural cause.

## Development sequence

1. Added optional answer-weighted cross-entropy and validation-only QA checkpoint
   selection. Default options retain the old objective and selection behaviour.
2. Compared two 4,000-update branches from the same old full-epoch checkpoint,
   with identical data/order/settings except answer weight 1 versus 20. Neither
   corrected colour. Their selected validation direct-colour scores were 29/120
   and 24/120 respectively; same-colour scores were 60/120 and 59/120.
3. Added a registered colour curriculum: 30% simple/contrasting colour tasks,
   70% broad base tasks. The new 128,000-episode corpus has 4,981,329 serialized
   tokens, 125,440 questions, all 128 vocabulary tokens and zero exact duplicates.
   Interpreter replay, declared capability coverage and manifest hashes passed.
4. Trained the curriculum from the old full-epoch checkpoint for 8,000 updates,
   with a documented continuation from the original 4,000-update plan. The
   seed-42 selected step 5,500 reached 1,198/1,200 simple validation diagnostics.
5. Added a fixed 4,000-update refinement at learning rate 0.0001 to stabilize
   broader colour performance. Repeated the correction with seed 43, changing
   training order but starting from the same original base weights.

The archived development protocol retained when and why each development decision
was made. The [implementation guide](../specifications/colour-correction.md)
documents the loss, task extension, diagnostics and reproduction commands.

## Selection and comparison limits

Selection uses the mean exact-generation accuracy across validation question
variants; token loss breaks ties. `best_validation.json` records the rule, value
and step. Expected answers are never supplied in generation prompts. Test results
must not choose checkpoints or trigger another correction within this evaluation.

This is a practical fix, not an isolated estimate of the effect of curriculum or
loss weighting. The successful procedure changes data exposure and training work
as well as weighting. The first comparison only establishes that 4,000 updates
of extra training/weighting on the original corpus were insufficient in that run.
The second seed measures correction-order sensitivity, not robustness across
independently pretrained base models.

## Final evaluation

The primary checkpoint was locked using validation before fresh test generation:
`artifacts/runs/transformer-colour-refined-01/best.pt`, refinement step 3,500,
SHA256 `b4132e3efd92ca3df6ab3cc3f0776e212387fbd5389c93386eb6e32b821bde84`.
Its parent is the seed-42 curriculum checkpoint selected at step 5,500. The
original transformer file hash is unchanged:
`0c252303a5965a8d71b81eef2132911ec5a58f0ba12f53cac0262ab7d8b54f37`.

Fresh data: seed 46130 standard (5,000 episodes, 221,513 tokens), seed 46131
challenge (3,000 episodes, 164,146 tokens), seed 46132 simple colour diagnostics
(2,400 unique question prompts, 400 per family). Standard/challenge each contain
4% statement-only episodes, so question totals are 4,800 and 2,880. All new
corpora passed replay, vocabulary, capability coverage, uniqueness and hash checks.
New standard/challenge sequences exclude the previously inspected v3/v4 test and
challenge corpora; the new challenge also excludes the new standard set. Simple
case prompts additionally exclude those sets, the development cases and the four
original diagnostics. Test object identities remain disjoint from training and
validation identities. Labels were not selected using model outputs.

| Evaluation | Original full-epoch model | Corrected primary | Seed-43 repeat |
|---|---:|---:|---:|
| Simple colour cases | 500/2,400 (20.83%) | 2,400/2,400 (100%) | 2,400/2,400 (100%) |
| Fresh broad test | 4,558/4,800 (94.96%) | 4,785/4,800 (99.69%) | 4,791/4,800 (99.81%) |
| Fresh challenge, aggregate | 2,425/2,880 (84.20%) | 2,650/2,880 (92.01%) | 2,579/2,880 (89.55%) |
| Original four diagnostics | 3/4 | 4/4 | 4/4 |

Both corrected models pass all 400 cases in each simple family: single paint,
repaint, conflicting distractor, mixed actions, same/different colour and unknown
colour. The first four groups each contain 100 expected answers per colour;
comparisons contain 200 yes and 200 no. Passing this suite does not mean every
possible language composition is solved.

The primary fixed-endpoint `latest.pt` (refinement step 4,000) scores 4,781/4,800
(99.60%) on the same fresh broad test. The selected checkpoint was not renamed
or replaced after seeing tests. The repeat's selected step 2,000 was chosen only
by validation; it is supporting evidence, not a test-selected replacement for
the primary model.

### Colour within broader episodes

| Variant | Original | Corrected primary | Repeat |
|---|---:|---:|---:|
| Direct colour | 30/120 (25%) | 119/120 (99.17%) | 119/120 (99.17%) |
| Same-colour comparison | 54/120 (45%) | 111/120 (92.5%) | 120/120 (100%) |
| Colour after mixed actions | 16/66 (24.24%) | 66/66 (100%) | 64/66 (96.97%) |

The primary comparison score is below its 95% validation score: distraction
generalization is not perfect. Nine of its 15 broad-test errors are colour
comparisons involving additional painted objects. One direct-colour error also
copies a distractor colour. These errors are preserved, not trained away using
the final test.

### Primary regression results by family

| Family | Original correct | Corrected correct | Questions |
|---|---:|---:|---:|
| Ability | 250 | 249 | 250 |
| Belief | 350 | 350 | 350 |
| Communication | 250 | 250 | 250 |
| Mixed composition | 138 | 198 | 200 |
| Conditional | 400 | 400 | 400 |
| Delayed recall | 482 | 498 | 500 |
| Ownership | 600 | 600 | 600 |
| Property | 442 | 590 | 600 |
| Quantifier | 346 | 350 | 350 |
| Spatial | 300 | 300 | 300 |
| State tracking | 700 | 700 | 700 |
| Temporal | 300 | 300 | 300 |

There is one new ability error (0.4 percentage points); other family totals are
unchanged or improved. Besides the ten colour-related errors, the primary has
two mixed-action errors, two delayed-recall errors and that one ability error.
The four original prompts now produce `garden |`, `hall |`, `ben |`, `red |`.

### Longer-chain limits

The primary reaches 68/84 (80.95%) on five-to-seven-move state tracking and
80/100 (80%) on delayed move recall. The repeat reaches 54/84 (64.29%) and 54/100
(54%) respectively. The primary's long colour-update score is 55/72 (76.39%).
The challenge aggregate includes variants not strictly length-held-out, so it
must not be interpreted as 92% on wholly novel structures. Basic colour accuracy
is repeatable here; long-chain behaviour remains substantially more variable.

### Evidence files

Raw summaries and complete prediction logs for the primary, original, final-endpoint,
and repeat evaluations are reproducible generated artifacts and are not committed.
The aggregate and per-family measurements needed to interpret them are retained here.

## Training work

| Run | Updates executed | Target tokens executed | Measured loop seconds |
|---|---:|---:|---:|
| Original-data control, weight 1 | 4,000 | 5,642,933 | 248.86 |
| Original-data trial, weight 20 | 4,000 | 5,642,933 | 245.37 |
| Curriculum seed 42 | 8,000 | 9,706,658 | 431.29 |
| Curriculum seed 43 | 8,000 | 9,706,658 | 520.40 |
| Refinement seed 42 | 4,000 | 4,853,329 | 258.39 |
| Refinement seed 43 | 4,000 | 4,853,329 | 193.99 |

Times include periodic validation/checkpointing but not initialization or final
evaluation. Some runs overlapped, so these are not an additive wall-time benchmark.
The retained primary model's lineage includes 6,670,192 curriculum target tokens
and 4,249,654 refinement target tokens after the original 22,021,484-token base
training. Work after a selected parent checkpoint is not part of that lineage.

## Use the corrected baseline

```bash
.venv-training/bin/python -m training.infer \
  --checkpoint artifacts/runs/transformer-colour-refined-01/best.pt --interactive
```

For example: `ava paint cube two red. what colour cube two?`.
The model remains frozen during inference. Existing commands/checkpoints remain
available; the original model was not overwritten.

## Reproducibility

Training code hashes were stored in each generated run, alongside exact source
copies. Each dataset and inference report records source/checkpoint hashes. The
step-3,500 development checkpoint
was saved as an exact-byte snapshot before its run resumed; that diagnostic
report points to the retained snapshot and explains the path correction.

The simple-diagnostic generator gained prompt exclusions after the first validation
suite was generated. Both producer versions are retained in the source snapshot;
the original validation case bytes and their original hash were not changed.

Final regression verification passed all **67 distinct tests**: 42 app/data tests
in `.venv`, plus 19 learning and six model tests in `.venv-training`. Data tests
also passed in the training environment. Checks cover shifted answer masks,
padding, statement-only records, default-loss equivalence, invalid weights,
selection/tie rules, exact CPU resume, extension semantics, balanced outcomes,
case exclusions and existing gridworld/service behaviour.

All final report prediction totals, checkpoint/source/data hashes, and frozen
weight/buffer checks were reverified. The architecture, original v3 corpus and
pre-correction full-epoch checkpoint hashes are unchanged. No further model
training or checkpoint selection was driven by final test results.
