# Colour correction without an architecture change

## What changes

The baseline remains a 1,024,512-parameter, five-layer causal transformer over
128 language tokens. No architectural component is added. All correction happens
in offline training, and inference continues to use fixed parameters.

Three separable training/data features are available:

1. `--answer-weight`: increase supervision of answer tokens while retaining the
   ordinary next-token objective. The weight-1 default preserves old runs.
2. `--selection-metric`: select on validation QA rather than only token loss.
   `answer_macro_accuracy` averages accuracy across question variants equally;
   `answer_exact_match` averages across individual questions. Both require exact
   generated answers including `|`. Ties prefer lower ordinary token loss.
3. `scripts/tasks/colour_grounding.py`: an optional task-registry extension with
   balanced simple assignments, reassignment, conflicting distractors, unrelated
   ownership/movement, colour comparison, and unknown values. It uses the existing
   grammar and interpreter; it does not change the default v4 generator.

These are configurable experimental tools, not proof that one choice is optimal.
Weighting alone did not correct colour within the first 4,000-update comparison.
The subsequent curriculum retains all base families rather than training only on
colour. Its JSON profile assigns 70% to those families and 30% to explicit colour
grounding. Minimal episodes coexist with optional initial object placement.

## Reproduction commands

Use new destination names, or reuse locally prepared data where shown. Generated
datasets and checkpoints are not committed to the source repository.

```bash
.venv-training/bin/python scripts/generate_dataset.py \
  --episodes 128000 --split train --seed 46120 \
  --task-module scripts/tasks/colour_grounding.py \
  --profile-file training/configs/colour-curriculum-profile.json \
  --output artifacts/datasets/my-colour-curriculum.jsonl \
  --require-full-coverage --require-capability-coverage

.venv-training/bin/python -m training.prepare_data \
  --input artifacts/datasets/my-colour-curriculum.jsonl \
  --output artifacts/datasets/prepared/my-colour-curriculum

.venv-training/bin/python -m training.train \
  --init-from artifacts/runs/transformer-v4-01/latest.pt \
  --train-data artifacts/datasets/prepared/colour-curriculum-128k \
  --validation-data artifacts/datasets/prepared/v4-validation-5k \
  --run-dir artifacts/runs/my-colour-correction \
  --steps 8000 --batch-size 32 --answer-weight 20 \
  --selection-metric answer_macro_accuracy --seed 42 \
  --eval-episodes 5000 --eval-every 500 --log-every 500 --device mps

.venv-training/bin/python -m training.train \
  --init-from artifacts/runs/my-colour-correction/best.pt \
  --train-data artifacts/datasets/prepared/colour-curriculum-128k \
  --validation-data artifacts/datasets/prepared/v4-validation-5k \
  --run-dir artifacts/runs/my-colour-refinement \
  --steps 4000 --batch-size 32 --answer-weight 20 --learning-rate 0.0001 \
  --selection-metric answer_macro_accuracy --seed 42 \
  --eval-episodes 5000 --eval-every 500 --log-every 500 --device mps
```

The training command above uses the supplied prepared curriculum. Substitute your
new prepared directory to use your own generated corpus. Eight thousand updates
at batch size 32 traverse this 128,000-episode dataset twice. A run initialized
from a checkpoint gets fresh optimizer state; `--resume` instead restores optimizer,
sampler and random state. Changing objective/selection settings on resume is
rejected; branch with `--init-from` for a different training procedure.

The second training command performs the lower-learning-rate refinement used for the
corrected model. The resulting primary checkpoint is
`artifacts/runs/transformer-colour-refined-01/best.pt`. It passes 2,400/2,400 fresh
simple-colour cases and scores 99.69% on the fresh broad test; it is not perfect
on longer/distractor-heavy compositions. See the [full results](../results/colour-correction.md).

## Diagnostics and validity

`scripts/generate_colour_cases.py` creates a separate, reproducible JSONL case
suite for `training.infer --cases`. Each record has a prompt, expected continuation,
case ID and family label. Defaults are 400 cases in each of six families (2,400
total), balanced across the four colours or yes/no as appropriate. Requests are
bounded by the finite single-paint diagnostic space. They are not intended as
an unlimited training-data generator.

Expected values are constructed directly from explicit final paint/equality/
absence, then checked with the reference interpreter. The model is not used to
generate or label the cases. Use validation identities for development and test
identities only after model selection. This separates two kinds of evidence:

- Simple diagnostics ask whether the basic mapping works, independently of longer
  standard training templates.
- Broad and longer-chain corpora check composition and regression. A strong colour
  score does not imply that all longer chains or language constructions are solved.

Prepared data still uses separate episodes, right padding, once-shifted targets,
and no cross-episode attention. Answer weighting is aligned after that shift:
the first answer target is at `answer_start - 1`. Padding is never scored;
statement-only episodes keep the ordinary objective. During free-generation
evaluation, the input ends at `?`; expected answers are available only to scoring.

Run metadata records objective, selection rule, selected step/value, architecture,
dataset identities and training-source hashes. `best_validation.json` explains
why `best.pt` was selected. `best_validation_loss` in checkpoints remains the
minimum ordinary token loss for backwards compatibility, not the QA criterion.
Model/data/preparation checks and source hashes remain in force.

A correction is accepted on measured held-out evidence, not merely because these
flags or task families were added. The curated results record the completed outcome;
raw development logs and prediction files are generated artifacts.
