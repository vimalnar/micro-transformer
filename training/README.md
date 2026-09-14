# Micro-Transformer Training

This is a vanilla next-token transformer baseline. Model size is configurable.
The complete architecture is `models/transformer.py`; this directory owns data
preparation, training, inference, evaluation, and checkpoints. Normal inference
uses fixed weights. LoRA is an optional, separate fine-tuning mode.

## Setup

Use Python 3.12 in a separate training environment. The existing Python 3.9 web
environment can continue running the inspector independently.

```bash
python3.12 -m venv .venv-training
.venv-training/bin/python -m pip install -r training/requirements.txt
```

The local training environment has already been created. On this machine its
Python came from the bundled Codex runtime; using another Python 3.12 installation
is fine. `requirements.txt` pins the tested direct dependencies; the complete
tested environment is recorded in `requirements-lock.txt`.

## Trained reference

`baselines/v1.0.0/checkpoints/seed42-inference.pt` is the compact frozen checkpoint
from the corrected one-million-record vanilla baseline. It is suitable for immediate
inference and as a base for LoRA. It deliberately omits optimizer and RNG state, so
use a full checkpoint from the optional release for exact training continuation.

```bash
.venv-training/bin/python -m training.infer \
  --checkpoint baselines/v1.0.0/checkpoints/seed42-inference.pt \
  --prompt 'hal move flag six to river. hal tell gia flag six at river. where gia knows flag six?'
```

The catalog, model card, dataset card, fixed results, and complete-release workflow
are under [`baselines/v1.0.0`](../baselines/v1.0.0/README.md).

Run the following commands from the project root. `--device auto` selects CUDA,
then Apple MPS, then CPU. Use `--device cpu` for a portable reference run.
Training uses float32 and PyTorch's scaled-dot-product attention. GPU kernel
selection depends on the installed PyTorch and device. Compilation, mixed
precision, and quantization are not prerequisites for this small baseline.

## Data preparation

For new work, generate the broader v4 corpus and its prepared splits. See
[Data generation v4](../docs/specifications/data-generation-v4.md) for task coverage,
capability extensions and generation commands. The standard local paths are:

```text
artifacts/datasets/prepared/v4-train-1m-pcf25
artifacts/datasets/prepared/v4-validation-5k
artifacts/datasets/prepared/v4-test-5k
artifacts/datasets/prepared/v4-challenge-3k
```

The older v3 commands below are retained for reproducing the initial trial. They
do not provide the broader v4 coverage. Substitute the corresponding v4 directories
in the training/evaluation commands for new runs. Prepared format v2 additionally
preserves task variant, version and suite annotations for the inference reports;
the reader remains compatible with prepared format v1.

The generator's JSONL records contain token strings and expected answers. Prepare
each split once; training then reads compact token bytes through memory maps.
The converter validates syntax/answers using the existing interpreter, checks
duplicates, preserves episode boundaries, and records source and output hashes.
It refuses to overwrite an existing prepared directory.

```bash
.venv-training/bin/python -m training.prepare_data \
  --input artifacts/datasets/micro-transformer-train-500k.jsonl \
  --output artifacts/datasets/prepared/train-500k

python3 scripts/generate_dataset.py --episodes 1000 --split validation \
  --seed 31415 --output artifacts/datasets/micro-transformer-validation-1k.jsonl \
  --require-full-coverage
python3 scripts/generate_dataset.py --episodes 1000 --split test \
  --seed 27182 --output artifacts/datasets/micro-transformer-test-1k.jsonl \
  --require-full-coverage

.venv-training/bin/python -m training.prepare_data \
  --input artifacts/datasets/micro-transformer-validation-1k.jsonl \
  --output artifacts/datasets/prepared/validation-1k
.venv-training/bin/python -m training.prepare_data \
  --input artifacts/datasets/micro-transformer-test-1k.jsonl \
  --output artifacts/datasets/prepared/test-1k
```

Skip preparation when the requested local output already exists. Each new split
contains 1,000 episodes and covers all 128 vocabulary tokens. The generator's
kind/identifier split contract separates their entities. This split tests unseen
identity combinations; it does not certify broader reasoning or rule learning.
Existing task shortcuts and reference/parser differences remain limitations of
the data, even when interpreter validation passes.

## Train a baseline

```bash
.venv-training/bin/python -m training.train \
  --architecture models/transformer.py \
  --model-config training/configs/baseline.json \
  --train-data artifacts/datasets/prepared/v4-train-1m-pcf25 \
  --validation-data artifacts/datasets/prepared/v4-validation-5k \
  --run-dir artifacts/runs/baseline-01 \
  --steps 31250 --batch-size 32 --seed 42 --device cpu
```

The default configuration is five layers, width 128, four attention heads,
feed-forward width 512, context length 128, and no dropout. It has exactly
1,024,512 parameters. There are 128 language output classes and one input-only
padding ID (128). Input and output embeddings share weights.

One epoch is one shuffled pass through all training episodes, including the final
partial batch. Alternatively, `--steps 100` performs 100 optimizer updates. A step
count measures training work; it is not a parameter count or episode count.

Use `--layers`, `--width`, `--heads`, `--ff-width`, `--context-length`, and
`--dropout` to override individual configuration fields. Width must be divisible
by the head count. For example, a smaller run can use:

```text
--layers 3 --width 96 --heads 4 --ff-width 384
```

Parameter counts are calculated from the constructed model and printed before
training. Custom configuration fields can be supplied in a JSON file if the copied
architecture's `ModelConfig` defines them. The current Micro-Transformer data adapter
requires 128 output classes; vocabulary extension is a separate data change.

Each batch holds independent episodes with right padding. Targets are shifted by
one token, padding targets are ignored, and attention cannot cross episodes.
The objective is next-token cross-entropy over statements, questions, answers, and
the episode terminator. It does not use task labels or expected answers as extra
input features. The first token is supplied as context, without a separate BOS token.

`--answer-weight 20` optionally multiplies answer-token and answer-end `|` losses
by 20, while other non-padding tokens keep weight 1. The loss is divided by the
sum of those weights. The question mark remains an ordinary token; statement-only
episodes have no boosted span. Default `--answer-weight 1` exactly preserves the
original objective. This is training supervision, not additional inference input
or an architectural change.

For question answering, use `--selection-metric answer_macro_accuracy` to select
checkpoints by the unweighted mean of per-variant exact-generation accuracy.
Alternatively, `answer_exact_match` weights each question equally. Equal answer
scores are resolved by lower ordinary validation token loss. Both use validation
data only, with prompts ending at `?`; answers and `|` must be freely generated.
The default `--selection-metric loss` preserves historical runs. Metrics include
ordinary token loss, answer-only token loss, overall QA and per-variant QA.

The retested checkpoint is produced locally at
`artifacts/runs/transformer-colour-refined-01/best.pt` by the documented run. The
[colour correction guide](../docs/specifications/colour-correction.md) contains the
full two-stage reproduction commands and links to measured results and limits.

An optional targeted colour curriculum is available through the existing generator
extension interface:

```bash
python3 scripts/generate_dataset.py --episodes 128000 --split train --seed 46120 \
  --task-module scripts/tasks/colour_grounding.py \
  --profile-file training/configs/colour-curriculum-profile.json \
  --output artifacts/datasets/my-colour-curriculum.jsonl \
  --require-full-coverage --require-capability-coverage
```

The mix retains 70% broad base tasks and adds 30% explicit colour grounding.
Core vocabulary, semantics and default generation profiles remain unchanged.
`scripts/generate_colour_cases.py` separately creates balanced evaluation cases;
those diagnostic files are not training data.

For ordinary or extension-backed generation, an optional flag such as
`--paired-counterfactual-fraction 0.10` places approximately 10% of the fixed
final record count in controlled flip/invariant pairs. Pairing is off by default
and pair metadata is not encoded into model input. The [data generation
specification](../docs/specifications/data-generation-v4.md) documents rounding,
supported variants, integrity checks, and the current no-resume limitation for
pair-enabled runs.

Optimizer: AdamW, learning rate 0.0003, weight decay 0.01, gradient clipping 1.0.
These are configurable starting settings, not empirically selected optimums.

## Checkpoints and resume

Each run directory contains:

- `architecture.py`: an exact copy of the selected model file.
- `run.json`: model settings, parameter counts, data hashes, and runtime versions.
- `initial_validation.json`: loss before the run's first update.
- `metrics.jsonl`: training loss, measured throughput, and validation metrics.
- `latest.pt`: the latest evaluated checkpoint, including optimizer and RNG state.
- `best.pt`: the checkpoint selected by `--selection-metric` (token loss by default).
- `best_validation.json`: selected validation metrics and selection rule/value/step.
- `final_validation.json`: loss and generated-answer accuracy for `latest.pt`.
- `*.adapter.pt`: adapter-only files for LoRA runs.

Checkpoints are saved at `--eval-every` updates (default 200) and at completion.
Use `--save-steps 3200` to retain an additional named `step-003200.pt` checkpoint
without overwriting it as training continues. This also schedules validation at
that step and supports comparison at a preselected training duration.
An interruption loses at most the work since the last saved checkpoint. Resume
requires the same data, model, optimizer settings, batch size, seed, and runtime.
Supply the same nondefault settings when resuming:

```bash
.venv-training/bin/python -m training.train \
  --resume artifacts/runs/baseline-01/latest.pt \
  --train-data artifacts/datasets/prepared/train-500k \
  --validation-data artifacts/datasets/prepared/validation-1k \
  --run-dir artifacts/runs/baseline-01 --steps 20000
```

`--steps` and `--epochs` are total targets, including previously completed work.
CPU resume has a test for exact numerical equality with an uninterrupted run.
Seeds and RNG states are saved on MPS/CUDA too, but bitwise equality across
different devices or software versions is not promised. The full reproducible
model artifact is the checkpoint together with its `architecture.py`.

Use `--init-from PATH` and a new run directory to begin ordinary full fine-tuning
from existing weights with a fresh optimizer. Checkpoint initialization uses the
checkpoint's saved model dimensions; it does not resize trained weight matrices.

## Copy and modify an architecture

```bash
cp models/transformer.py models/my_attention.py
```

Edit `CausalSelfAttention`, `FeedForward`, or `TransformerBlock` in the copy, then
train it with `--architecture models/my_attention.py` and a new `--run-dir`.
Use `--model-config` to supply any additional fields introduced by that variant.

The shared interface is intentionally small:

- `ModelConfig` is a dataclass accepting the configuration as keyword arguments.
- `Transformer(config)` returns a PyTorch module with a `.config` attribute.
- `forward(input_ids)` accepts `[batch, time]` integer IDs and returns
  `[batch, time, vocab_size]` logits. Input episodes are independent and right-padded.
- `generate(input_ids, max_new_tokens, eos_id, temperature=0.0)` accepts unpadded,
  equal-length prompts and returns prompt plus generated tokens. Greedy decoding
  stops at `|`. The baseline refuses to silently crop an oversized context.
- For optional LoRA, retain `enable_lora`, `lora_config`, and `adapter_state_dict`.

Configuration, layers, initialization, forward computation, generation, and LoRA
are all in the one model file. It imports no Micro-World simulator,
Micro-Transformer interpreter, training helper, or sibling model file. Training
and inference load the requested file dynamically. Inference normally loads the
architecture snapshot saved with the checkpoint. A manually supplied
`--architecture` must match that snapshot's hash, preventing accidental use of
changed code with old weights.

## LoRA fine-tuning

Train a base checkpoint first, then attach a small optional adapter:

```bash
.venv-training/bin/python -m training.train \
  --init-from baselines/v1.0.0/checkpoints/seed42-inference.pt \
  --lora-rank 4 --lora-alpha 8 --lora-targets q_proj v_proj \
  --train-data artifacts/datasets/prepared/v4-train-1m-pcf25 \
  --validation-data artifacts/datasets/prepared/v4-validation-5k \
  --run-dir artifacts/runs/lora-01 --steps 100
```

Only the adapter matrices are optimized. The base weights remain fixed, and a
newly attached adapter initially reproduces the base output. Other supported
targets are `k_proj`, `out_proj`, `up_proj`, and `down_proj`. Full checkpoints can
be used directly; adapter-only files require the exact original base checkpoint,
whose SHA-256 is checked on load. LoRA is optional at this size; full fine-tuning
is also supported. `model.merge_lora()` can merge the adapter into ordinary linear
layers for custom inference/export code.

## Inference and evaluation

```bash
.venv-training/bin/python -m training.infer \
  --checkpoint artifacts/runs/baseline-01/best.pt \
  --prompt 'ava move cube two to garden. where cube two?'

.venv-training/bin/python -m training.infer \
  --checkpoint artifacts/runs/baseline-01/best.pt \
  --adapter artifacts/runs/lora-01/best.adapter.pt \
  --prompt 'ava move cube two to garden. where cube two?'

.venv-training/bin/python -m training.evaluate \
  --checkpoint artifacts/runs/baseline-01/best.pt \
  --data artifacts/datasets/prepared/test-1k \
  --output artifacts/reports/baseline-01-test.json
```

Inference prints the generated continuation, not an interpreter-computed answer.
It uses frozen weights and greedy decoding by default. The model may be wrong.
Set `--temperature` for sampling. Provide one unfinished episode and omit `|`.
Unknown tokens, empty prompts, completed episodes, and oversized prompts are
rejected. Generation never crops the prompt: if the context fills before `|`, the
harness reports `context_limit` and returns the incomplete continuation.

### Reusable inference harness

`training/harness.py` loads the checkpoint once. `training/infer.py` exposes it as
a one-prompt CLI, an interactive CLI, or a scored batch harness. For the completed
trial checkpoint, run:

```bash
.venv-training/bin/python -m training.infer \
  --checkpoint artifacts/runs/transformer-trial-01/best.pt --interactive
```

Enter one unfinished episode per line; `:quit` or Ctrl-D exits. Each line is an
independent prompt, not an ongoing conversation. Include all relevant statements
on that line. Use `--json` for token IDs, timing, and the explicit stop reason.

Test a prepared held-out split and retain every prediction:

```bash
.venv-training/bin/python -m training.infer \
  --checkpoint artifacts/runs/transformer-trial-01/best.pt \
  --data artifacts/datasets/prepared/test-1k \
  --report-dir artifacts/reports/my-test
```

The report directory must be new. `summary.json` records checkpoint, architecture,
dataset and adapter identities, token metrics, exact question accuracy by task,
and frozen-weight/repeatability checks. `predictions.jsonl` records every question's
prompt, expected answer, actual output, and pass/fail. Statement-only episodes are
included in token metrics and explicitly excluded from question accuracy.

For targeted regression tests, `--cases` accepts a JSONL file with `prompt` and
`expected` strings, and optional `id` and `task`. Expected answers may omit the
final `|`; the scorer adds it. They are labels supplied by the test author, never
passed into the model. The checked-in trial examples have interpreter-verified
labels:

```bash
.venv-training/bin/python -m training.infer \
  --checkpoint artifacts/runs/transformer-trial-01/best.pt \
  --cases training/examples/trial_cases.jsonl \
  --report-dir artifacts/reports/my-cases
```

A completed test returns successfully even when model answers are wrong; inspect
`correct` and `answer_exact_match`. Execution errors, data corruption, and changes
to model state during testing are errors. Greedy decoding is required for scored
tests. Reports are not overwritten.

Python callers can reuse the same harness:

```python
from training.harness import InferenceHarness

harness = InferenceHarness("artifacts/runs/transformer-trial-01/best.pt")
result = harness.predict("ava move cube two to garden. where cube two?")
print(result["answer"])
```

Training selects checkpoints using the configured validation metric on a fixed prefix of the
validation split (`--eval-episodes`, default 512). Training never accepts the test
split as validation. Standalone evaluation uses the entire selected split unless
`--max-episodes` is supplied. It reports next-token loss/accuracy plus exact answer
accuracy by task. Exact answers are generated from the prompt ending at `?`, with
no stored answer in the prompt, and must match the answer and its `|` terminator.
For prepared v2 data, the inference harness also reports accuracy by task variant
and includes variant/suite annotations with each prediction.
The current fixed-vocabulary dataset uses answers of one or two tokens; generation
allows up to eight tokens, subject to context capacity. Results from these
templates should be interpreted within their known shortcut limitations.

## Verification and initial artifacts

```bash
.venv-training/bin/python -m unittest discover -s tests/model_architecture -p 'test_*.py' -v
.venv-training/bin/python -m unittest discover -s tests/learning -p 'test_*.py' -v
```

The tests check causality, padding isolation, finite gradients, unchanged inference
weights, configuration sizing, checkpoint resume, copied architecture execution,
LoRA freezing/merging/loading, data corruption, and answer-prompt separation.

The documented 60-step MPS smoke run is an implementation check, not a completed
baseline experiment. A small LoRA smoke run also checks adapter execution. Generated
runs are local artifacts and are not committed.

The documented 3,200-update trial tested the complete training → checkpoint reload
→ frozen inference path. See [the trial report](../docs/results/baseline.md) for commands,
metrics, and limitations. The selected checkpoint scored 95.3% on the generator's
held-out questions but failed two of four small manual diagnostic cases. Treat it
as a working trial model, not proof that the entire language has been learned.

Implementation references:
[PyTorch attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
[MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html),
[serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html).
