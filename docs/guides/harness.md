# Frozen-model harness

The repository's canonical command-line harness loads the bundled frozen
v1.0.0 seed-42 checkpoint and its matching architecture. It is a small inference
interface for the model's fixed 128-token formal language.

## Start the harness

From the repository root, after installing
[`training/requirements.txt`](https://github.com/vimalnar/micro-transformer/blob/main/training/requirements.txt):

```bash
.venv-training/bin/python harness/talk_to_micro_transformer.py --self-test
.venv-training/bin/python harness/talk_to_micro_transformer.py
```

The self-test verifies artifact identity, repeats named predictions, and checks
that inference leaves weights unchanged. Run the second command for an interactive
session. The separate standalone harness repository is historical release evidence;
this checkout's `harness/` directory is the canonical source.

## Prompt format

Each input is one self-contained episode made from the exact lowercase vocabulary.
Include any facts needed for the question, then exactly one question ending in `?`.
Do not type the `|` token; it is the generated episode/answer terminator. Prompts
must fit the 128-token context.

```text
hal move flag six to river. hal tell gia flag six at river. where gia knows flag six?
```

The model's completion is `river |`. This is a narrow query about the text supplied
in that episode. It does not imply retained conversation memory between prompts.

## Interactive commands

| Command | Function |
| --- | --- |
| `:examples` | Show four prompts with verified expected predictions for this checkpoint. |
| `:limitations` | Show two named prompts where this checkpoint's known output is wrong. |
| `:vocab` | Print the checkpoint's ordered 128-token vocabulary. |
| `:info` | Show model/checkpoint identity. |
| `:help` | Show command help. |
| `:quit` | End the process. |

The successful and failing examples are fixed regression cases, not an unbiased
sample of all possible prompts. The separate
[examples guide](examples.md) explains how to vary cases while preserving labels
and model outputs separately.

## One-shot inference from Python

For the lower-level model API, see `training.infer`:

```bash
.venv-training/bin/python -m training.infer \
  --checkpoint baselines/v1.0.0/checkpoints/seed42-inference.pt \
  --prompt 'hal move flag six to river. hal tell gia flag six at river. where gia knows flag six?'
```

To select a device, pass the supported `--device` option (`auto`, `cpu`, `cuda`,
or `mps`); auto selection uses available CUDA, then Apple MPS, then CPU. Device
availability depends on the local PyTorch build.

The model's tensor interface is more general than this CLI prompt convention.
Use the Python API for controlled generation tasks that do not match the harness's
one-question format, and define their tokenization, stopping rule, and evaluation
explicitly.

## Identity and limitations

The harness checks SHA-256 hashes for the bundled checkpoint and architecture,
checks that the checkpoint declares the expected format and matching architecture,
loads the 128-token vocabulary, and freezes model parameters. A changed artifact
will fail integrity validation instead of being silently treated as the same
release.

The harness does not provide chat memory, retrieval, tools, simulator control,
online learning, or a general natural-language interface. Greedy generation is
deterministic for the same model and runtime path; sampling with a positive
temperature is exploratory. The model can return confident but incorrect answers.
