# Micro-Transformer

Micro-Transformer is a compact, deterministic baseline for training and evaluating a
small transformer from scratch. It deliberately separates the language contract,
procedural data, interactive gridworld, and neural model so that each component
can be tested or replaced without hiding behaviour in a large framework.

This repository contains baseline components, not a claim of general reasoning,
persistent memory, or online learning. Normal model inference uses frozen weights.

The current checkout contains a **preliminary single-seed larger reference model**;
the remaining seed panel, held-out challenge evaluation, and external artifact
publication are still pending before a final V1 scaling claim. The machine-readable
suite contract is [`suite/v1/manifest.json`](suite/v1/manifest.json). Until those
steps are complete, the larger result must not be described as a final multi-seed
V1 release.

## Purpose

Micro-Transformer provides four complementary research components:

- **Language reference:** a fixed 128-token, English-like formal language with
  precise grammar, semantics, punctuation, and an episode terminator. It makes
  model inputs and outputs human-readable while retaining exact validation.
- **Training data:** a deterministic procedural generator that creates diverse,
  unique statement, question, and answer episodes. A reference interpreter derives
  the answers, and manifests record vocabulary, task, variant, split, and coverage
  information for reproducible experiments.
- **Micro-World gridworld:** a separate deterministic 16×16 top-down environment
  with partial observations, simple tile physics, interactions, time, day/night,
  visual observations, sound cues, replay, and a browser inspector. It provides a
  future-facing embodied test environment without coupling simulation rules to a
  model.
- **Transformer:** a self-contained PyTorch causal transformer trained from
  scratch on the formal language. The default configuration has 1,024,512
  parameters. Its architecture is intentionally small and copyable, supports
  frozen inference and optional LoRA fine-tuning, and contains no language rules.

The formal-language generator and visual gridworld are currently separate
baseline environments: the reference interpreter owns dataset truth, while the
grid simulation owns interactive world truth. See the
[system architecture](docs/specifications/system-architecture.md) for the exact
component boundaries.

## Repository guide

```text
docs/reference/                    formal language reference
micro-world/micro_transformer/    vocabulary, interpreter and data generation
micro-world/micro_world/          grid simulation and browser service
micro-world/app/                  browser inspector client
models/                           standalone transformer architecture
training/                         preparation, training, evaluation and inference
baselines/v1.0.0/                 trained reference, cards, results and sample data
docs/guides/                      experiment and Micro-World adapter guidance
harness/                          simple access to the canonical frozen checkpoint
benchmarks/v1/                    versioned language and Micro-World suite contract
suite/v1/                         machine-readable release source of truth
scripts/                          command-line entry points
artifacts/                        generated data, runs, checkpoints and reports
```

The complete language is available as a
[GitHub-readable reference](docs/reference/micro-transformer-language.md) and a
[Word edition](<docs/reference/Micro-Transformer Language Reference.docx>). Grid behaviour
is defined by [Gridworld v1](docs/specifications/gridworld-v1.md), and corpus
behaviour by [Data generation v4](docs/specifications/data-generation-v4.md). The
[composition interfaces](docs/specifications/system-architecture.md#51-composition-interfaces)
explain how experiments can connect these independent baselines.

## How the components compose

Each component exposes a small, explicit output that another component or a custom
experiment can consume:

| Component | Primary output | Typical consumer |
|---|---|---|
| Language | Stable vocabulary, grammar, and interpreter semantics | Dataset generators and token encoders |
| Dataset generator | Validated JSONL episodes and provenance manifests | Data preparation or custom evaluation |
| Training system | Prepared datasets, checkpoints, metrics, and reports | Frozen inference, LoRA, or architecture comparisons |
| Micro-World | Typed observations, actions, events, replay records, pixels, and sound cues | Browser inspection or a user-defined model adapter |
| Browser inspector | Human-readable views of service output | Debugging and experiment inspection |

No mandatory adapter joins the formal-language path to Micro-World. An experiment
may translate grid observations and actions into language tokens, train directly on
visual arrays, combine modalities, or leave the two environments separate. Such an
adapter belongs to the experiment that defines it; the baseline contracts remain
unchanged and independently testable.

## Requirements and setup

Python 3.12 is recommended. The source package and browser inspector are kept in a
lightweight environment; PyTorch training uses a separate environment so its large,
pinned dependencies do not become requirements for data generation or inspection.

From the repository root, create the application and development environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
```

Create the isolated training environment:

```bash
python3.12 -m venv .venv-training
.venv-training/bin/python -m pip install --upgrade pip
.venv-training/bin/python -m pip install -r training/requirements.txt
```

`pyproject.toml` is the authoritative application/development dependency
declaration. `training/requirements.txt` pins a publicly installable compatibility
environment for training, evaluation, and the harness. The baseline's original
Python 3.12/macOS arm64 environment is retained in
`training/requirements-lock.txt` as provenance; it is not the public installation
file. Exact archived baseline reproduction also requires the complete release.
Both `.venv/` and `.venv-training/` are deliberately ignored by Git; recreate
them from these declarations rather than committing them.

The commands below use macOS/Linux paths. On Windows, create environments with
`py -3.12 -m venv ...` and replace `.venv/bin/python` with
`.venv\Scripts\python.exe` (and likewise for `.venv-training`).

Run the test suite after setup:

```bash
.venv/bin/python -m unittest discover -s tests
.venv-training/bin/python -m unittest discover -s tests/model_architecture
.venv-training/bin/python -m unittest discover -s tests/learning
```

Verify the V1 contracts, or exercise the complete generate-to-evaluate smoke path:

```bash
.venv/bin/python scripts/verify_v1_suite.py
.venv-training/bin/python scripts/verify_v1_suite.py --full
```

The same full smoke path and complete test suite run in
`.github/workflows/ci.yml` for pushes and pull requests.

## Use the trained reference baseline

The repository includes a compact, inference-only seed-42 checkpoint from the
versioned v1.0.0 baseline. It is ready to use after installing the training
dependencies:

```bash
.venv-training/bin/python -m training.infer \
  --checkpoint baselines/v1.0.0/checkpoints/seed42-inference.pt \
  --prompt 'hal move flag six to river. hal tell gia flag six at river. where gia knows flag six?'

python scripts/baseline_artifacts.py verify-core
```

For a simple interactive demonstration using those same canonical artifacts:

```bash
.venv-training/bin/python harness/talk_to_micro_transformer.py --self-test
.venv-training/bin/python harness/talk_to_micro_transformer.py
```

The [baseline entry point](baselines/v1.0.0/README.md) also records the model and
dataset cards, fixed protocol, measured results, limitations, and installation of
the optional complete reproducibility release. The complete release contains all
seeds, exact packed data, frozen evaluations, predictions, reports, and evidence;
it remains outside ordinary Git history.

## Generate training data

Generation uses the fixed language and reference interpreter; it does not require
PyTorch or an external language model. The following creates the standard broad
train, validation, and final-test corpora. The requested count is **episodes**, not
tokens.

```bash
.venv/bin/python scripts/generate_dataset.py \
  --episodes 1000000 --split train --seed 46090 \
  --paired-counterfactual-fraction 0.25 \
  --output artifacts/datasets/v4-train-1m-pcf25.jsonl \
  --require-full-coverage --require-capability-coverage

.venv/bin/python scripts/generate_dataset.py \
  --episodes 5000 --split validation --seed 46091 \
  --output artifacts/datasets/v4-validation-5k.jsonl \
  --require-full-coverage --require-capability-coverage

.venv/bin/python scripts/generate_dataset.py \
  --episodes 5000 --split test --seed 46092 \
  --output artifacts/datasets/v4-test-5k.jsonl \
  --require-full-coverage --require-capability-coverage
```

Use smaller values for a smoke test, or add `--shard-size 100000` for a much larger
corpus. Existing outputs are protected; add `--overwrite` only when intentionally
replacing them. Validate a generated corpus at any time:

```bash
.venv/bin/python scripts/generate_dataset.py \
  --validate artifacts/datasets/v4-train-1m-pcf25.jsonl \
  --require-capability-coverage
```

JSONL records contain token strings, exact answer tokens, task metadata, and split
identity. A neighbouring manifest records configuration, hashes, coverage, balance,
uniqueness, and rejection counts. Generated corpora remain local under
`artifacts/datasets/` and are ignored by Git.

These commands create a new corpus under the current generator contract. For an
exact comparison with reference v1.0.0, use its archived packed arrays: generator
equivalence must not be assumed from matching counts and seeds alone.

## Run the Micro-World gridworld

Start the browser inspector:

```bash
.venv/bin/python scripts/run_inspector.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) and stop the server with
Ctrl-C. The main view shows the deterministic 16×16 world; the smaller view shows
the agent's forward-facing 9×9 observation. The full world is a human debugging
view and is not automatically supplied to an agent.

![Micro-World day and night concept](docs/concepts/micro-world-day-night-concept.png)

Run the bounded scenario benchmark with feasibility, wait, random, reactive, and
frozen-model policies:

```bash
.venv-training/bin/python scripts/run_microworld_benchmark.py \
  --split validation --device cpu \
  --report-dir artifacts/reports/my-micro-world-run
```

The frozen language checkpoint is not trained as an action policy. A negative or
inconclusive result is valid evidence; the runner retains complete decisions,
controls, replay identities, model hashes, and frozen-weight checks.

## Train the transformer

First convert each validated JSONL corpus into the compact, memory-mapped training
format:

```bash
.venv-training/bin/python -m training.prepare_data \
  --input artifacts/datasets/v4-train-1m-pcf25.jsonl \
  --output artifacts/datasets/prepared/v4-train-1m-pcf25

.venv-training/bin/python -m training.prepare_data \
  --input artifacts/datasets/v4-validation-5k.jsonl \
  --output artifacts/datasets/prepared/v4-validation-5k

.venv-training/bin/python -m training.prepare_data \
  --input artifacts/datasets/v4-test-5k.jsonl \
  --output artifacts/datasets/prepared/v4-test-5k
```

Train the default transformer for one shuffled pass over the training split:

```bash
.venv-training/bin/python -m training.train \
  --architecture models/transformer.py \
  --model-config training/configs/baseline.json \
  --train-data artifacts/datasets/prepared/v4-train-1m-pcf25 \
  --validation-data artifacts/datasets/prepared/v4-validation-5k \
  --run-dir artifacts/runs/baseline-01 \
  --steps 31250 --batch-size 32 --seed 42 --device cpu
```

The validation split selects checkpoints; the final-test split must remain untouched
until the run is complete. `--device auto` selects CUDA, Apple MPS, then CPU. Each
run stores its exact architecture source, configuration, dataset identities,
weights, optimizer state, metrics, and random state under its run directory.

Evaluate the frozen best checkpoint on the held-out test split:

```bash
.venv-training/bin/python -m training.evaluate \
  --checkpoint artifacts/runs/baseline-01/best.pt \
  --data artifacts/datasets/prepared/v4-test-5k \
  --output artifacts/reports/baseline-01-test.json
```

Run one prompt or start an interactive inference session:

```bash
.venv-training/bin/python -m training.infer \
  --checkpoint artifacts/runs/baseline-01/best.pt \
  --prompt 'ava move cube two to garden. where cube two?'

.venv-training/bin/python -m training.infer \
  --checkpoint artifacts/runs/baseline-01/best.pt --interactive
```

Prompts are unfinished episodes and therefore omit `|`; generation stops when the
model emits the episode terminator. The answer comes from the model, not the
reference interpreter, and may be wrong. In interactive mode every input line is
an independent episode.

For LoRA, architecture variants, resume semantics, answer-weighted objectives,
and scored batch inference, see the full [transformer training guide](training/README.md)
and the [experiment guide](docs/guides/experiment-guide.md).

The [V1 quick start](docs/guides/v1-quickstart.md) provides separate starting
paths for hobbyists, academic researchers, and product R&D teams.

## Outputs and reproducibility

Generated data, prepared arrays, full training checkpoints, logs, predictions, and
reports are written under `artifacts/` and excluded from version control because
they may be large. The repository contains source, specifications, tests,
dependency declarations, curated results, representative records, and one compact
inference-only reference checkpoint. Install the complete immutable release under
`artifacts/releases/` when exact multi-seed reproduction is needed.

Recorded baseline results and limitations are available in:

- [Initial baseline](docs/results/baseline.md)
- [Broader v4 data](docs/results/broader-data.md)
- [Colour correction](docs/results/colour-correction.md)
- [Reference baseline v1.0.0](baselines/v1.0.0/model-card.md)

Those reports document particular experiments rather than promises about every new
run. The value of the baseline is that future architectural changes can be compared
against the same explicit language, generation contract, environments, and held-out
evaluation process.

## Licence

Copyright 2026 Vimal Naran. Micro-Transformer is available under the
[Apache License 2.0](LICENSE). Generated datasets, trained weights, and third-party
materials may carry their own licence terms when distributed separately.
