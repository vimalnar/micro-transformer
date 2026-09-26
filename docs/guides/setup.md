# Install and set up Micro-Transformer

This guide takes you from a fresh checkout to the first verified run. The repository
contains a lightweight deterministic data and Micro-World implementation, plus an
optional PyTorch training stack. Keeping those environments separate avoids making
the simulator depend on the much larger training installation.

## Requirements

- Git and Python 3.12 for the documented training workflow.
- Python 3.9 or newer for the package metadata and lightweight core. The pinned
  training dependencies target Python 3.12.
- Sufficient disk space for the chosen training environment and any generated
  datasets or run artifacts; larger corpora and checkpoints can require substantial
  additional storage.
- CPU works for inference and training. CUDA or Apple MPS can be selected when
  supported by the installed PyTorch build.

The repository does not require a GPU, cloud account, external model API, or
pretrained download to run its bundled small checkpoint.

## Clone and install the core

```bash
git clone https://github.com/vimalnar/micro-transformer.git
cd micro-transformer
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
```

The `dev` extra installs FastAPI, Uvicorn, and HTTPX for the inspector and tests.
To install only the browser service dependencies, use `.[web]`; to install the
test client without the service, use `.[test]`. Core formal-language generation
does not require PyTorch.

On Windows, create the environment with `py -3.12 -m venv .venv` and invoke
`.venv\\Scripts\\python.exe` in place of `.venv/bin/python`.

## Install training and inference dependencies

Use a separate environment for PyTorch, training, evaluation, and the included
frozen-checkpoint harness:

```bash
python3.12 -m venv .venv-training
.venv-training/bin/python -m pip install --upgrade pip
.venv-training/bin/python -m pip install -r training/requirements.txt
```

The public compatibility requirements are pinned in
[`training/requirements.txt`](https://github.com/vimalnar/micro-transformer/blob/main/training/requirements.txt). The complete
original environment is recorded in `training/requirements-lock.txt` for
provenance. Runtime, PyTorch version, device, and thread settings can affect
training numerics; record the actual environment with each new run.

## Verify the setup

Run a short contract check and the bundled harness self-test:

```bash
.venv/bin/python scripts/verify_v1_suite.py
.venv-training/bin/python harness/talk_to_micro_transformer.py --self-test
```

The first command checks the versioned suite contracts without running a full
training job. The harness self-test checks its frozen model artifacts and a small
set of named predictions. It is not a comprehensive capability test.

To run all repository tests:

```bash
.venv/bin/python -m unittest discover -s tests
.venv-training/bin/python -m unittest discover -s tests/model_architecture
.venv-training/bin/python -m unittest discover -s tests/learning
```

## Start the Micro-World inspector

With the core development dependencies installed:

```bash
.venv/bin/python scripts/run_inspector.py
```

Open <http://127.0.0.1:8000>. Stop the local server with Ctrl-C. The inspector is
a human-facing view of the simulation and its permitted agent observation; its
full map is a debugging view, not an observation automatically given to an agent.

## Common setup issues

| Symptom | Likely cause and next step |
| --- | --- |
| `No module named torch` | Use `.venv-training/bin/python` and install `training/requirements.txt`. |
| Inspector reports missing FastAPI | Install `.[dev]` or `.[web]` in `.venv`. |
| CUDA or MPS is unavailable | Run with `--device cpu`, or install a PyTorch build compatible with the device. |
| Existing output directory is rejected | The tools protect outputs from accidental overwrite. Choose a new run/output path; only use an overwrite option when replacement is intentional. |
| A prompt token is rejected | The model accepts only its exact fixed vocabulary and prompt grammar. Check the [language reference](../reference/micro-transformer-language.md). |

Continue with the [quickstart](v1-quickstart.md), the [model specifications](../specifications/model-specifications.md),
or the [harness guide](harness.md).
