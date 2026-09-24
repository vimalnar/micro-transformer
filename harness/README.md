# Integrated inference harness

This is the simple human-facing entry point for the frozen Micro-Transformer
v1.0.0 baseline. It uses the canonical architecture and checkpoint already in this
repository; it does not maintain copied model artifacts.

From the repository root, after installing `training/requirements.txt`:

```bash
.venv-training/bin/python harness/talk_to_micro_transformer.py --self-test
.venv-training/bin/python harness/talk_to_micro_transformer.py
```

The harness provides verified examples and preserved known failures through
`:examples` and `:limitations`. Each prompt is independent. It is not a general
chatbot and does not have retrieval, persistent memory, online learning, a
Micro-World connection, or evidence of consciousness.

The former standalone repository is retained only as historical release evidence.
This directory is now the canonical harness source.
