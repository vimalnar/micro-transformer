# Reproducibility and reporting

Reproducibility requires identifying the implementation, data, model, runtime and
evaluation—not only sharing a command or a final score. The repository's result
reports and manifests are designed to preserve those identities.

## Record for every run

| Identity | Record |
| --- | --- |
| Source | Repository, commit, dirty changes, architecture file hash, configuration. |
| Data | Generator/configuration, split, seed, manifest and content hashes, prepared format. |
| Model | Initialization/checkpoint identity, vocabulary order, parameter count, training step, adapter state. |
| Runtime | Python, PyTorch, operating system, device, precision, threads and deterministic settings. |
| Protocol | Hypothesis, comparison, controls, endpoints, exclusions, stopping rule, selection rule and budgets. |
| Evaluation | Suite/version/hash, protected status, per-case predictions, aggregate and subgroup metrics. |
| Outcome | Execution status, conformance, validity, review state, result interpretation and unresolved alternatives. |

Keep generated files under `artifacts/`; that directory is ignored by Git. Preserve
manifests and reports in the durable experiment record used for the work rather
than committing large transient outputs by default.

## External artifact storage

Generated reports can be retained outside the checkout. In the maintainer's
current local layout, the relocated report directory is:

```text
/Volumes/EVERYTHING BACKUP/Development/micro-transformer/artifacts/reports/
```

This path is local to the maintainer's mounted external volume; it is not a public
artifact download or a required installation location. A fresh clone does not
include these ignored reports. The normal output convention remains
`artifacts/reports/` relative to the checkout, so existing commands may still
write there unless their documented output option specifies another location.

When inspecting a relocated report, locate its run directory under the external
root and use its manifest to identify the checkpoint, source, data and controls.
Recorded absolute paths may refer to the original checkout. Record the old-to-new
path mapping separately and verify content hashes before claiming reproduction;
do not rewrite original manifests merely to make old paths resolve. Mounting or
moving a directory does not establish artifact completeness or scientific validity.

Some retained reports belong to [historical, shelved experiments](../results/historical-experiments.md)
whose implementation is absent from the current baseline. Keep their source and
evaluation identities separate from current suite results.

## Separate result states

The fact that a command completed does not establish that the protocol was followed
or that the result supports the hypothesis. Report separately:

1. **Execution:** Did the run complete and emit its expected artifacts?
2. **Conformance:** Did it use the specified data, model, controls and endpoints?
3. **Validity:** Do the controls and measurements support the intended inference?
4. **Review:** Has the result received the required independent or programme review?
5. **Outcome:** Was the hypothesised effect observed, absent, mixed or inconclusive?

The [experiment report specification](../specifications/experiment-report-v1.md)
defines the report contract. Keep negative and inconclusive outcomes intact.

## Reproducing the small reference

The small model card describes the named three-seed baseline, while the bundled
seed-42 checkpoint is an inference-only artifact. For exact training reproduction,
use the complete release's packed arrays and protocol artifacts; a newly generated
dataset with matching record counts and seeds is not automatically byte-identical.
Verify core artifact hashes with:

```bash
.venv-training/bin/python scripts/baseline_artifacts.py verify-core
```

The full release is distributed separately from ordinary repository history. The
larger model's data-binding file currently records a locally prepared, unpublished
bundle and does not provide a public download URL.

## Protect the final evaluation

Use validation to select configurations and checkpoints. Once a final suite has
been inspected or used for tuning, treat it as exposed regression evidence and
collect new held-out cases for a fresh claim. Report task-family and seed-level
results, not only the best aggregate. Keep challenge and shortcut controls beside
the standard score; do not omit inconvenient outcomes.

For architecture comparisons, use matched from-scratch runs. A LoRA fine-tune
answers an adaptation question. A comparison between a modified model and a
baseline trained under different data, seeds or compute does not isolate the
architecture change.

## What reproduction can establish

Replaying the same code and artifacts can support a narrow claim about a named
system and task. It does not establish general reasoning, transfer to another model
family, agency, consciousness, or an internal human-like capacity from task labels
such as memory or belief. State the observed result and the inference boundary
separately.
