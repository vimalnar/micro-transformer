# Generated artifacts

- `datasets/`: generated JSONL corpora and manifests
- `checkpoints/`: trained model weights
- `runs/`: metrics, logs, and run configuration
- `reports/`: generated evaluation reports
- `releases/`: installed immutable reference releases

These outputs are kept separate from source code and are normally ignored by
version control. Commands create the required subdirectories when needed. Curated,
human-readable experiment summaries belong in `docs/results/`; raw predictions,
logs, corpora, and weights remain here or in external artifact storage.

The maintainer's relocated report collection is currently at
`/Volumes/EVERYTHING BACKUP/Development/micro-transformer/artifacts/reports/`.
This is local external storage, not a public download or an automatic redirect
for commands that write to this checkout. Preserve original manifests and record
path mappings separately; verify hashes before claiming reproduction. See the
[external storage guide](../docs/guides/reproducibility.md#external-artifact-storage)
and [historical experiments](../docs/results/historical-experiments.md) for the
distinction between current suite evidence and shelved agent/living-world work.

The repository ships one compact inference checkpoint under `baselines/v1.0.0/`.
Install and verify the optional complete baseline bundle without adding it to Git:

```bash
python scripts/baseline_artifacts.py install --source /path/to/micro-transformer-baseline-v1.0.0
python scripts/baseline_artifacts.py verify
```
