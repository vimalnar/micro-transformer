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

The repository ships one compact inference checkpoint under `baselines/v1.0.0/`.
Install and verify the optional complete baseline bundle without adding it to Git:

```bash
python scripts/baseline_artifacts.py install --source /path/to/micro-transformer-baseline-v1.0.0
python scripts/baseline_artifacts.py verify
```
