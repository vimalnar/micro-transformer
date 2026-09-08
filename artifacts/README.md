# Generated artifacts

- `datasets/`: generated JSONL corpora and manifests
- `checkpoints/`: trained model weights
- `runs/`: metrics, logs, and run configuration
- `reports/`: generated evaluation reports

These outputs are kept separate from source code and are normally ignored by
version control. Commands create the required subdirectories when needed. Curated,
human-readable experiment summaries belong in `docs/results/`; raw predictions,
logs, corpora, and weights remain here or in external artifact storage.
