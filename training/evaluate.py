"""Evaluate a frozen checkpoint on a prepared validation or final-test split."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.data import EpisodeDataset
from training.metrics import evaluate
from training.runtime import load_model, seed_everything, select_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--architecture", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-episodes", type=int, help="Omit to evaluate every episode")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise ValueError("Report already exists; choose a new output path")
    seed_everything(42)
    device = select_device(args.device)
    model, checkpoint, _ = load_model(args.checkpoint, device, args.architecture, args.adapter)
    model.requires_grad_(False)
    dataset = EpisodeDataset(args.data)
    try:
        if dataset.metadata["split"] not in ("validation", "test"):
            raise ValueError("Evaluation requires a validation or test split")
        if dataset.metadata["max_episode_tokens"] - 1 > model.config.context_length:
            raise ValueError("Evaluation episodes exceed model context")
        result = {"checkpoint": str(args.checkpoint.resolve()), "training_step": checkpoint["step"],
                  "split": dataset.metadata["split"],
                  **evaluate(model, dataset, device, args.batch_size, args.max_episodes)}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
