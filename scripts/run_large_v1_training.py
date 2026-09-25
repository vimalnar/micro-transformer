#!/usr/bin/env python3
"""Launch one matched, from-scratch large-v1 run against the verified packed release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from verify_large_model_data_binding import main as verify_data_binding


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = {
    "seed-42": 42,
    "seed-43": 43,
    "seed-44": 44,
    "seed-42-replay": 42,
}
EXPECTED_PARAMETERS = 9_946_560


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", choices=RUNS)
    parser.add_argument("--resume", action="store_true", help="Resume this run from latest.pt")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and print command only")
    args = parser.parse_args()

    if verify_data_binding() != 0:
        raise SystemExit("packed release verification failed")
    config_path = ROOT / "training/configs/large-v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    from models.transformer import ModelConfig, Transformer
    model = Transformer(ModelConfig(**config))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise SystemExit(f"large model parameter count changed: {parameters}")
    del model

    binding = json.loads((ROOT / "baselines/v1.0.0-large/data-binding.json").read_text(encoding="utf-8"))
    train_dir = ROOT / binding["model_facing_data"]["train"]["directory"]
    validation_dir = ROOT / binding["model_facing_data"]["validation"]["directory"]
    run_dir = ROOT / "artifacts/runs" / f"large-v1-{args.run_id}"
    latest = run_dir / "latest.pt"
    if args.resume and not latest.is_file():
        raise SystemExit(f"no resumable checkpoint: {latest}")
    if not args.resume and run_dir.exists():
        raise SystemExit(f"run already exists: {run_dir}; use --resume only for an interrupted run")

    command = [
        sys.executable, "-m", "training.train",
        "--train-data", str(train_dir),
        "--validation-data", str(validation_dir),
        "--run-dir", str(run_dir),
        "--steps", "31250", "--batch-size", "32",
        "--seed", str(RUNS[args.run_id]),
        "--device", "cpu", "--threads", "4",
        "--learning-rate", "0.0003", "--weight-decay", "0.01",
        "--adam-epsilon", "1e-8", "--adam-foreach", "false",
        "--deterministic", "--grad-clip", "1.0",
        "--answer-weight", "1.0",
        "--selection-metric", "answer_macro_accuracy",
        "--eval-every", "1000", "--eval-episodes", "5000",
        "--log-every", "20",
    ]
    if args.resume:
        command.extend(("--resume", str(latest)))
    else:
        command.extend(("--architecture", str(ROOT / "models/transformer.py"),
                        "--model-config", str(config_path)))
    print(json.dumps({"run_id": args.run_id, "seed": RUNS[args.run_id],
                      "parameters": parameters, "run_dir": str(run_dir),
                      "command": command}), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
