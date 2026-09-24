#!/usr/bin/env python3
"""Run the versioned Micro-World benchmark and write a non-overwriting report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "micro-world"))

from micro_world.agent_adapter import FrozenModelPolicy  # noqa: E402
from micro_world.benchmark import (  # noqa: E402
    RandomPolicy,
    ReactivePolicy,
    SCENARIO_IDS,
    ScriptedPolicy,
    WaitPolicy,
    run_benchmark,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "validation", "final"), default="development")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--without-model", action="store_true", help="run controls only; useful in the lightweight environment")
    args = parser.parse_args(argv)
    manifest_path = ROOT / "benchmarks/v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    seeds = manifest["micro_world"]["splits"][args.split]
    cases = [(scenario_id, seed) for scenario_id in SCENARIO_IDS for seed in seeds]
    policies = [ScriptedPolicy(), WaitPolicy(), RandomPolicy(42), ReactivePolicy()]
    if not args.without_model:
        policies.append(FrozenModelPolicy(
            ROOT / "baselines/v1.0.0/checkpoints/seed42-inference.pt",
            ROOT / "models/transformer.py",
            device=args.device,
        ))
    report = run_benchmark(
        policies,
        cases,
        args.report_dir,
        metadata={"split": args.split, "manifest": str(manifest_path)},
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
