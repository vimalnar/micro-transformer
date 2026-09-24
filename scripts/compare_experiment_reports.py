#!/usr/bin/env python3
"""Compare declared numeric metrics from two Micro-Transformer reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value_at(report: dict, path: str):
    value = report
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Metric is missing: {path}")
        value = value[part]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Metric is not numeric: {path}")
    return value


def compare(baseline_path: Path, candidate_path: Path, metrics: list[str]) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    rows = {}
    for metric in metrics:
        left, right = value_at(baseline, metric), value_at(candidate, metric)
        rows[metric] = {
            "baseline": left,
            "candidate": right,
            "absolute_delta": right - left,
            "relative_delta": None if left == 0 else (right - left) / abs(left),
        }
    return {
        "format": "micro-transformer-report-comparison-v1",
        "baseline": {"path": str(baseline_path.resolve()), "sha256": digest(baseline_path)},
        "candidate": {"path": str(candidate_path.resolve()), "sha256": digest(candidate_path)},
        "metrics": rows,
        "interpretation": "Deltas are descriptive. They do not establish statistical significance, causal attribution, or larger-scale transfer.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--metric", action="append", required=True, help="Dot-separated numeric field; repeat for multiple metrics")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare(args.baseline, args.candidate, args.metric)
    if args.output:
        if args.output.exists():
            raise ValueError("Comparison output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
