#!/usr/bin/env python3
"""Verify V1 artifact identity and optionally run the complete smoke pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "micro-world"))

from micro_transformer.data.generator import (  # noqa: E402
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    SPLIT_VERSION,
    VOCABULARY_ORDER,
)
from micro_world.benchmark import BENCHMARK_VERSION, SCENARIO_IDS  # noqa: E402
from scripts.baseline_artifacts import verify_core  # noqa: E402


MANIFEST = ROOT / "suite/v1/manifest.json"
BENCHMARK_MANIFEST = ROOT / "benchmarks/v1/manifest.json"
REFERENCE_RESULTS = ROOT / "benchmarks/v1/reference-results.json"
REFERENCE_DECISIONS = ROOT / "benchmarks/v1/reference-decisions.jsonl"
READINESS = ROOT / "suite/v1/readiness.json"


def _vocabulary_hash() -> str:
    value = json.dumps(list(VOCABULARY_ORDER), separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_contracts() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("format") != "micro-transformer-suite-manifest-v1":
        raise ValueError("unsupported suite manifest")
    artifacts = manifest["artifacts"]
    required_paths = []
    for record in artifacts.values():
        if not isinstance(record, dict):
            continue
        for key in (
            "specification",
            "implementation",
            "architecture",
            "configuration",
            "checkpoint",
            "catalog",
            "entrypoint",
            "guide",
            "requirements",
            "package",
            "benchmark_manifest",
            "reference_results",
            "raw_decisions",
            "report_specification",
            "evidence",
            "audit",
            "quickstart",
            "repository_ci",
        ):
            if key in record:
                required_paths.append(record[key])
    missing = sorted(path for path in set(required_paths) if not (ROOT / path).exists())
    if missing:
        raise ValueError("suite manifest references missing paths: " + ", ".join(missing))

    language = artifacts["language"]
    generator = artifacts["data_generator"]
    checks = {
        "vocabulary_size": len(VOCABULARY_ORDER) == language["vocabulary_size"] == 128,
        "vocabulary_order": _vocabulary_hash() == language["vocabulary_order_sha256"],
        "generator_version": GENERATOR_VERSION == generator["version"],
        "generator_schema": SCHEMA_VERSION == generator["schema"],
        "split_contract": SPLIT_VERSION == generator["split_contract"],
        "baseline_core": verify_core()["status"] == "passed",
    }
    benchmark = json.loads(BENCHMARK_MANIFEST.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE_RESULTS.read_text(encoding="utf-8"))
    decisions = [
        json.loads(line)
        for line in REFERENCE_DECISIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    final_evaluation = benchmark["micro_world"]["final_evaluation"]
    checks.update({
        "benchmark_scenarios": tuple(benchmark["micro_world"]["scenario_families"]) == SCENARIO_IDS,
        "benchmark_version": BENCHMARK_VERSION == "micro-world-benchmark-v1",
        "reference_result_identity": (
            reference["experiment_id"] == final_evaluation["experiment_id"] == "REF-MW-001"
            and reference["benchmark"] == BENCHMARK_VERSION
            and reference["split"] == "final"
        ),
        "reference_result_seeds": reference["seeds"] == benchmark["micro_world"]["splits"]["final"],
        "reference_result_hash": reference["decisions_sha256"] == final_evaluation["decisions_sha256"],
        "reference_raw_decisions": (
            reference["raw_decisions"] == "benchmarks/v1/reference-decisions.jsonl"
            and _file_hash(REFERENCE_DECISIONS) == reference["decisions_sha256"]
            and {row["scenario_id"] for row in decisions} == set(SCENARIO_IDS)
            and {row["scenario_seed"] for row in decisions} == set(reference["seeds"])
            and {row["policy"] for row in decisions} == set(reference["policies"])
        ),
        "reference_result_conformance": (
            reference["scenario_families"] == len(SCENARIO_IDS)
            and reference["model"]["weights_unchanged"] is True
            and reference["states"]["execution"] == "completed"
            and reference["states"]["conformance"] == "passed"
            and reference["policies"]["scripted-feasibility"]["completion_rate"] == 1.0
        ),
        "readiness_scope": (
            readiness["status"] == "v1_ready_except_larger_reference_model"
            and readiness["final_v1_claim_permitted"] is False
            and readiness["remaining_v1_blockers"] == ["larger_reference_model"]
        ),
        "larger_model_explicitly_deferred": artifacts["larger_reference_model"]["status"] == "deferred_by_project_owner",
    })
    if not all(checks.values()):
        raise ValueError("V1 contract verification failed: " + ", ".join(key for key, value in checks.items() if not value))
    return {
        "status": "passed",
        "manifest": str(MANIFEST),
        "release_status": manifest["release_status"],
        "checks": checks,
        "referenced_paths": len(set(required_paths)),
    }


def _run(command, cwd=ROOT) -> None:
    result = subprocess.run(
        [str(value) for value in command],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            "Command failed: " + " ".join(map(str, command)) + "\n" + result.stdout + result.stderr
        )


def verify_pipeline() -> dict:
    """Exercise public commands from generation through frozen evaluation."""
    with tempfile.TemporaryDirectory(prefix="micro-transformer-v1-") as temporary:
        work = Path(temporary)
        sources = {}
        prepared = {}
        split_settings = {
            "train": (120, 7101),
            "validation": (48, 7102),
            "test": (48, 7103),
        }
        for split, (episodes, seed) in split_settings.items():
            source = work / f"{split}.jsonl"
            command = [
                sys.executable,
                ROOT / "scripts/generate_dataset.py",
                "--episodes",
                episodes,
                "--split",
                split,
                "--seed",
                seed,
                "--difficulty",
                "simple",
                "--output",
                source,
                "--progress-every",
                0,
            ]
            for earlier in sources.values():
                command.extend(["--disjoint-from", earlier])
            _run(command)
            destination = work / f"prepared-{split}"
            _run([
                sys.executable,
                "-m",
                "training.prepare_data",
                "--input",
                source,
                "--output",
                destination,
            ])
            sources[split] = source
            prepared[split] = destination

        run_dir = work / "run"
        _run([
            sys.executable,
            "-m",
            "training.train",
            "--architecture",
            ROOT / "models/transformer.py",
            "--train-data",
            prepared["train"],
            "--validation-data",
            prepared["validation"],
            "--run-dir",
            run_dir,
            "--steps",
            2,
            "--batch-size",
            8,
            "--eval-every",
            2,
            "--eval-episodes",
            16,
            "--device",
            "cpu",
            "--threads",
            1,
            "--layers",
            1,
            "--width",
            16,
            "--heads",
            2,
            "--ff-width",
            32,
        ])
        evaluation = work / "evaluation.json"
        _run([
            sys.executable,
            "-m",
            "training.evaluate",
            "--checkpoint",
            run_dir / "best.pt",
            "--data",
            prepared["test"],
            "--output",
            evaluation,
            "--max-episodes",
            16,
            "--device",
            "cpu",
        ])
        _run([
            sys.executable,
            ROOT / "harness/talk_to_micro_transformer.py",
            "--self-test",
        ])
        report = json.loads(evaluation.read_text(encoding="utf-8"))
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        return {
            "status": "passed",
            "steps": ["generate", "validate-and-prepare", "train", "reload-and-evaluate", "canonical-harness-self-test"],
            "splits": {key: {"episodes": value[0], "seed": value[1]} for key, value in split_settings.items()},
            "run_architecture_sha256": run["architecture_sha256"],
            "evaluation_split": report["split"],
            "evaluation_episodes": report["episodes"],
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="also run a fresh generated-data training/evaluation smoke path")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = {
        "format": "micro-transformer-v1-verification-v1",
        "contracts": verify_contracts(),
        "pipeline": verify_pipeline() if args.full else {"status": "not_requested"},
    }
    if args.output:
        if args.output.exists():
            raise ValueError("Verification output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
