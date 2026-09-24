#!/usr/bin/env python3
"""Run the canonical frozen Micro-Transformer v1.0.0 baseline.

The harness is intentionally a small human-facing wrapper around the repository's
tested inference API. It does not contain copied model source or weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from training.harness import InferenceHarness
except ModuleNotFoundError as error:
    if error.name == "torch":
        raise SystemExit(
            "PyTorch is required. Install training/requirements.txt first."
        ) from error
    raise


HARNESS_VERSION = "1.1.0"
CHECKPOINT = ROOT / "baselines" / "v1.0.0" / "checkpoints" / "seed42-inference.pt"
ARCHITECTURE = ROOT / "models" / "transformer.py"
BASELINE_MANIFEST = ROOT / "baselines" / "v1.0.0" / "manifest.json"

VERIFIED_EXAMPLES = (
    (
        "communication",
        "hal move flag six to river. hal tell gia flag six at river. "
        "where gia knows flag six?",
        "river |",
    ),
    (
        "conditional state",
        "gia move gate eight to home. gia unlock gate eight. "
        "if gate eight locked then gia open gate eight else gia close gate eight. "
        "is gate eight closed?",
        "yes |",
    ),
    (
        "ownership",
        "gem ten is small. gem ten is heavy. eri move gem ten to bridge. "
        "ben move ball seven to dock. cy take ball seven. who has ball seven?",
        "cy |",
    ),
    (
        "counting",
        "box seven is small. fin move flag fourteen to room. "
        "ben move flag fourteen to forest. gia move flag fourteen to room. "
        "count flag at room?",
        "one |",
    ),
)

KNOWN_LIMITATIONS = (
    (
        "colour update",
        "gia move map twelve to cave. eri paint map twelve green. "
        "cy paint map twelve blue. what colour map twelve?",
        "blue |",
        "unknown |",
    ),
    (
        "swap followed by relocation",
        "cy move cube eight to dock. cy move orb two to lab. "
        "da swap cube eight with orb two. fin move cube eight to store. "
        "where cube eight?",
        "store |",
        "dock |",
    ),
)


class MicroTransformer:
    """Validated access to the canonical checkpoint and architecture."""

    def __init__(self, device: str = "auto", seed: int = 42):
        manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
        self.catalog = manifest
        self.harness = InferenceHarness(
            CHECKPOINT,
            device=device,
            architecture=ARCHITECTURE,
            seed=seed,
        )
        self.model = self.harness.model
        self.device = self.harness.device
        self.version = manifest["version"]
        self.step = self.harness.identity["checkpoint_training_step"]
        self.tokens = self.model.config.vocab_size

    def predict(self, prompt: str, max_new_tokens: int = 8, temperature: float = 0.0):
        result = self.harness.predict(prompt, max_new_tokens, temperature)
        return {
            **result,
            "raw_completion": result["prediction"],
        }

    def state_sha256(self) -> str:
        return self.harness.state_sha256()


def run_self_test(model: MicroTransformer) -> dict:
    before = model.state_sha256()
    checks = []
    for label, prompt, expected in VERIFIED_EXAMPLES:
        first = model.predict(prompt)
        second = model.predict(prompt)
        checks.append(
            {
                "label": label,
                "expected": expected,
                "observed": first["raw_completion"],
                "passed": first["raw_completion"] == expected,
                "repeatable": first["token_ids"] == second["token_ids"],
            }
        )
    after = model.state_sha256()
    if not all(row["passed"] and row["repeatable"] for row in checks):
        raise RuntimeError("Reference predictions failed: " + json.dumps(checks))
    if before != after:
        raise RuntimeError("Model weights or buffers changed during inference")
    return {
        "format": "micro-transformer-integrated-harness-self-test-v1",
        "status": "passed",
        "harness_version": HARNESS_VERSION,
        "model_release": model.version,
        "training_step": model.step,
        "checkpoint_sha256": model.harness.identity["checkpoint_sha256"],
        "architecture_sha256": model.harness.identity["architecture_sha256"],
        "predictions": checks,
        "repeatable": True,
        "weights_unchanged": True,
        "state_sha256": before,
    }


def _show(result: dict) -> None:
    print("MT> " + (result["answer"] or "[no answer token]"))
    if result["stop_reason"] != "episode_end":
        print(
            f"Generation stopped at {result['stop_reason']} without '|'.",
            file=sys.stderr,
        )


def _interactive(model: MicroTransformer, args: argparse.Namespace) -> None:
    print(
        f"Micro-Transformer v{model.version} on {model.device}. "
        "Each line is an independent formal-language episode."
    )
    while True:
        try:
            value = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if value in (":quit", ":q", ":exit"):
            return
        if value == ":examples":
            for label, prompt, expected in VERIFIED_EXAMPLES:
                print(f"[{label}] {prompt}\nexpected: {expected}")
        elif value == ":limitations":
            for label, prompt, expected, observed in KNOWN_LIMITATIONS:
                print(f"[{label}] {prompt}\nexpected: {expected} model: {observed}")
        elif value == ":info":
            print(json.dumps({
                "harness_version": HARNESS_VERSION,
                "model_release": model.version,
                **model.harness.identity,
            }, indent=2))
        elif value == ":help":
            print(":examples :limitations :info :help :quit")
        elif value.startswith(":"):
            print(f"Unknown command: {value}")
        elif value:
            try:
                _show(model.predict(value, args.max_new_tokens, args.temperature))
            except ValueError as error:
                print(f"Input error: {error}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {HARNESS_VERSION}")
    parser.add_argument("--prompt")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        model = MicroTransformer(args.device, args.seed)
        if args.self_test:
            result = run_self_test(model)
            print(json.dumps(result, indent=2) if args.json else "Self-test passed.")
            return 0
        if args.prompt is not None:
            result = model.predict(args.prompt, args.max_new_tokens, args.temperature)
            print(json.dumps(result) if args.json else result["prediction"])
            return 0
        _interactive(model, args)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
