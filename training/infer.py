"""Frozen Micro-Transformer inference: one prompt, interactive prompts, or a held-out test."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.harness import InferenceHarness, test_cases, test_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="Optional standalone LoRA adapter, used with its original base checkpoint")
    parser.add_argument("--architecture", type=Path, help="Relocated architecture file; its checksum must match the checkpoint")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompt")
    mode.add_argument("--interactive", action="store_true", help="Load once; enter independent episodes until :quit or EOF")
    mode.add_argument("--data", type=Path, help="Prepared validation/test dataset for a scored batch test")
    mode.add_argument("--cases", type=Path, help="JSONL file of custom prompt/expected regression cases")
    parser.add_argument("--report-dir", type=Path, help="New directory for summary.json and predictions.jsonl; required for batch tests")
    parser.add_argument("--max-episodes", type=int, help="Batch test limit; omit to test the full split")
    parser.add_argument("--json", action="store_true", help="Print structured inference results")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 selects deterministic greedy decoding")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if (args.data or args.cases) and not args.report_dir:
        parser.error("--data and --cases require --report-dir")
    if not (args.data or args.cases) and args.report_dir:
        parser.error("--report-dir is only used with --data or --cases")
    if not args.data and args.max_episodes is not None:
        parser.error("--max-episodes is only used with --data")
    if (args.data or args.cases) and args.temperature != 0:
        parser.error("Scored batch tests use greedy decoding; omit --temperature")
    harness = InferenceHarness(args.checkpoint, args.device, args.architecture, args.adapter, args.seed)
    if args.data:
        report = test_dataset(harness, args.data, args.report_dir, args.max_episodes, args.max_new_tokens)
        print(json.dumps(report, indent=2))
        return report
    if args.cases:
        report = test_cases(harness, args.cases, args.report_dir, args.max_new_tokens)
        print(json.dumps(report, indent=2))
        return report

    def answer(prompt):
        result = harness.predict(prompt, args.max_new_tokens, args.temperature)
        print(json.dumps(result) if args.json else result["prediction"])
        if not args.json and result["stop_reason"] != "episode_end":
            print(f"Stopped at {result['stop_reason']}; no | was generated.", file=sys.stderr)
        return result

    if args.prompt is not None:
        return answer(args.prompt)
    print("Micro-Transformer: enter one unfinished episode per line. :quit exits. Prompts do not share state.", file=sys.stderr)
    while True:
        try:
            prompt = input("mt> " if sys.stdin.isatty() else "")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip() == ":quit":
            break
        if prompt.strip():
            try:
                answer(prompt)
            except ValueError as error:
                print(f"Invalid prompt: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
