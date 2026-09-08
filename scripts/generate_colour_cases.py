#!/usr/bin/env python3
"""Generate independent, balanced colour diagnostics; never a training corpus.

Labels come from the explicitly chosen final paint colour (or equality), then
the reference interpreter checks them. The neural model is not consulted.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "micro-world"))
from micro_transformer.data.generator import AGENTS, COLOURS, KINDS, LOCATIONS, derive_answer, entity_pool, VOCABULARY

VARIANTS = ("single_paint", "repaint", "distractor", "mixed_actions", "same_colour", "unknown_colour")


def fingerprint(text):
    return " ".join(re.findall(r"[a-z]+|[.?|]|\S", text.lower()))


def excluded_prompt(record):
    """Exclude by prompt alone, from either case files or generator episodes."""
    if "prompt" in record:
        return fingerprint(record["prompt"])
    tokens = record.get("tokens", [])
    if "?" in tokens:
        return " ".join(tokens[:tokens.index("?") + 1])
    return None


def build_cases(seed=46111, split="test", cases_per_variant=400, excluded=()):
    if split not in {"validation", "test"} or cases_per_variant < 4 or cases_per_variant % 4:
        raise ValueError("Use validation/test and a positive multiple of four cases per variant")
    rng = random.Random(seed)
    entities = [list(tokens) for _, tokens in entity_pool(KINDS, split)]
    if cases_per_variant > len(entities) * len(AGENTS) * len(COLOURS):
        raise ValueError("Requested count exceeds the finite unique single-paint diagnostic space")
    seen = set(excluded)
    rows = []
    for variant in VARIANTS:
        for index in range(cases_per_variant):
            for attempt in range(10000):
                obj, other = rng.sample(entities, 2)
                actor, receiver = rng.sample(AGENTS, 2)
                colour = COLOURS[index % 4]
                different = rng.choice([c for c in COLOURS if c != colour])
                event = lambda *parts: [*parts, "."]
                painted = event(actor, "paint", *obj, colour)
                question = ["what", "colour", *obj, "?"]
                expected = colour
                if variant == "single_paint":
                    statements = painted
                elif variant == "repaint":
                    statements = event(receiver, "paint", *obj, different) + painted
                elif variant == "distractor":
                    irrelevant = event(receiver, "paint", *other, different)
                    # Half target-first; half target-last. Always conflicting colours.
                    statements = painted + irrelevant if index // 4 % 2 else irrelevant + painted
                elif variant == "mixed_actions":
                    statements = (painted + event(actor, "take", *obj) +
                                  event(actor, "give", *obj, "to", receiver) +
                                  event(receiver, "drop", *obj, "at", rng.choice(LOCATIONS)))
                elif variant == "same_colour":
                    truth = index % 2 == 0
                    # Each yes/no outcome sees all four first colours equally.
                    colour = COLOURS[index // 2 % 4]
                    different = rng.choice([c for c in COLOURS if c != colour])
                    statements = (event(actor, "paint", *obj, colour) +
                                  event(receiver, "paint", *other, colour if truth else different))
                    question = ["is", *obj, "same", "colour", *other, "?"]
                    expected = "yes" if truth else "no"
                else:
                    statements = (event(actor, "move", *obj, "to", rng.choice(LOCATIONS)) +
                                  event(receiver, "paint", *other, colour))
                    expected = "unknown"
                tokens = statements + question
                key = " ".join(tokens)
                if key in seen:
                    continue
                if not set(tokens + [expected, "|"]) <= VOCABULARY:
                    raise ValueError("Unknown diagnostic token")
                if derive_answer(tokens + [expected, "|"]) != [expected]:
                    raise ValueError("Independent colour expectation disagrees with interpreter")
                seen.add(key)
                rows.append({"id": f"{variant}-{index:04d}", "task": variant,
                             "prompt": key.replace(" .", ".").replace(" ?", "?"),
                             "expected": expected + " |"})
                break
            else:
                raise ValueError(f"Could not generate a unique {variant} case; reduce count")
    rng.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=46111)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--cases-per-variant", type=int, default=400)
    parser.add_argument("--exclude-cases", type=Path, action="append", default=[],
                        help="JSONL cases or generator episodes whose question prompts must not recur")
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Output or manifest exists; choose a new path")
    excluded = set()
    sources = {}
    for path in args.exclude_cases:
        sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.open():
            prompt = excluded_prompt(json.loads(line))
            if prompt is not None:
                excluded.add(prompt)
    rows = build_cases(args.seed, args.split, args.cases_per_variant, excluded)
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    report = {"format": "micro-transformer-colour-cases-v1", "seed": args.seed, "split": args.split,
              "cases": len(rows), "cases_per_variant": args.cases_per_variant,
              "task_counts": dict(Counter(row["task"] for row in rows)),
              "answer_counts": {variant: dict(Counter(row["expected"] for row in rows if row["task"] == variant))
                                for variant in VARIANTS},
              "unique_prompts": len({fingerprint(row["prompt"]) for row in rows}),
              "excluded_sources": sources, "output_sha256": hashlib.sha256(payload).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "reference_sha256": hashlib.sha256((ROOT / "micro-world/micro_transformer/data/generator.py").read_bytes()).hexdigest(),
              "labels": "Explicit final paint/equality/absence, independently checked by interpreter"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(payload)
    with manifest_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
