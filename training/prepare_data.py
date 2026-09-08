"""python -m training.prepare_data --input DATA.jsonl --output PREPARED_DIR"""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.data import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit-episodes", type=int)
    args = parser.parse_args()
    result = prepare(args.input, args.output, args.limit_episodes)
    print(json.dumps({key: result[key] for key in ("episodes", "tokens", "split", "questions")}, indent=2))


if __name__ == "__main__":
    main()
