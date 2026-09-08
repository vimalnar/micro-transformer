import argparse
import sys
import tempfile
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[2] / "micro-world"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from micro_transformer.data import generator as mt


def arguments(output: Path, episodes: int = 300, **overrides):
    values = {
        "episodes": episodes, "output": output, "manifest": None, "seed": 91,
        "profile": "balanced", "difficulty": "mixed", "split": "train",
        "shard_size": 0, "resume": False, "overwrite": True,
        "max_duplicate_retries": 1000, "min_token_occurrences": 0,
        "require_full_coverage": False, "disjoint_from": [], "progress_every": 0,
        "max_tokens": 128,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class MicroTransformerGeneratorTests(unittest.TestCase):
    def test_vocabulary_has_stable_128_token_mapping(self):
        self.assertEqual(len(mt.VOCABULARY_ORDER), 128)
        self.assertEqual(len(set(mt.VOCABULARY_ORDER)), 128)
        self.assertEqual(mt.TOKEN_TO_ID, {token: index for index, token in enumerate(mt.VOCABULARY_ORDER)})

    def test_balanced_entity_splits_are_disjoint(self):
        groups = {name: set() for name in ("train", "validation", "test")}
        for kind in mt.KINDS:
            for identifier in mt.IDENTIFIERS:
                key = f"{kind}:{identifier}"
                groups[mt.assigned_split(key)].add(key)
        self.assertEqual({name: len(values) for name, values in groups.items()}, {"train": 205, "validation": 26, "test": 25})
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])

    def test_interpreter_rejects_semantic_contradiction(self):
        context = mt.Context(mt.random.Random(3), "train", "mixed")
        episode = mt.spatial(context, 0)
        tokens = episode.tokens[:]
        tokens[tokens.index("behind")] = "beside"
        record = {
            "id": 0, "tokens": tokens, "answer_tokens": episode.answer_tokens,
            "task": episode.task, "statement_count": episode.statement_count,
            "structural_template": episode.structural_template,
            "split": "train", "split_key": episode.split_key,
        }
        self.assertTrue(any("semantic answer mismatch" in issue for issue in mt.validate_record(record, 0)))

    def test_malformed_record_is_reported_without_crashing(self):
        record = {
            "id": 0, "tokens": ["is", "?", "yes", "|"], "answer_tokens": ["yes"],
            "task": "property", "statement_count": 0, "structural_template": "bad",
            "split": "train", "split_key": "cube:one",
        }
        self.assertTrue(any("interpreter error" in issue for issue in mt.validate_record(record, 0)))

    def test_generated_records_are_bounded_unique_and_interpretable(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "sample.jsonl"
            mt.generate(arguments(output, episodes=600, difficulty="hard"))
            count, _, seen, _, _, lengths, _, _, issues = mt.scan_records([output])
            self.assertEqual(issues, [])
            self.assertEqual(count, len(seen))
            self.assertLessEqual(max(lengths), 128)

    def test_resume_reconstructs_the_uninterrupted_corpus(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); full = root / "full.jsonl"; resumed = root / "resumed.jsonl"
            args = arguments(full, episodes=400)
            mt.generate(args)
            rows = full.read_text(encoding="utf-8").splitlines()
            partial = mt.working_output(resumed)
            partial.write_text("\n".join(rows[:137]) + "\n" + '{"id":', encoding="utf-8")
            resume_args = arguments(resumed, episodes=400, resume=True, overwrite=False)
            state = {**mt.generation_configuration(resume_args, mt.task_registry(), mt.PROFILES[resume_args.profile]),
                     "status": "generating", "completed_episodes": 137}
            mt.atomic_write_json(mt.state_path(resumed), state)
            mt.generate(resume_args)
            self.assertEqual(full.read_bytes(), resumed.read_bytes())

    def test_failed_overwrite_preserves_previous_dataset(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "data.jsonl"; output.write_text("sentinel\n", encoding="utf-8")
            args = arguments(output, episodes=20, require_full_coverage=True, min_token_occurrences=10000)
            with self.assertRaises(ValueError): mt.generate(args)
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")
            self.assertTrue(mt.state_path(output).exists())

    def test_world_rejects_self_containment(self):
        world = mt.World()
        with self.assertRaises(ValueError): world.put_inside("box:one", "box:one")


if __name__ == "__main__":
    unittest.main()
