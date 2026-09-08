import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "micro-world"))
from micro_transformer.data import generator as g
from micro_transformer.data.coverage import Coverage


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.registry = g.task_registry()

    def episode(self, task, occurrence, seed=19, split="train", suite="standard"):
        spec = self.registry.tasks[task]
        return spec.builder(g.Context(random.Random(seed), split, "mixed", suite), occurrence)

    def record(self, episode, split="train"):
        return {**vars(episode), "schema": g.SCHEMA_VERSION, "id": 0, "split": split}

    def test_every_variant_is_valid_across_identity_splits(self):
        for split in ("train", "validation", "test"):
            for task, spec in self.registry.tasks.items():
                for occurrence in range(2 * len(spec.variants)):
                    with self.subTest(split=split, task=task, occurrence=occurrence):
                        episode = self.episode(task, occurrence, split=split)
                        self.assertEqual(g.validate_record(self.record(episode, split)), [])

    def test_boolean_variants_are_counterbalanced(self):
        for task, spec in self.registry.tasks.items():
            for index, variant in enumerate(spec.variants):
                if variant not in spec.boolean_variants:
                    continue
                pair = [self.episode(task, index + cycle * len(spec.variants)) for cycle in (0, 1)]
                self.assertEqual({tuple(x.answer_tokens) for x in pair}, {("yes",), ("no",)}, (task, variant))

    def test_semantic_golden_cases_do_not_use_generator_labels(self):
        cases = [
            ("ava move cube two to garden. ben move cube two to hall. where cube two?", ["hall"]),
            ("ava move cube two to garden. ben take cube two. who has cube two?", ["ben"]),
            ("ben take cube two. ben drop cube two at hall. who has cube two?", ["unknown"]),
            ("ben take cube two. ben give cube two to ava. ava give cube two to cy. who has cube two?", ["cy"]),
            ("ava move cube two to garden before ben move cube two to hall. where cube two?", ["hall"]),
            ("ava move cube two to garden after ben move cube two to hall. where cube two?", ["garden"]),
            ("ava lock door one. if door one locked then ava close door one else ava open door one. is door one closed?", ["yes"]),
            ("ava unlock door one. if door one locked then ava close door one else ava open door one. is door one closed?", ["no"]),
            ("ava move cube two to garden. is cube two heavy?", ["no"]),
            ("ava move cube two to garden. is cube two not heavy?", ["yes"]),
            ("ava move cube two to garden. count cube at hall?", ["none"]),
            ("ava move cube two to garden. ben move cube three to hall. is all cube at garden?", ["no"]),
        ]
        for prompt, expected in cases:
            tokens = prompt.replace(".", " . ").replace("?", " ? ").split()
            self.assertEqual(g.derive_answer(tokens + ["unknown", "|"]), expected, prompt)

    def test_repeated_updates_and_simple_take_are_explicitly_present(self):
        found_updated = found_simple = False
        for seed in range(30):
            episode = self.episode("state_tracking", 0, seed)
            key = episode.split_key.split(":")
            target_moves = []
            for statement in " ".join(episode.tokens).split(" . "):
                parts = statement.split()
                if len(parts) >= 6 and parts[1:4] == ["move", *key]:
                    target_moves.append(parts[5])
            if len(target_moves) >= 2:
                found_updated = True
                self.assertEqual(episode.answer_tokens, [target_moves[-1]])
            simple = self.episode("ownership", 0, seed)
            self.assertNotIn("give", simple.tokens)
            self.assertIn(simple.answer_tokens[0], g.AGENTS)
            found_simple = True
        self.assertTrue(found_updated and found_simple)

    def test_long_chain_challenge_is_not_the_standard_move_distribution(self):
        def moves(ep):
            target = ep.split_key.split(":")
            return sum(ep.tokens[i:i+3] == ["move", *target] for i in range(len(ep.tokens)-2))
        for seed in range(15):
            self.assertLessEqual(moves(self.episode("state_tracking", 0, seed)), 4)
            self.assertGreaterEqual(moves(self.episode("state_tracking", 0, seed, suite="challenge")), 5)

    def test_capability_report_detects_missing_negative_cases(self):
        coverage = Coverage()
        coverage.add(self.record(self.episode("conditional", 0)))
        report = coverage.report(self.registry, ["conditional"])
        self.assertFalse(report["capability_coverage_passed"])
        self.assertTrue(report["unbalanced_boolean_variants"])

    def test_extension_generation_and_plain_validation_need_no_core_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "custom.jsonl"
            args = g.build_parser().parse_args([
                "--episodes", "50", "--output", str(output),
                "--task-module", str(ROOT / "scripts/examples/location_check_task.py"),
                "--profile-file", str(ROOT / "scripts/examples/location_check_profile.json"),
                "--require-capability-coverage",
            ])
            g.generate(args)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertTrue(all(not g.validate_record(row) for row in rows))
            self.assertEqual({row["task"] for row in rows}, {"location_check"})
            manifest = json.loads(g.manifest_path(output).read_text())
            self.assertTrue(manifest["configuration"]["task_modules"])
            self.assertTrue(manifest["capability_coverage"]["capability_coverage_passed"])

    def test_sharding_and_invalid_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = g.build_parser().parse_args(["--episodes", "350", "--output", str(Path(tmp)/"shards.jsonl"), "--shard-size", "100"])
            result = g.generate(args)
            self.assertEqual(len(result["output"]), 4)
            count, _, seen, *rest = g.scan_records(map(Path, result["output"]))
            self.assertEqual((count, len(seen), rest[-1]), (350, 350, []))
            profile = Path(tmp)/"profile.json"
            profile.write_text('{"state_tracking": -1, "ownership": 101}')
            args.profile_file = profile
            with self.assertRaisesRegex(ValueError, "positive integer"):
                g.generate(args)


if __name__ == "__main__":
    unittest.main()
