from collections import Counter
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.generate_colour_cases import VARIANTS, build_cases, excluded_prompt, fingerprint
from micro_transformer.data.generator import entity_keys, assigned_split, derive_answer


class ColourCaseTests(unittest.TestCase):
    def test_cases_are_deterministic_balanced_unique_and_correct(self):
        cases = build_cases(cases_per_variant=40)
        self.assertEqual(cases, build_cases(cases_per_variant=40))
        self.assertEqual(len(cases), 240)
        self.assertEqual(len({c["prompt"] for c in cases}), len(cases))
        for case in cases:
            tokens = fingerprint(case["prompt"]).split()
            expected = case["expected"].split()
            self.assertEqual(derive_answer(tokens + expected), expected[:-1])
            for entity in entity_keys(tokens):
                self.assertEqual(assigned_split(entity), "test")
        for variant in VARIANTS[:4]:
            counts = Counter(c["expected"] for c in cases if c["task"] == variant)
            self.assertEqual(sorted(counts.values()), [10] * 4)
        same = Counter(c["expected"] for c in cases if c["task"] == "same_colour")
        self.assertEqual(same, {"yes |": 20, "no |": 20})

    def test_exclusions_and_request_limits(self):
        previous = build_cases(cases_per_variant=4)
        excluded = {fingerprint(c["prompt"]) for c in previous}
        fresh = build_cases(cases_per_variant=4, excluded=excluded)
        self.assertFalse(excluded & {fingerprint(c["prompt"]) for c in fresh})
        for amount in (0, 3, 10000):
            with self.assertRaises(ValueError):
                build_cases(cases_per_variant=amount)
        self.assertEqual(excluded_prompt({"tokens": ["what", "colour", "cube", "one", "?", "red", "|"]}),
                         "what colour cube one ?")
        self.assertEqual(excluded_prompt({"prompt": "what colour cube one?"}), "what colour cube one ?")
        self.assertIsNone(excluded_prompt({"tokens": ["ava", "take", "cube", "one", ".", "|"]}))

    def test_curriculum_extension_answers_and_boolean_balance_across_splits(self):
        import random
        from micro_transformer.data import generator as api
        registry = api.task_registry([ROOT / "scripts/tasks/colour_grounding.py"])
        task = registry.tasks["colour_grounding"]
        for split in ("train", "validation", "test"):
            outcomes = {}
            for index in range(240):
                episode = task.builder(api.Context(random.Random(index), split, "mixed"), index)
                self.assertEqual(derive_answer(episode.tokens), episode.answer_tokens)
                for entity in entity_keys(episode.tokens):
                    self.assertEqual(assigned_split(entity), split)
                outcomes.setdefault(episode.variant, Counter())[tuple(episode.answer_tokens)] += 1
            self.assertEqual(outcomes["same"], {("yes",): 20, ("no",): 20})
            for variant in ("single", "repaint", "distractor", "mixed"):
                self.assertEqual(sorted(outcomes[variant].values()), [10] * 4)


if __name__ == "__main__":
    unittest.main()
