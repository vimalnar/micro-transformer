import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "micro-world"))
from micro_transformer.data import generator as g


class CounterfactualGenerationTests(unittest.TestCase):
    def generate(self, root, name, *, episodes=120, seed=91, fraction=0.25, split="train"):
        output = Path(root) / f"{name}.jsonl"
        args = g.build_parser().parse_args([
            "--episodes", str(episodes), "--output", str(output), "--seed", str(seed),
            "--split", split, "--paired-counterfactual-fraction", str(fraction),
            "--max-tokens", "0", "--overwrite",
        ])
        g.generate(args)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        manifest = json.loads(g.manifest_path(output).read_text())
        return output, rows, manifest

    def test_default_off_preserves_the_existing_seeded_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, rows, manifest = self.generate(tmp, "off", fraction=0.0)
            self.assertFalse(any("pair_id" in row for row in rows))
            self.assertEqual(
                hashlib.sha256(output.read_bytes()).hexdigest(),
                "879978d3da70443270d5ca8dccd95ee6a3895d1e34dd040b582e73987f6401c8",
            )
            self.assertEqual(manifest["paired_counterfactuals"]["paired_record_count"], 0)

    def test_fraction_validation_and_nearest_even_rounding(self):
        self.assertEqual(g.rounded_pair_count(101, 0.10), 5)
        self.assertEqual(g.rounded_pair_count(10, 0.25), 1)
        self.assertEqual(g.rounded_pair_count(11, 1.0), 5)
        for value in (-0.01, 1.01, float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                g.rounded_pair_count(100, value)

    def test_size_determinism_seed_variation_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, rows, manifest = self.generate(tmp, "first", episodes=101, seed=17, fraction=0.25)
            second, rows2, _ = self.generate(tmp, "second", episodes=101, seed=17, fraction=0.25)
            third, _, _ = self.generate(tmp, "third", episodes=101, seed=18, fraction=0.25)
            self.assertEqual(len(rows), 101)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertNotEqual(first.read_bytes(), third.read_bytes())
            diagnostics = manifest["paired_counterfactuals"]
            self.assertEqual(diagnostics["pair_count"], 13)
            self.assertEqual(diagnostics["paired_record_count"], 26)
            self.assertEqual(diagnostics["realized_paired_record_fraction"], round(26 / 101, 12))
            self.assertTrue(diagnostics["integrity_passed"])
            self.assertEqual(rows, rows2)

    def test_pair_completeness_relations_interventions_and_no_marker_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, rows, manifest = self.generate(tmp, "pairs", episodes=240, fraction=0.90)
            report = g.pair_integrity_report([output], 0.90)
            self.assertTrue(report["integrity_passed"], report["issues"])
            self.assertEqual(len({row["id"] for row in rows}), 240)
            groups = {}
            for row in rows:
                if "pair_id" in row:
                    groups.setdefault(row["pair_id"], []).append(row)
            self.assertTrue(all(len(group) == 2 for group in groups.values()))
            difference_offsets = set(); has_trailing_statement = False
            for group in groups.values():
                first, second = group
                if first["pair_relation"] == "flip":
                    self.assertNotEqual(first["answer_tokens"], second["answer_tokens"])
                    self.assertEqual(first["intervention_role"], "relevant")
                else:
                    self.assertEqual(first["answer_tokens"], second["answer_tokens"])
                    self.assertEqual(first["intervention_role"], "irrelevant")
                self.assertEqual(first["split"], second["split"])
                self.assertTrue(first["intervention_field"])
                left, right = g._model_input_without_answer(first), g._model_input_without_answer(second)
                difference = next(index for index, values in enumerate(zip(left, right)) if values[0] != values[1])
                difference_offsets.add(left.index("?") - difference)
                has_trailing_statement |= "." in left[difference + 1:left.index("?")]
            self.assertEqual(report["adjacent_pair_count"], 0)
            self.assertGreater(len(difference_offsets), 3)
            self.assertTrue(has_trailing_statement)
            self.assertTrue(all(set(row["tokens"]) <= g.VOCABULARY for row in rows))
            self.assertIsNone(manifest["paired_counterfactuals"]["model_input_pair_marker"])

    def test_pairing_spans_all_supported_profile_families_and_partitions(self):
        expected = set(g.PROFILES["balanced"]) - {"statement_only"}
        with tempfile.TemporaryDirectory() as tmp:
            _, _, manifest = self.generate(tmp, "coverage", episodes=240, fraction=0.90)
            self.assertEqual(set(manifest["paired_counterfactuals"]["pair_counts_by_task"]), expected)
            for split in ("train", "validation", "test"):
                output, rows, _ = self.generate(tmp, split, episodes=40, fraction=0.25, split=split)
                self.assertTrue(g.pair_integrity_report([output], 0.25)["integrity_passed"])
                self.assertTrue(all(row["split"] == split for row in rows))

    def test_pairing_preserves_configured_task_family_quotas(self):
        registry = g.task_registry(())
        expected = Counter(g.task_schedule(10000, g.PROFILES["balanced"], 42))
        plans = g.paired_generation_plan(
            10000, 0.25, 42, "train", registry, g.PROFILES["balanced"]
        )
        self.assertEqual(Counter(plan["task"] for plan in plans), expected)


if __name__ == "__main__":
    unittest.main()
