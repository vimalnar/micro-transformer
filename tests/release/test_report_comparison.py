import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.compare_experiment_reports import compare


class ReportComparisonTests(unittest.TestCase):
    def test_declared_metrics_are_compared_without_hidden_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, candidate = root / "baseline.json", root / "candidate.json"
            baseline.write_text(json.dumps({"accuracy": 0.5, "cost": {"steps": 10}}))
            candidate.write_text(json.dumps({"accuracy": 0.6, "cost": {"steps": 12}}))
            result = compare(baseline, candidate, ["accuracy", "cost.steps"])
            self.assertAlmostEqual(result["metrics"]["accuracy"]["absolute_delta"], 0.1)
            self.assertEqual(result["metrics"]["cost.steps"]["absolute_delta"], 2)

    def test_missing_or_nonnumeric_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right = root / "left.json", root / "right.json"
            left.write_text(json.dumps({"status": "passed"}))
            right.write_text(json.dumps({"status": "passed"}))
            with self.assertRaises(ValueError):
                compare(left, right, ["status"])
            with self.assertRaises(ValueError):
                compare(left, right, ["accuracy"])


if __name__ == "__main__":
    unittest.main()
