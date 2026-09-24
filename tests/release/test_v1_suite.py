import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "micro-world"))

from scripts.verify_v1_suite import verify_contracts


class V1SuiteContractTests(unittest.TestCase):
    def test_manifest_contracts_resolve_and_match_runtime(self):
        result = verify_contracts()
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(result["checks"].values()))

    def test_deferred_larger_model_prevents_final_v1_claim(self):
        manifest = json.loads((ROOT / "suite/v1/manifest.json").read_text())
        self.assertEqual(manifest["release_status"], "v1_ready_except_larger_reference_model")
        self.assertTrue(manifest["artifacts"]["larger_reference_model"]["required_for_final_v1"])

    def test_lightweight_packages_are_importable(self):
        self.assertIsNotNone(importlib.util.find_spec("micro_transformer"))
        self.assertIsNotNone(importlib.util.find_spec("micro_world"))


if __name__ == "__main__":
    unittest.main()
