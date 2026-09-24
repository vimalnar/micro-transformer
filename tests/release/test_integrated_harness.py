import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from harness import talk_to_micro_transformer as harness
except ModuleNotFoundError:
    torch = None
    harness = None


class HarnessSourceTests(unittest.TestCase):
    def test_suite_manifest_points_to_canonical_harness(self):
        manifest = json.loads((ROOT / "suite/v1/manifest.json").read_text())
        record = manifest["artifacts"]["inference_harness"]
        self.assertEqual(record["entrypoint"], "harness/talk_to_micro_transformer.py")
        self.assertTrue(record["uses_canonical_model_artifacts"])

    def test_standalone_artifacts_match_canonical_baseline(self):
        external = ROOT.parent / "micro-transformer-harness"
        if not external.is_dir():
            self.skipTest("historical standalone checkout is not present")
        pairs = (
            (ROOT / "models/transformer.py", external / "transformer.py"),
            (
                ROOT / "baselines/v1.0.0/checkpoints/seed42-inference.pt",
                external / "micro-transformer-v1.0.0.pt",
            ),
        )
        for canonical, historical in pairs:
            with self.subTest(path=canonical.name):
                self.assertEqual(
                    hashlib.sha256(canonical.read_bytes()).hexdigest(),
                    hashlib.sha256(historical.read_bytes()).hexdigest(),
                )


@unittest.skipIf(torch is None, "training environment required")
class IntegratedHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = harness.MicroTransformer(device="cpu")

    def test_canonical_identity_and_frozen_state(self):
        self.assertEqual(self.model.version, "1.0.0")
        self.assertEqual(self.model.step, 31250)
        self.assertEqual(sum(p.numel() for p in self.model.model.parameters()), 1_024_512)
        self.assertFalse(any(p.requires_grad for p in self.model.model.parameters()))

    def test_examples_and_known_failures(self):
        for label, prompt, expected in harness.VERIFIED_EXAMPLES:
            with self.subTest(label=label):
                self.assertEqual(self.model.predict(prompt)["raw_completion"], expected)
        for label, prompt, expected, observed in harness.KNOWN_LIMITATIONS:
            with self.subTest(label=label):
                result = self.model.predict(prompt)["raw_completion"]
                self.assertEqual(result, observed)
                self.assertNotEqual(result, expected)

    def test_self_test_and_cli(self):
        self.assertEqual(harness.run_self_test(self.model)["status"], "passed")
        result = subprocess.run(
            [sys.executable, str(ROOT / "harness/talk_to_micro_transformer.py"), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn(harness.HARNESS_VERSION, result.stdout)


if __name__ == "__main__":
    unittest.main()
