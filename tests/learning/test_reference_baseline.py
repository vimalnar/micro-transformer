import hashlib
import json
from pathlib import Path
import unittest

try:
    import torch
    from training.harness import InferenceHarness
except ModuleNotFoundError:  # The lightweight application environment omits PyTorch.
    torch = None
    InferenceHarness = None


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "baselines" / "v1.0.0"


class ReferenceBaselineIdentityTests(unittest.TestCase):
    def test_cataloged_core_artifacts_match(self):
        manifest = json.loads((BASELINE / "manifest.json").read_text())
        architecture = ROOT / "models" / "transformer.py"
        checkpoint = BASELINE / manifest["included_checkpoint"]["path"]
        self.assertEqual(hashlib.sha256(architecture.read_bytes()).hexdigest(), manifest["architecture"]["sha256"])
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), manifest["included_checkpoint"]["sha256"])


@unittest.skipIf(torch is None, "PyTorch is installed only in the training environment")
class ReferenceBaselineInferenceTests(unittest.TestCase):
    def test_included_checkpoint_predicts(self):
        checkpoint = BASELINE / "checkpoints" / "seed42-inference.pt"
        harness = InferenceHarness(checkpoint, torch.device("cpu"), seed=42)
        result = harness.predict(
            "hal move flag six to river. hal tell gia flag six at river. "
            "where gia knows flag six?",
            8,
            0.0,
        )
        self.assertEqual(result["prediction"], "river |")
        self.assertEqual(result["stop_reason"], "episode_end")

    def test_known_simple_location_miss_is_preserved(self):
        checkpoint = BASELINE / "checkpoints" / "seed42-inference.pt"
        harness = InferenceHarness(checkpoint, torch.device("cpu"), seed=42)
        result = harness.predict("ava move ball seven to cave. where ball seven?", 8, 0.0)
        self.assertEqual(result["prediction"], "unknown |")


if __name__ == "__main__":
    unittest.main()
