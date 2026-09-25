import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class LargeReferenceStageTests(unittest.TestCase):
    def test_large_configuration_has_declared_parameter_count(self):
        from models.transformer import ModelConfig, Transformer

        config = json.loads((ROOT / "training/configs/large-v1.json").read_text())
        model = Transformer(ModelConfig(**config))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 9_946_560)
        self.assertEqual(model.config.vocab_size, 128)

    def test_public_manifest_marks_unfinished_release_explicitly(self):
        manifest = json.loads((ROOT / "baselines/v1.0.0-large/manifest.json").read_text())
        self.assertEqual(manifest["status"], "preliminary_single_seed")
        self.assertEqual(len(manifest["training"]["pending_runs"]), 3)
        self.assertFalse(manifest["preliminary_results"]["challenge_evaluated"])
        self.assertIsNone(manifest["publication"]["download_url"])


if __name__ == "__main__":
    unittest.main()
