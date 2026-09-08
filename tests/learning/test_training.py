import contextlib
import importlib.util
import io
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "micro-world"))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from micro_transformer.data.generator import Context, state_tracking
    from training.data import EpisodeDataset, collate, prepare
    from training.metrics import evaluate
    from training.runtime import load_model, read_checkpoint
    from training.train import parser, train


@unittest.skipUnless(HAS_TORCH, "Install training/requirements.txt for training tests")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for split in ("train", "validation"):
            path = self.root / (split + ".jsonl")
            with path.open("w") as stream:
                for index in range(16):
                    episode = state_tracking(Context(random.Random(index), split, "simple"), index)
                    record = {**vars(episode), "id": index, "split": split}
                    stream.write(json.dumps(record) + "\n")
            prepare(path, self.root / split)

    def tearDown(self):
        self.temp.cleanup()

    def run_training(self, name, steps, extra=(), tiny=True):
        arguments = ["--train-data", str(self.root / "train"), "--validation-data", str(self.root / "validation"),
                     "--run-dir", str(self.root / name), "--steps", str(steps), "--device", "cpu", "--threads", "1",
                     "--batch-size", "4", "--eval-episodes", "8", "--eval-every", "2", "--log-every", "2"]
        if tiny:
            arguments += ["--layers", "1", "--width", "16", "--heads", "2", "--ff-width", "32", "--dropout", "0.1"]
        with contextlib.redirect_stdout(io.StringIO()):
            return train(parser().parse_args(arguments + list(extra)))

    def test_preparation_and_collation_preserve_episode_boundaries(self):
        dataset = EpisodeDataset(self.root / "train")
        try:
            a, b = dataset[0], dataset[1]
            x, y = collate([a, b], 128, "cpu")
            for row, record in enumerate((a, b)):
                tokens = record[0]
                torch.testing.assert_close(x[row, :len(tokens)-1], tokens[:-1])
                torch.testing.assert_close(y[row, :len(tokens)-1], tokens[1:])
                self.assertTrue((y[row, len(tokens)-1:] == -100).all())
        finally:
            dataset.close()

    def test_train_checkpoint_inference_and_exact_resume(self):
        self.run_training("uninterrupted", 4)
        self.run_training("resumed", 2)
        self.run_training("resumed", 4, ["--resume", str(self.root / "resumed/latest.pt")], tiny=False)
        whole = read_checkpoint(self.root / "uninterrupted/latest.pt")
        resumed = read_checkpoint(self.root / "resumed/latest.pt")
        for name, tensor in whole["model_state"].items():
            torch.testing.assert_close(tensor, resumed["model_state"][name], atol=0, rtol=0)
        model, _, _ = load_model(self.root / "resumed/latest.pt", "cpu")
        self.assertEqual(model.generate(torch.tensor([[4, 5]]), 2).shape[0], 1)

    def test_copied_architecture_and_standalone_adapter_roundtrip(self):
        variant = self.root / "variant.py"
        source = (ROOT / "models/transformer.py").read_text()
        # An actual feed-forward change verifies dynamic loading, not just copying.
        variant.write_text(source.replace("F.gelu(self.up_proj(x))", "F.relu(self.up_proj(x))"))
        self.run_training("base", 2, ["--architecture", str(variant)])
        base = self.root / "base/latest.pt"
        original = read_checkpoint(base)["model_state"]
        self.run_training("lora", 2, ["--init-from", str(base), "--lora-rank", "2"], tiny=False)
        full, _, _ = load_model(self.root / "lora/latest.pt", "cpu")
        adapter, _, _ = load_model(base, "cpu", adapter=self.root / "lora/latest.adapter.pt")
        x = torch.tensor([[4, 5, 6]])
        torch.testing.assert_close(full(x), adapter(x), rtol=0, atol=0)
        for name, tensor in full.state_dict().items():
            if not name.endswith((".lora_A", ".lora_B")):
                torch.testing.assert_close(tensor, original[name.replace(".base.", ".")], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            load_model(base, "cpu", architecture=ROOT / "models/transformer.py")

    def test_weighted_answer_selection_exact_resume_and_setting_guard(self):
        settings = ["--answer-weight", "20", "--selection-metric", "answer_macro_accuracy"]
        self.run_training("weighted-whole", 4, settings)
        self.run_training("weighted-resumed", 2, settings)
        resume = ["--resume", str(self.root / "weighted-resumed/latest.pt")]
        with self.assertRaisesRegex(ValueError, "answer_weight"):
            self.run_training("weighted-resumed", 4, resume + settings[2:], tiny=False)
        self.run_training("weighted-resumed", 4, resume + settings, tiny=False)
        whole = read_checkpoint(self.root / "weighted-whole/latest.pt")
        resumed = read_checkpoint(self.root / "weighted-resumed/latest.pt")
        self.assertEqual(whole["run"]["selection_state"], resumed["run"]["selection_state"])
        for name, tensor in whole["model_state"].items():
            torch.testing.assert_close(tensor, resumed["model_state"][name], atol=0, rtol=0)
        best = read_checkpoint(self.root / "weighted-resumed/best.pt")
        self.assertEqual(best["step"], resumed["run"]["selection_state"]["step"])
        metrics = json.loads((self.root / "weighted-resumed/best_validation.json").read_text())
        self.assertIn("by_variant", metrics)
        self.assertEqual(metrics["selection"]["value"], metrics["answer_macro_accuracy"])

    def test_answer_scoring_does_not_feed_answers_to_model(self):
        dataset = EpisodeDataset(self.root / "validation")
        class Spy(torch.nn.Module):
            class Config:
                context_length = 128
            config = Config()
            def forward(self, x):
                return torch.zeros(*x.shape, 128)
            def generate(self, prompts, **kwargs):
                # Every prompt must end at '?' (ID 1), before the answer.
                if not bool((prompts[:, -1] == 1).all()):
                    raise AssertionError("Ground-truth answer leaked into prompt")
                return torch.cat([prompts, torch.full((len(prompts), 1), 2)], dim=1)
        try:
            result = evaluate(Spy(), dataset, "cpu", max_episodes=8)
            self.assertEqual(result["answer_exact_match"], 0)
            self.assertEqual(result["questions"], 8)
        finally:
            dataset.close()

    def test_corrupt_prepared_data_is_rejected(self):
        path = self.root / "train/tokens.bin"
        content = path.read_bytes()
        path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
        with self.assertRaisesRegex(ValueError, "checksum"):
            EpisodeDataset(self.root / "train")

    def test_registered_extension_data_trains_without_plugin_in_trainer(self):
        from micro_transformer.data import generator as g
        for split in ("train", "validation"):
            source = self.root / f"extension-{split}.jsonl"
            options = g.build_parser().parse_args([
                "--episodes", "30", "--output", str(source), "--split", split,
                "--task-module", str(ROOT / "scripts/examples/location_check_task.py"),
                "--profile-file", str(ROOT / "scripts/examples/location_check_profile.json"),
            ])
            g.generate(options)
            prepare(source, self.root / f"extension-{split}")
        dataset = EpisodeDataset(self.root / "extension-train")
        try:
            self.assertEqual(dataset[0][2], "location_check")
            self.assertEqual(dataset.annotations(0)["variant"], "contains")
        finally:
            dataset.close()
        self.run_training("extension-model", 2, [
            "--train-data", str(self.root / "extension-train"),
            "--validation-data", str(self.root / "extension-validation"), "--save-steps", "1",
        ])
        saved = self.root / "extension-model/step-000001.pt"
        self.assertEqual(read_checkpoint(saved)["step"], 1)


if __name__ == "__main__":
    unittest.main()
