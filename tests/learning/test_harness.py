import contextlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "micro-world"))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from micro_transformer.data.generator import Context, derive_answer, state_tracking
    from training.data import EpisodeDataset, TOKENS, encode, prepare
    from training.harness import InferenceHarness, test_cases, test_dataset
    from training.infer import main
    from training.metrics import evaluate
    from training.train import parser, train


@unittest.skipUnless(HAS_TORCH, "Install training/requirements.txt for harness tests")
class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for split in ("train", "validation"):
            source = self.root / f"{split}.jsonl"
            with source.open("w") as stream:
                for index in range(12):
                    episode = state_tracking(Context(random.Random(index), split, "simple"), index)
                    stream.write(json.dumps({**vars(episode), "id": index, "split": split}) + "\n")
            prepare(source, self.root / split)
        args = parser().parse_args([
            "--train-data", str(self.root / "train"), "--validation-data", str(self.root / "validation"),
            "--run-dir", str(self.root / "run"), "--steps", "2", "--batch-size", "4",
            "--eval-episodes", "4", "--eval-every", "2", "--device", "cpu", "--threads", "1",
            "--layers", "1", "--width", "16", "--heads", "2", "--ff-width", "32",
        ])
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
        self.checkpoint = self.root / "run/best.pt"
        self.harness = InferenceHarness(self.checkpoint, device="cpu")
        self.prompt = "ava move cube two to garden. where cube two?"

    def tearDown(self):
        self.temp.cleanup()

    def test_reload_repetition_and_frozen_weights(self):
        before = self.harness.state_sha256()
        first = self.harness.predict(self.prompt)
        self.harness.predict("ben move key one to hall. where key one?")
        second = self.harness.predict(self.prompt)
        reloaded = InferenceHarness(self.checkpoint, device="cpu").predict(self.prompt)
        self.assertEqual(first["token_ids"], second["token_ids"])
        self.assertEqual(first["token_ids"], reloaded["token_ids"])
        self.assertEqual(before, self.harness.state_sha256())
        self.assertFalse(any(p.requires_grad for p in self.harness.model.parameters()))

    def test_reject_invalid_input(self):
        for prompt in ("", "unknownword", self.prompt + " |", "ava " * 129):
            with self.assertRaises(ValueError):
                self.harness.predict(prompt)
        for budget in (0, -1, True):
            with self.assertRaises(ValueError):
                self.harness.predict(self.prompt, max_new_tokens=budget)
        for temperature in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                self.harness.predict(self.prompt, temperature=temperature)

    def test_context_limit_is_reported_without_cropping(self):
        def no_eos(x, max_new_tokens, **kwargs):
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(x.shape[1], 128)
            self.assertEqual(max_new_tokens, 1)
            return torch.cat((x, torch.zeros(1, 1, dtype=torch.long)), dim=1)
        with patch.object(self.harness.model, "generate", side_effect=no_eos):
            result = self.harness.predict("ava " * 128, max_new_tokens=8)
        self.assertEqual(result["stop_reason"], "context_limit")
        self.assertEqual(result["prompt_tokens"], 128)
        self.assertEqual(result["generated_tokens"], 1)

    def test_scored_report_agrees_with_evaluator_and_hides_answers(self):
        original = self.harness.predict
        def watched(prompt, **kwargs):
            self.assertTrue(prompt.endswith("?"))
            return original(prompt, **kwargs)
        with patch.object(self.harness, "predict", side_effect=watched):
            report = test_dataset(self.harness, self.root / "validation", self.root / "report")
        rows = [json.loads(line) for line in (self.root / "report/predictions.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 12)
        self.assertEqual(report["questions"], 12)
        self.assertEqual(report["correct"], sum(row["prediction"] == row["expected"] for row in rows))
        self.assertTrue(report["checks"]["weights_unchanged"])
        self.assertTrue(report["checks"]["repeatable_greedy_output"])
        dataset = EpisodeDataset(self.root / "validation")
        try:
            reference = evaluate(self.harness.model, dataset, "cpu")
            self.assertEqual(report["answer_exact_match"], reference["answer_exact_match"])
            self.assertAlmostEqual(report["loss"], reference["loss"], places=6)
        finally:
            dataset.close()
        with self.assertRaisesRegex(ValueError, "already exists"):
            test_dataset(self.harness, self.root / "validation", self.root / "report")
        with self.assertRaisesRegex(ValueError, "validation or test"):
            test_dataset(self.harness, self.root / "train", self.root / "rejected")
        self.assertFalse((self.root / "rejected").exists())

    def test_interactive_cli_recovers_from_invalid_prompt(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        lines = f"{self.prompt}\ninvalidword\n{self.prompt}\n:quit\n"
        with patch("sys.stdin", io.StringIO(lines)), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            main(["--checkpoint", str(self.checkpoint), "--interactive", "--json", "--device", "cpu"])
        outputs = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0]["prediction"], outputs[1]["prediction"])
        self.assertIn("Invalid prompt", stderr.getvalue())

    def test_single_and_batch_cli(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(["--checkpoint", str(self.checkpoint), "--prompt", self.prompt, "--json", "--device", "cpu"])
        self.assertEqual(json.loads(output.getvalue())["prompt"], self.prompt)
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--checkpoint", str(self.checkpoint), "--data", str(self.root / "validation"),
                  "--report-dir", str(self.root / "cli-report"), "--max-episodes", "3", "--device", "cpu"])
        summary = json.loads((self.root / "cli-report/summary.json").read_text())
        self.assertEqual(summary["questions"], 3)
        self.assertEqual(summary["checkpoint_sha256"], self.harness.identity["checkpoint_sha256"])

    def test_custom_cases_score_expected_values_without_leaking_them(self):
        source = self.root / "cases.jsonl"
        case = {"id": "location", "prompt": self.prompt, "expected": "garden"}
        source.write_text(json.dumps(case) + "\n")
        original = self.harness.predict
        def watched(prompt, **kwargs):
            self.assertEqual(prompt, self.prompt)
            return original(prompt, **kwargs)
        with patch.object(self.harness, "predict", side_effect=watched):
            report = test_cases(self.harness, source, self.root / "cases-report")
        row = json.loads((self.root / "cases-report/predictions.jsonl").read_text())
        self.assertEqual(row["expected"], "garden |")
        self.assertEqual(report["correct"], int(row["prediction"] == row["expected"]))
        self.assertTrue(report["checks"]["weights_unchanged"])
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--checkpoint", str(self.checkpoint), "--cases", str(source),
                  "--report-dir", str(self.root / "cases-cli"), "--device", "cpu"])
        self.assertTrue((self.root / "cases-cli/summary.json").exists())
        source.write_text(json.dumps({"prompt": self.prompt}) + "\n")
        with self.assertRaisesRegex(ValueError, "requires string"):
            test_cases(self.harness, source, self.root / "bad-cases")
        self.assertFalse((self.root / "bad-cases").exists())

    def test_checked_in_case_labels_match_the_reference_interpreter(self):
        for line in (ROOT / "training/examples/trial_cases.jsonl").read_text().splitlines():
            case = json.loads(line)
            episode = [TOKENS[token] for token in encode(case["prompt"] + " " + case["expected"])]
            self.assertEqual(derive_answer(episode), case["expected"].split()[:-1], case["id"])


if __name__ == "__main__":
    unittest.main()
