import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from training.data import collate
    from training.objectives import answer_mask, improves_selection, language_loss, selection_candidate


@unittest.skipUnless(HAS_TORCH, "Install training/requirements.txt for objective tests")
class ObjectiveTests(unittest.TestCase):
    def records(self):
        # 1='?', 2='|'; include one-token, two-token and statement-only answers.
        return [(torch.tensor([5, 1, 7, 2]), 2, "one"),
                (torch.tensor([6, 8, 1, 9, 10, 2]), 3, "two"),
                (torch.tensor([5, 6, 2]), 0, "statement")]

    def test_answer_mask_shift_eos_padding_and_statement(self):
        records = self.records()
        _, y = collate(records, 128, "cpu")
        self.assertEqual(answer_mask(records, y).tolist(),
                         [[False, True, True, False, False],
                          [False, False, True, True, True],
                          [False, False, False, False, False]])

    def test_weighted_loss_matches_manual_and_default_matches_original(self):
        records = self.records()
        _, y = collate(records, 128, "cpu")
        logits = torch.arange(3 * 5 * 128, dtype=torch.float32).reshape(3, 5, 128) / 80
        losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100, reduction="none")
        mask = answer_mask(records, y)
        expected = (losses[y != -100].sum() + 19 * losses[mask].sum()) / (y.ne(-100).sum() + 19 * mask.sum())
        torch.testing.assert_close(language_loss(logits, y, records, 20), expected)
        original = torch.nn.functional.cross_entropy(logits.reshape(-1, 128), y.reshape(-1), ignore_index=-100)
        torch.testing.assert_close(language_loss(logits, y, records), original, rtol=0, atol=0)
        changed = logits.clone()
        changed[y == -100] = -100
        torch.testing.assert_close(language_loss(changed, y, records, 20), expected)

    def test_invalid_weight_is_rejected(self):
        records = self.records()
        _, y = collate(records, 128, "cpu")
        logits = torch.zeros(*y.shape, 128)
        for weight in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                language_loss(logits, y, records, weight)

    def test_selection_prioritizes_qa_and_uses_loss_only_to_break_ties(self):
        def candidate(accuracy, loss, step=1):
            return selection_candidate({"answer_macro_accuracy": accuracy, "loss": loss}, "answer_macro_accuracy", step)
        a, b = candidate(.9, 1.2), candidate(.95, 1.5)
        self.assertTrue(improves_selection(b, a))
        self.assertFalse(improves_selection(a, b))
        self.assertTrue(improves_selection(candidate(.95, 1.4), b))
        self.assertFalse(improves_selection(candidate(.95, 1.5, 2), b))
        with self.assertRaises(ValueError):
            candidate(None, 1.0)


if __name__ == "__main__":
    unittest.main()
