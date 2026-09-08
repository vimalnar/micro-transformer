import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from models.transformer import ModelConfig, Transformer


@unittest.skipUnless(HAS_TORCH, "Install training/requirements.txt for model tests")
class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.config = ModelConfig(n_layers=2, d_model=32, n_heads=4, d_ff=64, context_length=32)
        self.model = Transformer(self.config).eval()

    def test_configurable_size_and_output_vocabulary(self):
        x = torch.tensor([[1, 2, 3]])
        self.assertEqual(self.model(x).shape, (1, 3, 128))
        baseline = Transformer(ModelConfig())
        self.assertEqual(sum(p.numel() for p in baseline.parameters()), 1024512)
        self.assertLess(sum(p.numel() for p in self.model.parameters()), 1024512)
        with self.assertRaises(ValueError):
            ModelConfig(d_model=31, n_heads=4)

    def test_future_tokens_cannot_change_prefix_logits(self):
        first = torch.tensor([[4, 5, 6, 7, 8]])
        second = torch.tensor([[4, 5, 6, 90, 91]])
        torch.testing.assert_close(self.model(first)[:, :3], self.model(second)[:, :3])

    def test_right_padding_does_not_change_valid_logits(self):
        x = torch.tensor([[4, 5, 6]])
        padded = torch.tensor([[4, 5, 6, 128, 128]])
        torch.testing.assert_close(self.model(x), self.model(padded)[:, :3])

    def test_forward_backward_and_padding_embedding(self):
        self.model.train()
        loss = self.model(torch.tensor([[4, 5, 128]])).square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
        self.assertEqual(self.model.token_embedding.weight.grad[128].abs().sum(), 0)

    def test_eval_is_repeatable_and_generation_does_not_change_weights(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        x = torch.tensor([[4, 5, 6]])
        a = self.model.generate(x, 3)
        b = self.model.generate(x, 3)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        for name, tensor in self.model.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            self.model(torch.ones(1, 33, dtype=torch.long))

    def test_lora_initial_equivalence_freezing_and_merge(self):
        x = torch.tensor([[4, 5, 6]])
        before_logits = self.model(x).detach().clone()
        frozen = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        count = self.model.enable_lora(rank=2, alpha=4.0)
        self.assertEqual(count, 512)
        torch.testing.assert_close(before_logits, self.model(x))
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=0.01)
        self.model(x).square().mean().backward(); opt.step()
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                original = name.replace(".base.", ".")
                torch.testing.assert_close(parameter, frozen[original], rtol=0, atol=0)
        self.assertTrue(any(t.abs().sum() > 0 for n, t in self.model.adapter_state_dict().items() if n.endswith("lora_B")))
        adapted = self.model(x).detach().clone()
        self.model.merge_lora()
        torch.testing.assert_close(adapted, self.model(x), rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
