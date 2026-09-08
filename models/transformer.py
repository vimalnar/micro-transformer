"""Vanilla Micro-Transformer causal baseline. Copy this file to start an experiment.

Only PyTorch and the standard library are required. The training contract is:
ModelConfig(**json_config), Transformer(config), and model(input_ids) -> logits
of shape [batch, time, vocabulary]. Inputs are independent, right-padded episodes.
Forward and generation never update model parameters.
"""

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 128  # Output classes; padding is a separate input-only ID.
    context_length: int = 128
    n_layers: int = 5
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self):
        for name in ("vocab_size", "context_length", "n_layers", "d_model", "n_heads", "d_ff"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def pad_id(self):
        return self.vocab_size


class CausalSelfAttention(nn.Module):
    """The main extension point for attention experiments."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

    def forward(self, x):
        batch, length, width = x.shape

        def heads(projection):
            return projection(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        # Right padding is always after valid tokens. Causality prevents it from
        # affecting valid outputs; the training loss separately ignores padding.
        attended = F.scaled_dot_product_attention(
            heads(self.q_proj), heads(self.k_proj), heads(self.v_proj),
            is_causal=True, dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(attended.transpose(1, 2).contiguous().view(batch, length, width))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ff_norm = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        return x + self.dropout(self.ff(self.ff_norm(x)))


class LoRALinear(nn.Module):
    """Optional W x + (alpha/r) B A x; base weights are frozen."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.normal_(self.lora_A, std=0.02)
        self.base.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size + 1, config.d_model, padding_idx=config.pad_id)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lora_config = None
        self.apply(self._initialize)
        # Residual projections use a smaller initialization as depth increases.
        for block in self.blocks:
            nn.init.normal_(block.attention.out_proj.weight, std=0.02 / (2 * config.n_layers) ** 0.5)
            nn.init.normal_(block.ff.down_proj.weight, std=0.02 / (2 * config.n_layers) ** 0.5)
        with torch.no_grad():
            self.token_embedding.weight[config.pad_id].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= self.config.context_length:
            raise ValueError("input_ids must be [batch, time] within the configured context length")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.embedding_dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        # Tied output weights, with no padding output class.
        return F.linear(self.final_norm(x), self.token_embedding.weight[:self.config.vocab_size])

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=8, eos_id=2, temperature=0.0):
        """Generate from equal-length, unpadded prompts. Never silently crop context."""
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError("max_new_tokens and temperature must be nonnegative")
        was_training = self.training
        self.eval()
        result = input_ids.clone()
        finished = torch.zeros(result.shape[0], dtype=torch.bool, device=result.device)
        try:
            for _ in range(max_new_tokens):
                if result.shape[1] > self.config.context_length:
                    raise ValueError("generation exceeded context length; increase model context or shorten the prompt")
                logits = self(result)[:, -1]
                token = (logits.argmax(dim=-1) if temperature == 0 else
                         torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).squeeze(1))
                token = torch.where(finished, eos_id, token)
                result = torch.cat((result, token[:, None]), dim=1)
                finished |= token == eos_id
                if bool(finished.all()):
                    break
            return result
        finally:
            self.train(was_training)

    def enable_lora(self, rank=4, alpha=8.0, targets: Iterable[str] = ("q_proj", "v_proj")):
        if self.lora_config is not None:
            raise ValueError("LoRA is already enabled")
        targets = tuple(targets)
        if not isinstance(rank, int) or rank < 1 or alpha <= 0 or not targets:
            raise ValueError("LoRA needs a positive rank/alpha and at least one target")
        modules = [(name, module) for name, module in self.named_modules()
                   if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets]
        missing = set(targets) - {name.rsplit(".", 1)[-1] for name, _ in modules}
        if missing:
            raise ValueError(f"LoRA targets not found: {sorted(missing)}")
        self.requires_grad_(False)
        for name, module in modules:
            parent_path, leaf = name.rsplit(".", 1)
            setattr(self.get_submodule(parent_path), leaf, LoRALinear(module, rank, alpha))
        self.lora_config = {"rank": rank, "alpha": alpha, "targets": list(targets)}
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def adapter_state_dict(self):
        if self.lora_config is None:
            raise ValueError("LoRA is not enabled")
        return {name: tensor.detach().cpu().clone() for name, tensor in self.state_dict().items()
                if name.endswith((".lora_A", ".lora_B"))}

    @torch.no_grad()
    def merge_lora(self):
        """Merge a trained adapter into ordinary linear layers for inference/export."""
        for name, module in list(self.named_modules()):
            if isinstance(module, LoRALinear):
                module.base.weight.add_((module.lora_B @ module.lora_A) * module.scale)
                parent_path, leaf = name.rsplit(".", 1)
                setattr(self.get_submodule(parent_path), leaf, module.base)
        self.lora_config = None
