"""Architecture loading, devices, and portable checkpoints."""

from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile

import torch

from training.data import TOKENS, file_hash

CHECKPOINT_FORMAT = "micro-transformer-checkpoint-v1"
ADAPTER_FORMAT = "micro-transformer-lora-v1"


def select_device(requested="auto"):
    if requested == "auto":
        requested = ("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    return torch.device(requested)


def seed_everything(seed, threads=2):
    if threads < 1:
        raise ValueError("CPU threads must be positive")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "platform": platform.platform(), "machine": platform.machine(), "device": str(device)}


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def rng_state(device):
    state = {"python": random.getstate(), "cpu": torch.get_rng_state()}
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["cpu"])
    if device.type == "mps":
        torch.mps.set_rng_state(state["mps"])
    elif device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def load_architecture(path):
    path = Path(path).resolve()
    if path.suffix != ".py" or not path.is_file():
        raise ValueError(f"Architecture must be a Python file: {path}")
    name = "micro_transformer_arch_" + file_hash(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Dataclass annotations need a registered module.
    spec.loader.exec_module(module)
    if not hasattr(module, "ModelConfig") or not hasattr(module, "Transformer"):
        raise ValueError("Architecture must export ModelConfig and Transformer")
    return module


def read_checkpoint(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("tokens_by_id") != TOKENS:
        raise ValueError("Unsupported checkpoint or mismatched token vocabulary")
    return payload


def load_model(checkpoint, device, architecture=None, adapter=None):
    path = Path(checkpoint).resolve()
    payload = read_checkpoint(path)
    source = Path(architecture).resolve() if architecture else path.parent / payload["architecture_file"]
    if file_hash(source) != payload["architecture_sha256"]:
        raise ValueError("Architecture changed since this checkpoint was saved; use its saved architecture.py")
    module = load_architecture(source)
    model = module.Transformer(module.ModelConfig(**payload["model_config"]))
    if payload["lora"]:
        model.enable_lora(**payload["lora"])
    model.load_state_dict(payload["model_state"], strict=True)
    if adapter:
        if payload["lora"]:
            raise ValueError("Load a standalone adapter onto its original base checkpoint")
        values = torch.load(adapter, map_location="cpu", weights_only=True)
        if (values.get("format") != ADAPTER_FORMAT or values["base_sha256"] != file_hash(path)
                or values["architecture_sha256"] != payload["architecture_sha256"]
                or values["tokens_by_id"] != TOKENS):
            raise ValueError("Adapter does not match this base checkpoint/architecture/vocabulary")
        model.enable_lora(**values["lora"])
        expected = model.adapter_state_dict()
        if set(expected) != set(values["adapter_state"]):
            raise ValueError("Adapter parameter names do not match the model")
        for name, tensor in values["adapter_state"].items():
            if tensor.shape != expected[name].shape:
                raise ValueError(f"Adapter shape mismatch: {name}")
        model.load_state_dict(values["adapter_state"], strict=False)
    model.to(device).eval()
    return model, payload, source


def atomic_torch_save(payload, path):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def write_checkpoint(path, model, optimizer, step, sampler, best_loss, run, device):
    payload = {
        "format": CHECKPOINT_FORMAT, "architecture_file": "architecture.py",
        "architecture_sha256": run["architecture_sha256"], "model_config": asdict(model.config),
        "tokens_by_id": TOKENS, "lora": getattr(model, "lora_config", None),
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(), "step": step, "sampler": sampler.state_dict(),
        "rng": rng_state(device), "best_validation_loss": best_loss, "run": run,
    }
    atomic_torch_save(payload, path)
    if payload["lora"]:
        adapter = {"format": ADAPTER_FORMAT, "architecture_sha256": run["architecture_sha256"],
                   "tokens_by_id": TOKENS, "lora": payload["lora"], "base_sha256": run["base_sha256"],
                   "base_checkpoint": run["base_checkpoint"], "adapter_state": model.adapter_state_dict(),
                   "step": step}
        atomic_torch_save(adapter, Path(path).with_suffix(".adapter.pt"))


def write_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
