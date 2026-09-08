"""Train a standalone Micro-Transformer architecture, or fine-tune a frozen base with LoRA."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import sys
import time

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.data import EpisodeDataset, TOKENS, collate, file_hash
from training.metrics import evaluate
from training.objectives import improves_selection, language_loss, selection_candidate
from training.runtime import (environment, load_architecture, load_model, restore_rng,
                              seed_everything, select_device, synchronize, write_checkpoint, write_json)


class EpisodeSampler:
    """Deterministic shuffled passes, reconstructed from epoch and offset on resume."""

    def __init__(self, count, seed, state=None):
        self.count, self.seed = count, seed
        self.epoch, self.offset = (state["epoch"], state["offset"]) if state else (0, 0)
        self._shuffle()

    def _shuffle(self):
        self.order = torch.randperm(self.count, generator=torch.Generator().manual_seed(self.seed + self.epoch))

    def next(self, batch_size):
        if self.offset == self.count:
            self.epoch += 1
            self.offset = 0
            self._shuffle()
        stop = min(self.offset + batch_size, self.count)
        batch = self.order[self.offset:stop].tolist()
        self.offset = stop
        return batch

    def state_dict(self):
        return {"epoch": self.epoch, "offset": self.offset}


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--architecture", type=Path)
    command.add_argument("--model-config", type=Path, help="JSON object passed to the architecture's ModelConfig")
    command.add_argument("--train-data", type=Path, required=True)
    command.add_argument("--validation-data", type=Path, required=True)
    command.add_argument("--run-dir", type=Path, required=True)
    mode = command.add_mutually_exclusive_group()
    mode.add_argument("--resume", type=Path)
    mode.add_argument("--init-from", type=Path, help="Start a new full-fine-tuning or LoRA run from a checkpoint")
    duration = command.add_mutually_exclusive_group()
    duration.add_argument("--steps", type=int, help="Total optimizer-step target, including any resumed steps")
    duration.add_argument("--epochs", type=int, help="Total shuffled passes; default 1")
    for flag in ("layers", "width", "heads", "ff-width", "context-length"):
        command.add_argument("--" + flag, type=int)
    command.add_argument("--dropout", type=float)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--learning-rate", type=float, default=0.0003)
    command.add_argument("--weight-decay", type=float, default=0.01)
    command.add_argument("--grad-clip", type=float, default=1.0)
    command.add_argument("--answer-weight", type=float, default=1.0,
                         help="Answer and episode-end target weight; ordinary tokens retain weight 1")
    command.add_argument("--selection-metric", choices=("loss", "answer_exact_match", "answer_macro_accuracy"), default="loss",
                         help="best.pt criterion: token loss, QA accuracy, or equal-weight mean QA variant accuracy")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    command.add_argument("--threads", type=int, default=2)
    command.add_argument("--eval-every", type=int, default=200)
    command.add_argument("--eval-episodes", type=int, default=512)
    command.add_argument("--log-every", type=int, default=20)
    command.add_argument("--save-steps", type=int, nargs="*", default=[], help="Also retain named checkpoints at these optimizer steps")
    command.add_argument("--lora-rank", type=int, default=0)
    command.add_argument("--lora-alpha", type=float, default=8.0)
    command.add_argument("--lora-targets", nargs="+", default=["q_proj", "v_proj"])
    return command


def train(args):
    for name in ("batch_size", "eval_every", "eval_episodes", "log_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if any(step < 1 for step in args.save_steps):
        raise ValueError("Saved checkpoint steps must be positive")
    if args.learning_rate <= 0 or args.grad_clip <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer settings")
    if not math.isfinite(args.answer_weight) or args.answer_weight < 1:
        raise ValueError("answer_weight must be finite and at least 1")
    select_answers = args.selection_metric != "loss"
    if args.lora_rank < 0 or (args.lora_rank and not args.init_from):
        raise ValueError("New LoRA runs require --init-from and a positive --lora-rank")
    device = select_device(args.device)
    seed_everything(args.seed, args.threads)
    train_data, validation = EpisodeDataset(args.train_data), EpisodeDataset(args.validation_data)
    try:
        if train_data.metadata["split"] != "train" or validation.metadata["split"] != "validation":
            raise ValueError("Training requires train and validation splits; final test data is never used for selection")
        if set(train_data.metadata["entities"]) & set(validation.metadata["entities"]):
            raise ValueError("Training and validation contain overlapping held-out entity identities")
        target = args.steps if args.steps is not None else (args.epochs or 1) * math.ceil(len(train_data) / args.batch_size)
        if target < 1 or args.epochs is not None and args.epochs < 1:
            raise ValueError("Training duration must be positive")
        mapping = {"layers": "n_layers", "width": "d_model", "heads": "n_heads", "ff_width": "d_ff",
                   "context_length": "context_length", "dropout": "dropout"}
        overrides = {key: getattr(args, flag) for flag, key in mapping.items() if getattr(args, flag) is not None}
        initial = None
        if args.resume or args.init_from:
            if overrides or args.model_config:
                raise ValueError("Checkpoint initialization uses its saved model configuration; omit architecture-size overrides")
            model, initial, source = load_model(args.resume or args.init_from, device, args.architecture)
            if args.init_from and initial["lora"]:
                raise ValueError("Use --resume for an existing LoRA run, or initialize from a base checkpoint")
        else:
            source = (args.architecture or Path(__file__).resolve().parents[1] / "models" / "transformer.py").resolve()
            module = load_architecture(source)
            config = json.loads(args.model_config.read_text()) if args.model_config else {}
            model = module.Transformer(module.ModelConfig(**{**config, **overrides})).to(device)
        if model.config.vocab_size != len(TOKENS):
            raise ValueError("This dataset requires exactly 128 language output classes")
        for dataset in (train_data, validation):
            if dataset.metadata["max_episode_tokens"] - 1 > model.config.context_length:
                raise ValueError("Dataset episodes exceed model context; increase context length")
        if args.lora_rank:
            model.enable_lora(args.lora_rank, args.lora_alpha, args.lora_targets)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
        run = {"architecture_sha256": file_hash(source), "model_config": asdict(model.config),
               "train_metadata_sha256": file_hash(train_data.directory / "metadata.json"),
               "validation_metadata_sha256": file_hash(validation.directory / "metadata.json"),
               "seed": args.seed, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
               "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
               "answer_weight": args.answer_weight, "selection_metric": args.selection_metric,
               "training_source_sha256": {name: file_hash(Path(__file__).parent / name)
                                          for name in ("train.py", "metrics.py", "objectives.py", "data.py", "runtime.py")},
               "eval_episodes": args.eval_episodes, "environment": environment(device),
               "parameters": sum(p.numel() for p in model.parameters()),
               "trainable_parameters": sum(p.numel() for p in parameters),
               "base_checkpoint": str(args.init_from.resolve()) if args.init_from else None,
               "base_sha256": file_hash(args.init_from) if args.init_from else None}
        step, best_loss, sampler_state = 0, float("inf"), None
        selected = None
        if args.resume:
            for key in ("architecture_sha256", "train_metadata_sha256", "validation_metadata_sha256", "seed",
                        "batch_size", "learning_rate", "weight_decay", "grad_clip", "eval_episodes", "environment"):
                if run[key] != initial["run"][key]:
                    raise ValueError(f"Resume setting changed: {key}; use --init-from for a new run")
            for key, default in (("answer_weight", 1.0), ("selection_metric", "loss")):
                if run[key] != initial["run"].get(key, default):
                    raise ValueError(f"Resume setting changed: {key}; use --init-from for a new run")
            if ("training_source_sha256" in initial["run"] and
                    run["training_source_sha256"] != initial["run"]["training_source_sha256"]):
                raise ValueError("Resume training source changed; use --init-from for a new run")
            run["base_checkpoint"], run["base_sha256"] = initial["run"]["base_checkpoint"], initial["run"]["base_sha256"]
            optimizer.load_state_dict(initial["optimizer_state"])
            step, best_loss, sampler_state = initial["step"], initial["best_validation_loss"], initial["sampler"]
            selected = initial["run"].get("selection_state")
            if selected is None:  # Legacy checkpoints selected by minimum token loss.
                selected = {"metric": "loss", "value": best_loss, "step": None, "token_loss": best_loss}
            restore_rng(initial["rng"], device)
        if target <= step:
            raise ValueError(f"Target steps {target} must exceed completed steps {step}")
        directory = args.run_dir.resolve()
        if directory.exists() and not args.resume:
            raise ValueError(f"Run directory exists; choose a new directory or use --resume: {directory}")
        if args.resume and args.resume.resolve() != directory / "latest.pt":
            raise ValueError("Resume from this run directory's latest.pt; use --init-from to branch")
        directory.mkdir(parents=True, exist_ok=True)
        if source != directory / "architecture.py":
            shutil.copyfile(source, directory / "architecture.py")
        run["requested_steps"] = target
        write_json(directory / "run.json", run)
        sampler = EpisodeSampler(len(train_data), args.seed, sampler_state)
        print(json.dumps({"device": str(device), "parameters": run["parameters"],
                          "trainable_parameters": run["trainable_parameters"], "steps": target}), flush=True)
        initial_metrics = evaluate(model, validation, device, args.batch_size, args.eval_episodes, answers=select_answers)
        if select_answers and not initial_metrics["questions"]:
            raise ValueError("Answer-based selection requires validation questions")
        if not args.resume:
            write_json(directory / "initial_validation.json", initial_metrics)
        model.train()
        synchronize(device)
        start = time.perf_counter()
        initial_step = step
        processed_tokens = 0
        with (directory / "metrics.jsonl").open("a") as log:
            while step < target:
                records = [train_data[index] for index in sampler.next(args.batch_size)]
                x, y = collate(records, model.config.context_length, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = language_loss(logits, y, records, args.answer_weight)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Training loss is not finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                processed_tokens += sum(len(tokens) - 1 for tokens, _, _ in records)
                if step % args.log_every == 0 or step == target:
                    synchronize(device)
                    elapsed = time.perf_counter() - start
                    row = {"step": step, "train_loss": loss.item(), "target_tokens_processed_this_invocation": processed_tokens,
                           "elapsed_seconds": elapsed, "tokens_per_second": processed_tokens / elapsed}
                    log.write(json.dumps(row) + "\n"); log.flush()
                    print(json.dumps(row), flush=True)
                if step % args.eval_every == 0 or step == target or step in args.save_steps:
                    metrics = evaluate(model, validation, device, args.batch_size, args.eval_episodes, answers=select_answers)
                    row = {"step": step, "validation": metrics}
                    log.write(json.dumps(row) + "\n"); log.flush()
                    print(json.dumps(row), flush=True)
                    candidate = selection_candidate(metrics, args.selection_metric, step)
                    improved = improves_selection(candidate, selected)
                    if improved:
                        selected = candidate
                    run["selection_state"] = selected
                    best_loss = min(best_loss, metrics["loss"])
                    write_checkpoint(directory / "latest.pt", model, optimizer, step, sampler, best_loss, run, device)
                    if improved:
                        write_json(directory / "best_validation.json", {"selection": selected, **metrics})
                        shutil.copyfile(directory / "latest.pt", directory / "best.pt")
                        if getattr(model, "lora_config", None):
                            shutil.copyfile(directory / "latest.adapter.pt", directory / "best.adapter.pt")
                    if step in args.save_steps:
                        shutil.copyfile(directory / "latest.pt", directory / f"step-{step:06d}.pt")
                        if getattr(model, "lora_config", None):
                            shutil.copyfile(directory / "latest.adapter.pt", directory / f"step-{step:06d}.adapter.pt")
        final_metrics = evaluate(model, validation, device, args.batch_size, args.eval_episodes, answers=True)
        write_json(directory / "run.json", run)
        write_json(directory / "final_validation.json", {"checkpoint": "latest.pt", "step": step, **final_metrics})
        print(json.dumps({"completed_steps": step, "steps_this_invocation": step - initial_step,
                          "final_validation": final_metrics, "checkpoint": str(directory / "latest.pt")}), flush=True)
        return final_metrics
    finally:
        train_data.close()
        validation.close()


if __name__ == "__main__":
    train(parser().parse_args())
