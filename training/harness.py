"""Reusable frozen-weight inference and held-out, free-generation testing.

The model sees prompt tokens only. Expected answers belong to the scorer, never
to predict(). No interpreter or world state is used to produce model outputs.
"""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import torch

from training.data import EOS_ID, EpisodeDataset, decode, encode, file_hash
from training.metrics import evaluate
from training.runtime import environment, load_model, seed_everything, select_device, synchronize, write_json


class InferenceHarness:
    """Load once; each call is an independent episode with no conversational state."""

    def __init__(self, checkpoint, device="auto", architecture=None, adapter=None, seed=42):
        seed_everything(seed)
        self.device = select_device(device)
        self.model, payload, source = load_model(checkpoint, self.device, architecture, adapter)
        self.model.requires_grad_(False)
        self.identity = {
            "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": file_hash(checkpoint),
            "checkpoint_training_step": payload["step"],
            "architecture": str(source), "architecture_sha256": file_hash(source),
            "seed": seed, "environment": environment(self.device),
            "adapter": None,
        }
        if adapter:
            values = torch.load(adapter, map_location="cpu", weights_only=True)
            self.identity["adapter"] = {"path": str(Path(adapter).resolve()),
                                        "sha256": file_hash(adapter), "training_step": values["step"]}

    def state_sha256(self):
        """Fingerprint all weights AND buffers to detect unintended inference updates."""
        digest = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @torch.inference_mode()
    def predict(self, prompt, max_new_tokens=8, temperature=0.0):
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        ids = encode(prompt)
        if EOS_ID in ids:
            raise ValueError("Use one unfinished episode; omit the | boundary from the prompt")
        if len(ids) > self.model.config.context_length:
            raise ValueError("Prompt exceeds the checkpoint's context length")
        # The final available input position can still predict one output token.
        capacity = self.model.config.context_length - len(ids) + 1
        budget = min(max_new_tokens, capacity)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        self.model.eval()
        synchronize(self.device)
        start = time.perf_counter()
        output = self.model.generate(x, max_new_tokens=budget, eos_id=EOS_ID, temperature=temperature)
        completion = output[0, len(ids):].cpu().tolist()
        synchronize(self.device)
        elapsed = time.perf_counter() - start
        terminated = EOS_ID in completion
        if terminated:
            completion = completion[:completion.index(EOS_ID) + 1]
        reason = "episode_end" if terminated else ("context_limit" if budget == capacity else "token_limit")
        return {"prompt": decode(ids), "prediction": decode(completion),
                "answer": decode(completion[:-1] if terminated else completion),
                "token_ids": completion, "prompt_tokens": len(ids), "generated_tokens": len(completion),
                "stop_reason": reason, "elapsed_seconds": elapsed}


def test_dataset(harness, directory, report_dir, max_episodes=None, max_new_tokens=8):
    """Score every question; save every success/failure, never overwrite a report.

    Statement-only episodes contribute token metrics, not question accuracy.
    Incorrect answers are legitimate test results, not execution failures.
    """
    destination = Path(report_dir).resolve()
    if destination.exists():
        raise ValueError("Report directory already exists; choose a new path")
    dataset = EpisodeDataset(directory)
    try:
        if dataset.metadata["split"] not in ("validation", "test"):
            raise ValueError("Harness testing requires a validation or test split")
        if max_episodes is not None and max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        count = len(dataset) if max_episodes is None else min(len(dataset), max_episodes)
        if dataset.metadata["max_episode_tokens"] - 1 > harness.model.config.context_length:
            raise ValueError("Dataset episodes exceed the checkpoint's context length")
        before = harness.state_sha256()
        start = time.perf_counter()
        token_metrics = evaluate(harness.model, dataset, harness.device, max_episodes=count, answers=False)
        tasks = defaultdict(lambda: {"questions": 0, "correct": 0, "terminated": 0})
        variants = defaultdict(lambda: {"questions": 0, "correct": 0})
        first = None
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".inference-test-", dir=destination.parent) as temporary:
            stage = Path(temporary) / "report"
            stage.mkdir()
            with (stage / "predictions.jsonl").open("w") as stream:
                for index in range(count):
                    tokens, answer_start, task = dataset[index]
                    if not answer_start:
                        continue
                    prompt = decode(tokens[:answer_start].tolist())
                    result = harness.predict(prompt, max_new_tokens=max_new_tokens)
                    # Scoring reads the expected answer only after model generation.
                    expected = tokens[answer_start:].tolist()
                    correct = result["token_ids"] == expected
                    tasks[task]["questions"] += 1
                    tasks[task]["correct"] += int(correct)
                    tasks[task]["terminated"] += int(result["stop_reason"] == "episode_end")
                    annotation = dataset.annotations(index)
                    variant_key = f"{task}/{annotation['variant']}"
                    variants[variant_key]["questions"] += 1
                    variants[variant_key]["correct"] += int(correct)
                    row = {"episode_index": index, "task": task, **annotation, **result,
                           "expected": decode(expected), "correct": correct}
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    if first is None:
                        first = result
            repeatable = None if first is None else harness.predict(
                first["prompt"], max_new_tokens=max_new_tokens)["token_ids"] == first["token_ids"]
            after = harness.state_sha256()
            if before != after:
                raise RuntimeError("Model weights or buffers changed during frozen inference")
            if repeatable is False:
                raise RuntimeError("Greedy output changed when the same independent prompt was repeated")
            questions = sum(row["questions"] for row in tasks.values())
            correct = sum(row["correct"] for row in tasks.values())
            report = {"format": "micro-transformer-inference-test-v1", **harness.identity,
                      "dataset": str(dataset.directory), "split": dataset.metadata["split"],
                      "dataset_metadata_sha256": file_hash(dataset.directory / "metadata.json"),
                      "dataset_source_sha256": dataset.metadata["source_sha256"],
                      "decoding": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
                      **token_metrics, "questions": questions, "correct": correct,
                      "statement_only_episodes": count - questions,
                      "answer_exact_match": correct / questions if questions else None,
                      "by_task": {task: {**row, "accuracy": row["correct"] / row["questions"]}
                                  for task, row in sorted(tasks.items())},
                      "by_variant": {key: {**row, "accuracy": row["correct"] / row["questions"]}
                                     for key, row in sorted(variants.items())},
                      "checks": {"weights_unchanged": before == after, "repeatable_greedy_output": repeatable,
                                 "trainable_parameters": sum(p.numel() for p in harness.model.parameters() if p.requires_grad),
                                 "state_sha256_before": before, "state_sha256_after": after},
                      "elapsed_seconds": time.perf_counter() - start,
                      "predictions_sha256": file_hash(stage / "predictions.jsonl")}
            write_json(stage / "summary.json", report)
            stage.rename(destination)
        return report
    finally:
        dataset.close()


def test_cases(harness, source, report_dir, max_new_tokens=8):
    """Run small, human-authored JSONL regression cases separately from corpus tests.

    Each line has a prompt and expected continuation, plus optional id/task.
    Expected values are external test labels; this function does not derive them.
    """
    source, destination = Path(source).resolve(), Path(report_dir).resolve()
    if destination.exists():
        raise ValueError("Report directory already exists; choose a new path")
    before = harness.state_sha256()
    source_digest = file_hash(source)
    count = correct = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".inference-cases-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        with source.open() as inputs, (stage / "predictions.jsonl").open("w") as outputs:
            for number, line in enumerate(inputs, 1):
                case = json.loads(line)
                if not isinstance(case, dict) or not all(isinstance(case.get(key), str) for key in ("prompt", "expected")):
                    raise ValueError(f"Case {number} requires string prompt and expected fields")
                expected = encode(case["expected"])
                if expected[-1] != EOS_ID:
                    expected.append(EOS_ID)
                if expected.count(EOS_ID) != 1:
                    raise ValueError(f"Case {number} must have one final | boundary")
                prediction = harness.predict(case["prompt"], max_new_tokens=max_new_tokens)
                matched = prediction["token_ids"] == expected
                count += 1
                correct += int(matched)
                row = {"id": case.get("id", number), "task": case.get("task", "manual"),
                       **prediction, "expected": decode(expected), "correct": matched}
                outputs.write(json.dumps(row, allow_nan=False) + "\n")
        if not count:
            raise ValueError("Case file is empty")
        after = harness.state_sha256()
        if after != before:
            raise RuntimeError("Model weights or buffers changed during frozen inference")
        if file_hash(source) != source_digest:
            raise ValueError("Case source changed during evaluation")
        report = {"format": "micro-transformer-inference-cases-v1", **harness.identity,
                  "cases": str(source), "cases_sha256": source_digest,
                  "decoding": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
                  "questions": count, "correct": correct, "answer_exact_match": correct / count,
                  "checks": {"weights_unchanged": before == after,
                             "state_sha256_before": before, "state_sha256_after": after},
                  "predictions_sha256": file_hash(stage / "predictions.jsonl")}
        write_json(stage / "summary.json", report)
        stage.rename(destination)
    return report
