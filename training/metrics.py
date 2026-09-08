"""Teacher-forced token loss and separate free-generation answer accuracy."""

from collections import defaultdict
import math

import torch
from torch.nn import functional as F

from training.data import EOS_ID, collate
from training.objectives import answer_mask


@torch.inference_mode()
def evaluate(model, dataset, device, batch_size=32, max_episodes=None, answers=True):
    count = len(dataset) if max_episodes is None else min(len(dataset), max_episodes)
    if count < 1 or batch_size < 1:
        raise ValueError("Evaluation needs at least one episode and a positive batch size")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = correct_tokens = 0
    groups = defaultdict(list)
    tasks = defaultdict(lambda: {"correct": 0, "questions": 0})
    variants = defaultdict(lambda: {"correct": 0, "questions": 0})
    answer_loss = 0.0
    answer_tokens = 0

    def score_answers(batch, prompt_length):
        prompts = torch.stack([record[0][:prompt_length] for record, _ in batch]).to(device)
        generated = model.generate(prompts, max_new_tokens=min(8, model.config.context_length - prompt_length + 1), eos_id=EOS_ID)
        completions = generated[:, prompt_length:].cpu().tolist()
        for completion, ((tokens, answer_start, task), variant) in zip(completions, batch):
            if EOS_ID in completion:
                completion = completion[:completion.index(EOS_ID) + 1]
            tasks[task]["questions"] += 1
            correct = completion == tokens[answer_start:].tolist()
            tasks[task]["correct"] += correct
            variants[f"{task}/{variant}"]["questions"] += 1
            variants[f"{task}/{variant}"]["correct"] += correct

    try:
        for start in range(0, count, batch_size):
            records = [dataset[index] for index in range(start, min(start + batch_size, count))]
            x, y = collate(records, model.config.context_length, device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="sum")
            total_loss += float(loss.item())
            total_tokens += int((y != -100).sum().item())
            correct_tokens += int(((logits.argmax(-1) == y) & (y != -100)).sum().item())
            if answers:
                mask = answer_mask(records, y)
                token_losses = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100, reduction="none")
                answer_loss += float((token_losses * mask).sum().item())
                answer_tokens += int(mask.sum().item())
                for index, record in enumerate(records, start):
                    if record[1]:
                        annotation = dataset.annotations(index) if hasattr(dataset, "annotations") else {"variant": "legacy"}
                        groups[record[1]].append((record, annotation["variant"]))
                        if len(groups[record[1]]) == batch_size:
                            score_answers(groups[record[1]], record[1])
                            groups[record[1]].clear()
        # Group equal-length prompts so generation needs neither padding nor
        # ground-truth answers. Expected answers are only used AFTER generation.
        for prompt_length, records in groups.items():
            if records:
                score_answers(records, prompt_length)
        mean_loss = total_loss / total_tokens
        result = {"episodes": count, "target_tokens": total_tokens, "loss": mean_loss,
                  "perplexity": math.exp(min(mean_loss, 80)), "token_accuracy": correct_tokens / total_tokens}
        if answers:
            questions = sum(values["questions"] for values in tasks.values())
            correct = sum(values["correct"] for values in tasks.values())
            result.update({"questions": questions, "answer_exact_match": correct / questions if questions else None,
                           "answer_token_loss": answer_loss / answer_tokens if answer_tokens else None,
                           "answer_tokens": answer_tokens,
                           "answer_macro_accuracy": (sum(v["correct"] / v["questions"] for v in variants.values()) / len(variants)
                                                     if variants else None),
                           "by_variant": {key: {**values, "accuracy": values["correct"] / values["questions"]}
                                          for key, values in sorted(variants.items())},
                           "by_task": {task: {**values, "accuracy": values["correct"] / values["questions"]}
                                       for task, values in sorted(tasks.items())}})
        return result
    finally:
        model.train(was_training)
