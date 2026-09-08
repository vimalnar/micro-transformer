"""Training-only answer weighting; model inputs and architecture stay unchanged."""

import math

import torch
from torch.nn import functional as F


def answer_mask(records, targets):
    """Align completion tokens (including |) with once-shifted LM targets.

    answer_start indexes the original episode, so its first predicted answer is
    at answer_start - 1 in targets. Statement-only episodes have no answer span.
    """
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for row, (tokens, start, _) in enumerate(records):
        if start:
            if not 1 <= start < len(tokens):
                raise ValueError("Invalid answer boundary")
            mask[row, start - 1:len(tokens) - 1] = True
    return mask & targets.ne(-100)


def language_loss(logits, targets, records, answer_weight=1.0):
    """Weighted mean CE: ordinary tokens weight 1, answer tokens weight w.

    Padding never contributes. The default exactly retains the original loss.
    Weighting is supervision only; no expected answer is added to inference.
    """
    if not math.isfinite(answer_weight) or answer_weight < 1:
        raise ValueError("answer_weight must be finite and at least 1")
    if answer_weight == 1:
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
    losses = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100, reduction="none")
    weights = targets.ne(-100).to(losses.dtype)
    weights = weights + answer_mask(records, targets).to(losses.dtype) * (answer_weight - 1)
    return (losses * weights).sum() / weights.sum()


def selection_candidate(metrics, metric, step):
    value = metrics[metric]
    if value is None or not math.isfinite(value):
        raise ValueError(f"Cannot select checkpoint using {metric}: no finite validation score")
    return {"metric": metric, "value": value, "step": step, "token_loss": metrics["loss"]}


def improves_selection(candidate, previous):
    if previous is None:
        return True
    if candidate["metric"] != previous["metric"]:
        raise ValueError("Checkpoint selection metric changed")
    if candidate["metric"] == "loss":
        return candidate["value"] < previous["value"]
    # Equal answer accuracy is resolved by lower ordinary validation token loss;
    # an exact tie retains the earlier checkpoint.
    return (candidate["value"], -candidate["token_loss"]) > (previous["value"], -previous["token_loss"])
