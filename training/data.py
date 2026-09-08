"""Validated JSONL -> compact byte tokens, and independent episode batches."""

import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import struct
import sys
import tempfile

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "micro-world"))
from micro_transformer.data.generator import VOCABULARY_ORDER, validate_record, entity_keys

TOKENS = list(VOCABULARY_ORDER)
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKENS)}
PAD_ID = len(TOKENS)
EOS_ID = TOKEN_TO_ID["|"]
FORMAT = "micro-transformer-packed-v2"


def file_hash(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def encode(text):
    tokens = re.findall(r"[a-z]+|[.?|]|\S", text.lower())
    unknown = sorted(set(tokens) - TOKEN_TO_ID.keys())
    if unknown:
        raise ValueError(f"Unknown language tokens: {unknown}")
    if not tokens:
        raise ValueError("Prompt is empty")
    return [TOKEN_TO_ID[token] for token in tokens]


def decode(ids):
    return " ".join(TOKENS[value] for value in ids).replace(" .", ".").replace(" ?", "?")


def prepare(source, destination, limit=None):
    """Keep one episode per record. Reject bad labels, duplicates, or mixed splits."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError(f"Prepared output already exists: {destination}")
    if limit is not None and limit < 1:
        raise ValueError("Episode limit must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tasks, variants, splits, seen, entities = [], [], set(), set(), set()
    source_digest = file_hash(source)
    count = total_tokens = max_length = question_count = 0
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        with (stage / "tokens.bin").open("wb") as tokens_out, \
             (stage / "offsets.bin").open("wb") as offsets_out, \
             (stage / "labels.bin").open("wb") as labels_out, source.open() as stream:
            offsets_out.write(struct.pack("<Q", 0))
            for line in stream:
                record = json.loads(line)
                issues = validate_record(record)
                if issues:
                    raise ValueError(f"Episode {count}: {issues[0]}")
                tokens = record["tokens"]
                encoded = bytes(TOKEN_TO_ID[token] for token in tokens)
                fingerprint = hashlib.blake2b(encoded, digest_size=16).digest()
                if fingerprint in seen:
                    raise ValueError(f"Duplicate episode at record {count}")
                seen.add(fingerprint)
                splits.add(record["split"])
                entities.update(entity_keys(tokens))
                if record["task"] not in tasks:
                    tasks.append(record["task"])
                # Index of the first answer token, or zero for statement-only data.
                answer_start = tokens.index("?") + 1 if "?" in tokens else 0
                annotation = {key: record.get(key, "legacy") for key in ("variant", "task_version", "suite")}
                annotation["task"] = record["task"]
                if annotation not in variants:
                    variants.append(annotation)
                labels_out.write(struct.pack("<III", answer_start, tasks.index(record["task"]), variants.index(annotation)))
                tokens_out.write(encoded)
                total_tokens += len(tokens)
                offsets_out.write(struct.pack("<Q", total_tokens))
                max_length = max(max_length, len(tokens))
                question_count += bool(answer_start)
                count += 1
                if count % 100000 == 0:
                    print(f"Validated and packed {count:,} episodes", flush=True)
                if limit is not None and count >= limit:
                    break
        if not count or len(splits) != 1:
            raise ValueError("Input must contain episodes from exactly one split")
        if source_digest != file_hash(source):
            raise ValueError("Source changed during preparation")
        files = {name: file_hash(stage / name) for name in ("tokens.bin", "offsets.bin", "labels.bin")}
        metadata = {
            "format": FORMAT, "source": str(source), "source_sha256": source_digest,
            "episodes": count, "tokens": total_tokens, "questions": question_count,
            "max_episode_tokens": max_length, "split": splits.pop(), "tokens_by_id": TOKENS,
            "tasks": tasks, "variants": variants, "entities": sorted(entities), "file_sha256": files,
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        stage.rename(destination)
    return metadata


class EpisodeDataset:
    """Read compact data through memory maps; no corpus-sized Python token lists."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if self.metadata["format"] not in (FORMAT, "micro-transformer-packed-v1") or self.metadata["tokens_by_id"] != TOKENS:
            raise ValueError("Unsupported dataset format or token mapping")
        for name, digest in self.metadata["file_sha256"].items():
            if file_hash(self.directory / name) != digest:
                raise ValueError(f"Prepared dataset checksum mismatch: {name}")
        self._handles = [open(self.directory / name, "rb") for name in ("tokens.bin", "offsets.bin", "labels.bin")]
        self.tokens, self.offsets, self.labels = [mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) for f in self._handles]
        self.label_bytes = 12 if self.metadata["format"] == FORMAT else 8
        if (len(self.offsets) != (len(self) + 1) * 8 or len(self.labels) != len(self) * self.label_bytes or
                len(self.tokens) != self.metadata["tokens"]):
            self.close()
            raise ValueError("Prepared dataset dimensions are inconsistent")

    def __len__(self):
        return self.metadata["episodes"]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start, end = struct.unpack_from("<QQ", self.offsets, index * 8)
        answer_start, task = struct.unpack_from("<II", self.labels, index * self.label_bytes)
        return torch.tensor(list(self.tokens[start:end]), dtype=torch.long), answer_start, self.metadata["tasks"][task]

    def annotations(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.label_bytes == 8:
            return {"variant": "legacy", "task_version": "legacy", "suite": "legacy"}
        variant = struct.unpack_from("<I", self.labels, index * self.label_bytes + 8)[0]
        return {key: value for key, value in self.metadata["variants"][variant].items() if key != "task"}

    def close(self):
        for name in ("tokens", "offsets", "labels"):
            if hasattr(self, name):
                getattr(self, name).close()
        for handle in self._handles:
            handle.close()


def collate(records, context_length, device):
    """Shift targets once; never join separate episodes or learn padding."""
    length = max(len(tokens) - 1 for tokens, _, _ in records)
    if length > context_length:
        raise ValueError(f"Episode needs {length} input positions; context is {context_length}")
    inputs = torch.full((len(records), length), PAD_ID, dtype=torch.long)
    targets = torch.full_like(inputs, -100)
    for row, (tokens, _, _) in enumerate(records):
        inputs[row, :len(tokens) - 1] = tokens[:-1]
        targets[row, :len(tokens) - 1] = tokens[1:]
    return inputs.to(device), targets.to(device)
