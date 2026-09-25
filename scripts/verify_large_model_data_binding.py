#!/usr/bin/env python3
"""Verify the immutable packed-data binding for the larger v1 reference run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / "baselines/v1.0.0-large/data-binding.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    binding = json.loads(BINDING.read_text(encoding="utf-8"))
    if binding["status"] != "bound_to_verified_packed_release":
        raise SystemExit("unexpected binding status")
    release = ROOT / binding["release"]["root"]
    manifest = release / "SHA256.json"
    if sha256(manifest) != binding["release"]["sha256_manifest_sha256"]:
        raise SystemExit("release SHA256.json hash mismatch")
    expected = json.loads(manifest.read_text(encoding="utf-8"))
    checked = 0
    for split in binding["model_facing_data"].values():
        directory = ROOT / split["directory"]
        for key, filename in (("metadata_sha256", "metadata.json"),
                              ("tokens_sha256", "tokens.bin"),
                              ("offsets_sha256", "offsets.bin"),
                              ("labels_sha256", "labels.bin")):
            path = directory / filename
            actual = sha256(path)
            if actual != split[key]:
                raise SystemExit(f"{path}: binding hash mismatch")
            release_key = str(path.relative_to(release))
            if expected[release_key] != actual:
                raise SystemExit(f"{path}: release manifest hash mismatch")
            checked += 1
    print(json.dumps({"status": "passed", "binding": str(BINDING), "files_checked": checked}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
