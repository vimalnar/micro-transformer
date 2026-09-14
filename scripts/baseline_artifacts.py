"""Inspect, install, and verify the optional complete reference-baseline release."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "baselines" / "v1.0.0" / "manifest.json"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def catalog():
    return json.loads(CATALOG_PATH.read_text())


def default_destination(values):
    return ROOT / values["full_release"]["installed_path"]


def safe_member(root, relative):
    member = (root / relative).resolve()
    try:
        member.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Unsafe inventory path: {relative}") from error
    return member


def verify_release(directory, values=None):
    values = values or catalog()
    directory = Path(directory).resolve()
    inventory_path = directory / "SHA256.json"
    if not inventory_path.is_file():
        raise ValueError(f"Missing release inventory: {inventory_path}")
    expected_inventory = values["full_release"]["inventory_sha256"]
    if digest(inventory_path) != expected_inventory:
        raise ValueError("Release inventory hash does not match the v1.0.0 catalog")
    inventory = json.loads(inventory_path.read_text())
    if len(inventory) != values["full_release"]["inventory_entries"]:
        raise ValueError("Release inventory entry count changed")
    for relative, expected in inventory.items():
        path = safe_member(directory, relative)
        if not path.is_file():
            raise ValueError(f"Missing release file: {relative}")
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Hash mismatch: {relative}")
    actual_files = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
    }
    unexpected = actual_files - set(inventory) - {"SHA256.json"}
    if unexpected:
        raise ValueError("Unregistered release files: " + ", ".join(sorted(unexpected)))
    return {"status": "passed", "directory": str(directory), "verified_files": len(inventory)}


def verify_core(values=None):
    values = values or catalog()
    architecture = ROOT / "models" / "transformer.py"
    checkpoint = ROOT / "baselines" / "v1.0.0" / values["included_checkpoint"]["path"]
    checks = {
        "architecture": digest(architecture) == values["architecture"]["sha256"],
        "checkpoint": digest(checkpoint) == values["included_checkpoint"]["sha256"],
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise ValueError(f"Core baseline identity failed: {failed}")
    return {"status": "passed", "checks": checks, "checkpoint": str(checkpoint)}


def install(source, destination, values=None):
    values = values or catalog()
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    verified = verify_release(source, values)
    if destination.exists():
        installed = verify_release(destination, values)
        installed["status"] = "already_installed"
        return installed
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".baseline-install-", dir=destination.parent))
    staging = staging_root / destination.name
    try:
        shutil.copytree(source, staging)
        verify_release(staging, values)
        os.replace(staging, destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    verified.update({"status": "installed", "directory": str(destination)})
    return verified


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="Show the tracked baseline catalog")
    commands.add_parser("verify-core", help="Verify the included architecture and small checkpoint")
    verify = commands.add_parser("verify", help="Verify an installed complete release")
    verify.add_argument("--path", type=Path)
    add = commands.add_parser("install", help="Verify and install an extracted complete release")
    add.add_argument("--source", type=Path, required=True)
    add.add_argument("--destination", type=Path)
    args = parser.parse_args(argv)
    values = catalog()
    destination = default_destination(values)
    if args.command == "info":
        result = values
    elif args.command == "verify-core":
        result = verify_core(values)
    elif args.command == "verify":
        result = verify_release(args.path or destination, values)
    else:
        result = install(args.source, args.destination or destination, values)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
