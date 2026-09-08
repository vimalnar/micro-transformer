"""Small extension contract for executable-language task families."""

from dataclasses import dataclass
import importlib.util
import re
from pathlib import Path
import sys
from typing import Callable


@dataclass(frozen=True)
class TaskSpec:
    name: str
    version: str
    description: str
    variants: tuple[str, ...]
    builder: Callable
    boolean_variants: tuple[str, ...] = ()


class TaskRegistry:
    def __init__(self):
        self.tasks = {}
        self.sources = []

    def register(self, spec):
        if not isinstance(spec, TaskSpec) or not re.fullmatch(r"[a-z][a-z0-9_]*", spec.name):
            raise ValueError("Tasks need a TaskSpec and a lowercase identifier")
        if spec.name in self.tasks:
            raise ValueError(f"Task already registered: {spec.name}")
        if not spec.version or not spec.variants or len(set(spec.variants)) != len(spec.variants):
            raise ValueError("Task version and unique variants are required")
        if not set(spec.boolean_variants) <= set(spec.variants) or not callable(spec.builder):
            raise ValueError("Invalid task builder or boolean variants")
        self.tasks[spec.name] = spec

    def describe(self):
        return {name: {"version": spec.version, "description": spec.description,
                       "variants": list(spec.variants), "boolean_variants": list(spec.boolean_variants)}
                for name, spec in sorted(self.tasks.items())}

    def load(self, path, api):
        """Explicit opt-in: a local extension is executable Python, not a data file."""
        path = Path(path).resolve()
        if path.suffix != ".py" or not path.is_file():
            raise ValueError(f"Task module must be a Python file: {path}")
        spec = importlib.util.spec_from_file_location("micro_transformer_task_" + str(len(self.sources)), path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if not callable(getattr(module, "register", None)):
            raise ValueError("Task modules must export register(registry, api)")
        module.register(self, api)
        self.sources.append(path)
