"""Serializable events emitted by the authoritative simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    tick: int
    kind: str
    actor: str | None = None
    position: tuple[int, int] | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "kind": self.kind,
            "actor": self.actor,
            "position": list(self.position) if self.position is not None else None,
            "details": self.details,
        }

