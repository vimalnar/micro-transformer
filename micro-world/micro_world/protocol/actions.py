"""Discrete environmental actions, separate from the text vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class Direction(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    @property
    def delta(self) -> tuple[int, int]:
        return ((0, -1), (1, 0), (0, 1), (-1, 0))[int(self)]

    def left(self) -> "Direction":
        return Direction((int(self) - 1) % 4)

    def right(self) -> "Direction":
        return Direction((int(self) + 1) % 4)

    def opposite(self) -> "Direction":
        return Direction((int(self) + 2) % 4)


class ActionType(str, Enum):
    FORWARD = "forward"
    TURN_LEFT = "turn-left"
    TURN_RIGHT = "turn-right"
    INTERACT = "interact"
    TAKE = "take"
    DROP = "drop"
    WAIT = "wait"


@dataclass(frozen=True)
class Action:
    kind: ActionType

    @classmethod
    def parse(cls, value: str | ActionType | "Action") -> "Action":
        if isinstance(value, cls):
            return value
        if isinstance(value, ActionType):
            return cls(value)
        return cls(ActionType(value))

