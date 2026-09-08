"""Agent-facing observation and step-result schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .events import Event
from .actions import Direction
from .state import DayPhase, EntityState, TileState, WorldState


@dataclass
class ObservedCell:
    local_x: int
    local_y: int
    world_x: int | None
    world_y: int | None
    visible: bool
    tile: TileState | None
    entities: list[EntityState]

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_x": self.local_x,
            "local_y": self.local_y,
            "world_x": self.world_x,
            "world_y": self.world_y,
            "visible": self.visible,
            "tile": self.tile.to_dict() if self.tile is not None else None,
            "entities": [entity.to_dict() for entity in self.entities],
        }


@dataclass
class Observation:
    agent_id: str
    tick: int
    phase: DayPhase
    width: int
    height: int
    agent_local_position: tuple[int, int]
    view_direction: Direction
    cells: list[list[ObservedCell]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "tick": self.tick,
            "phase": self.phase.value,
            "width": self.width,
            "height": self.height,
            "agent_local_position": list(self.agent_local_position),
            "view_direction": self.view_direction.name.lower(),
            "cells": [[cell.to_dict() for cell in row] for row in self.cells],
        }


@dataclass
class StepResult:
    state: WorldState
    observation: Observation
    events: list[Event]

    def to_dict(self, include_state: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "observation": self.observation.to_dict(),
            "events": [event.to_dict() for event in self.events],
        }
        if include_state:
            result["state"] = self.state.to_dict()
        return result
