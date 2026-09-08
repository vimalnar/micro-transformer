"""World-state schemas shared by simulation, media, data, and service layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .actions import Direction


class TileKind(str, Enum):
    FLOOR = "floor"
    WALL = "wall"
    DOOR = "door"
    WATER = "water"
    ICE = "ice"
    PUSH_NORTH = "push-north"
    PUSH_EAST = "push-east"
    PUSH_SOUTH = "push-south"
    PUSH_WEST = "push-west"
    SPIN_CLOCKWISE = "spin-clockwise"
    SPIN_ANTICLOCKWISE = "spin-anticlockwise"
    PRESSURE_PLATE = "pressure-plate"
    LAMP = "lamp"


class EntityKind(str, Enum):
    AGENT = "agent"
    KEY = "key"
    CUBE = "cube"
    ORB = "orb"
    CRATE = "crate"


class DayPhase(str, Enum):
    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"
    DAWN = "dawn"


@dataclass
class TileState:
    kind: TileKind = TileKind.FLOOR
    open: bool = False
    locked: bool = False
    active: bool = False

    @property
    def blocks_movement(self) -> bool:
        return self.kind == TileKind.WALL or (
            self.kind == TileKind.DOOR and not self.open
        )

    @property
    def blocks_vision(self) -> bool:
        return self.blocks_movement

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "open": self.open,
            "locked": self.locked,
            "active": self.active,
        }


@dataclass
class EntityState:
    id: str
    kind: EntityKind
    x: int
    y: int
    direction: Direction = Direction.NORTH
    colour: str = "neutral"
    inventory: list[str] = field(default_factory=list)
    sliding: Direction | None = None

    @property
    def position(self) -> tuple[int, int]:
        return self.x, self.y

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "x": self.x,
            "y": self.y,
            "direction": self.direction.name.lower(),
            "colour": self.colour,
            "inventory": list(self.inventory),
            "sliding": self.sliding.name.lower() if self.sliding is not None else None,
        }


@dataclass(frozen=True)
class WorldConfig:
    width: int = 16
    height: int = 16
    observation_width: int = 9
    observation_depth: int = 9
    day_cycle_ticks: int = 100
    # Static terrain is the baseline. Flooding is an opt-in experiment so an
    # ordinary action never appears to paint new tiles into the world.
    water_spreads: bool = False

    def __post_init__(self) -> None:
        if self.width < 5 or self.height < 5:
            raise ValueError("world dimensions must be at least 5x5")
        if self.observation_width != 9 or self.observation_depth != 9:
            raise ValueError("the baseline observation contract is fixed at 9x9")
        if self.day_cycle_ticks < 10:
            raise ValueError("day_cycle_ticks must be at least 10")

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "observation_width": self.observation_width,
            "observation_depth": self.observation_depth,
            "day_cycle_ticks": self.day_cycle_ticks,
            "water_spreads": self.water_spreads,
        }


@dataclass
class WorldState:
    seed: int
    map_name: str
    tick: int
    config: WorldConfig
    tiles: list[list[TileState]]
    entities: dict[str, EntityState]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "map_name": self.map_name,
            "tick": self.tick,
            "phase": phase_for_tick(self.tick, self.config.day_cycle_ticks).value,
            "config": self.config.to_dict(),
            "tiles": [[tile.to_dict() for tile in row] for row in self.tiles],
            "entities": {
                entity_id: entity.to_dict()
                for entity_id, entity in sorted(self.entities.items())
            },
        }


def phase_for_tick(tick: int, cycle_ticks: int = 100) -> DayPhase:
    point = tick % cycle_ticks
    day_end = round(cycle_ticks * 0.60)
    dusk_end = round(cycle_ticks * 0.70)
    night_end = round(cycle_ticks * 0.90)
    if point < day_end:
        return DayPhase.DAY
    if point < dusk_end:
        return DayPhase.DUSK
    if point < night_end:
        return DayPhase.NIGHT
    return DayPhase.DAWN
