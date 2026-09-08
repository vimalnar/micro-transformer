"""Flat, original colour palette for canonical Micro-World observations."""

from __future__ import annotations

from micro_world.protocol import DayPhase, TileKind


RGB = tuple[int, int, int]

UNKNOWN: RGB = (18, 25, 48)
GRID: RGB = (104, 112, 122)
OUTLINE: RGB = (20, 30, 52)

DAY_TILES: dict[TileKind, RGB] = {
    TileKind.FLOOR: (242, 238, 219),
    TileKind.WALL: (52, 66, 84),
    TileKind.DOOR: (155, 135, 100),
    TileKind.WATER: (174, 189, 192),
    TileKind.ICE: (217, 224, 223),
    TileKind.PUSH_NORTH: (216, 209, 184),
    TileKind.PUSH_EAST: (216, 209, 184),
    TileKind.PUSH_SOUTH: (216, 209, 184),
    TileKind.PUSH_WEST: (216, 209, 184),
    TileKind.SPIN_CLOCKWISE: (209, 201, 212),
    TileKind.SPIN_ANTICLOCKWISE: (209, 201, 212),
    TileKind.PRESSURE_PLATE: (203, 187, 187),
    TileKind.LAMP: (221, 212, 170),
}


ENTITY_COLOURS: dict[str, RGB] = {
    "teal": (48, 176, 190),
    "coral": (238, 92, 71),
    "red": (224, 66, 62),
    "green": (79, 149, 61),
    "amber": (246, 184, 45),
    "brown": (143, 92, 48),
    "neutral": (220, 224, 228),
}


def shade(colour: RGB, factor: float) -> RGB:
    return tuple(max(0, min(255, round(channel * factor))) for channel in colour)  # type: ignore[return-value]


def tile_colour(kind: TileKind, phase: DayPhase) -> RGB:
    base = DAY_TILES[kind]
    factor = {
        DayPhase.DAY: 1.0,
        DayPhase.DUSK: 0.82,
        DayPhase.NIGHT: 0.55,
        DayPhase.DAWN: 0.75,
    }[phase]
    return shade(base, factor)
