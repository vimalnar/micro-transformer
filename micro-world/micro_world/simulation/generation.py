"""Seeded procedural map construction."""

from __future__ import annotations

import random

from micro_world.protocol import (
    Direction,
    EntityKind,
    EntityState,
    TileKind,
    TileState,
    WorldConfig,
    WorldState,
)

from .maps import empty_tiles


SPECIAL_TILES = (
    TileKind.WATER,
    TileKind.ICE,
    TileKind.PUSH_NORTH,
    TileKind.PUSH_EAST,
    TileKind.PUSH_SOUTH,
    TileKind.PUSH_WEST,
    TileKind.SPIN_CLOCKWISE,
    TileKind.SPIN_ANTICLOCKWISE,
    TileKind.PRESSURE_PLATE,
    TileKind.LAMP,
)


def procedural_world(seed: int, config: WorldConfig) -> WorldState:
    """Create a simple reproducible map with a connected central play area."""

    rng = random.Random(seed)
    tiles = empty_tiles(config)

    # Sparse wall segments; central cross remains open for robust navigation.
    protected = {
        (config.width // 2, y) for y in range(1, config.height - 1)
    } | {
        (x, config.height // 2) for x in range(1, config.width - 1)
    }
    candidates = [
        (x, y)
        for y in range(2, config.height - 2)
        for x in range(2, config.width - 2)
        if (x, y) not in protected
    ]
    rng.shuffle(candidates)
    for x, y in candidates[: max(6, (config.width * config.height) // 24)]:
        tiles[y][x] = TileState(TileKind.WALL)

    floor_cells = [
        (x, y)
        for y in range(1, config.height - 1)
        for x in range(1, config.width - 1)
        if tiles[y][x].kind == TileKind.FLOOR
    ]
    rng.shuffle(floor_cells)

    def take_cell() -> tuple[int, int]:
        if not floor_cells:
            raise RuntimeError("procedural map ran out of free cells")
        return floor_cells.pop()

    for kind in SPECIAL_TILES:
        x, y = take_cell()
        tiles[y][x] = TileState(kind, active=kind == TileKind.LAMP)

    # One accessible door embedded in a short wall segment.
    door_x, door_y = config.width // 2, config.height // 2 - 2
    tiles[door_y][door_x] = TileState(TileKind.DOOR)
    for x in (door_x - 1, door_x + 1):
        tiles[door_y][x] = TileState(TileKind.WALL)
    occupied_by_structure = {
        (door_x - 1, door_y),
        (door_x, door_y),
        (door_x + 1, door_y),
    }
    floor_cells = [cell for cell in floor_cells if cell not in occupied_by_structure]

    entity_specs = (
        ("ava", EntityKind.AGENT, "teal"),
        ("ben", EntityKind.AGENT, "coral"),
        ("key-one", EntityKind.KEY, "red"),
        ("cube-one", EntityKind.CUBE, "green"),
        ("orb-one", EntityKind.ORB, "amber"),
        ("crate-one", EntityKind.CRATE, "brown"),
    )
    entities: dict[str, EntityState] = {}
    for index, (entity_id, kind, colour) in enumerate(entity_specs):
        x, y = take_cell()
        entities[entity_id] = EntityState(
            entity_id,
            kind,
            x,
            y,
            Direction(index % 4),
            colour,
        )

    return WorldState(seed, "procedural", 0, config, tiles, entities)
