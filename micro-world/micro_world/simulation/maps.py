"""Handcrafted deterministic maps used for demonstrations and regression tests."""

from __future__ import annotations

from micro_world.protocol import (
    Direction,
    EntityKind,
    EntityState,
    TileKind,
    TileState,
    WorldConfig,
    WorldState,
)


def empty_tiles(config: WorldConfig) -> list[list[TileState]]:
    tiles = [
        [TileState() for _ in range(config.width)]
        for _ in range(config.height)
    ]
    for x in range(config.width):
        tiles[0][x] = TileState(TileKind.WALL)
        tiles[config.height - 1][x] = TileState(TileKind.WALL)
    for y in range(config.height):
        tiles[y][0] = TileState(TileKind.WALL)
        tiles[y][config.width - 1] = TileState(TileKind.WALL)
    return tiles


def physics_lab(seed: int, config: WorldConfig) -> WorldState:
    """Return the fixed reference map containing every baseline mechanic."""

    tiles = empty_tiles(config)

    # Two rooms with one central door.
    divider_y = config.height // 2
    door_x = config.width // 2
    for x in range(1, config.width - 1):
        tiles[divider_y][x] = TileState(TileKind.WALL)
    tiles[divider_y][door_x] = TileState(TileKind.DOOR, open=False)

    # A partial upper wall creates visible occlusion without sealing the map.
    for y in range(2, divider_y - 1):
        tiles[y][config.width - 5] = TileState(TileKind.WALL)
    tiles[4][config.width - 5] = TileState(TileKind.DOOR, open=False, locked=True)

    # Small reservoir. Flooding expands by at most one cell per tick.
    for x, y in ((2, 2), (3, 2), (2, 3), (3, 3)):
        tiles[y][x] = TileState(TileKind.WATER)

    # A lower-room physics path.
    for x in range(4, 8):
        tiles[12][x] = TileState(TileKind.ICE)
    tiles[12][8] = TileState(TileKind.PUSH_EAST)
    tiles[12][9] = TileState(TileKind.PUSH_NORTH)
    tiles[11][9] = TileState(TileKind.SPIN_CLOCKWISE)
    tiles[13][3] = TileState(TileKind.PRESSURE_PLATE)
    tiles[10][2] = TileState(TileKind.LAMP, active=True)
    tiles[4][13] = TileState(TileKind.LAMP, active=True)

    entities = {
        "ava": EntityState(
            "ava", EntityKind.AGENT, 4, 13, Direction.NORTH, "teal"
        ),
        "ben": EntityState(
            "ben", EntityKind.AGENT, 13, 5, Direction.WEST, "coral"
        ),
        "key-one": EntityState(
            "key-one", EntityKind.KEY, 5, 10, colour="red"
        ),
        "cube-one": EntityState(
            "cube-one", EntityKind.CUBE, 6, 6, colour="green"
        ),
        "orb-one": EntityState(
            "orb-one", EntityKind.ORB, 4, 12, colour="amber"
        ),
        "crate-one": EntityState(
            "crate-one", EntityKind.CRATE, 10, 12, colour="brown"
        ),
    }
    return WorldState(seed, "physics-lab", 0, config, tiles, entities)


HANDCRAFTED_MAPS = {"physics-lab": physics_lab}
