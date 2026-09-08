"""Dependency-free canonical 128x128 RGB and PNG observation renderer."""

from __future__ import annotations

import struct
import zlib

from micro_world.protocol import Direction, EntityKind, Observation, TileKind

from .palette import ENTITY_COLOURS, GRID, OUTLINE, UNKNOWN, RGB, tile_colour


CANONICAL_SIZE = 128
TILE_SIZE = 14
GRID_OFFSET = 1


class Raster:
    def __init__(self, width: int, height: int, colour: RGB) -> None:
        self.width = width
        self.height = height
        self.data = bytearray(colour * (width * height))

    def pixel(self, x: int, y: int, colour: RGB) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            self.data[offset : offset + 3] = bytes(colour)

    def rect(self, x: int, y: int, width: int, height: int, colour: RGB) -> None:
        for py in range(max(0, y), min(self.height, y + height)):
            for px in range(max(0, x), min(self.width, x + width)):
                self.pixel(px, py, colour)

    def line(self, x0: int, y0: int, x1: int, y1: int, colour: RGB) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            self.pixel(x0, y0, colour)
            if x0 == x1 and y0 == y1:
                return
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += sx
            if doubled <= dx:
                error += dx
                y0 += sy

    def circle(self, cx: int, cy: int, radius: int, colour: RGB) -> None:
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                    self.pixel(x, y, colour)


def _relative_direction(world: Direction, view: Direction) -> Direction:
    return Direction((int(world) - int(view)) % 4)


def _draw_arrow(raster: Raster, x: int, y: int, direction: Direction) -> None:
    cx, cy = x + TILE_SIZE // 2, y + TILE_SIZE // 2
    dx, dy = direction.delta
    raster.line(cx - dx * 3, cy - dy * 3, cx + dx * 3, cy + dy * 3, OUTLINE)
    if direction in (Direction.NORTH, Direction.SOUTH):
        raster.line(cx + dx * 3, cy + dy * 3, cx - 2, cy + dy, OUTLINE)
        raster.line(cx + dx * 3, cy + dy * 3, cx + 2, cy + dy, OUTLINE)
    else:
        raster.line(cx + dx * 3, cy + dy * 3, cx + dx, cy - 2, OUTLINE)
        raster.line(cx + dx * 3, cy + dy * 3, cx + dx, cy + 2, OUTLINE)


def _draw_tile(raster: Raster, x: int, y: int, cell, phase) -> RGB:
    if not cell.visible or cell.tile is None:
        raster.rect(x, y, TILE_SIZE, TILE_SIZE, UNKNOWN)
        return UNKNOWN
    tile = cell.tile
    colour = tile_colour(tile.kind, phase)
    raster.rect(x, y, TILE_SIZE, TILE_SIZE, colour)
    if tile.kind == TileKind.WATER:
        mark = tuple(round(channel * 0.72) for channel in colour)
        raster.line(x + 3, y + 5, x + 10, y + 5, mark)
        raster.line(x + 3, y + 8, x + 10, y + 8, mark)
    elif tile.kind == TileKind.ICE:
        mark = tuple(round(channel * 0.78) for channel in colour)
        raster.line(x + 3, y + 10, x + 10, y + 3, mark)
    elif tile.kind in (
        TileKind.PUSH_NORTH,
        TileKind.PUSH_EAST,
        TileKind.PUSH_SOUTH,
        TileKind.PUSH_WEST,
    ):
        direction = {
            TileKind.PUSH_NORTH: Direction.NORTH,
            TileKind.PUSH_EAST: Direction.EAST,
            TileKind.PUSH_SOUTH: Direction.SOUTH,
            TileKind.PUSH_WEST: Direction.WEST,
        }[tile.kind]
        _draw_arrow(raster, x, y, direction)
    elif tile.kind in (TileKind.SPIN_CLOCKWISE, TileKind.SPIN_ANTICLOCKWISE):
        raster.circle(x + 7, y + 7, 4, OUTLINE)
        raster.circle(x + 7, y + 7, 2, colour)
    elif tile.kind == TileKind.DOOR:
        inset = 2 if tile.open else 4
        raster.rect(x + inset, y + 2, max(2, TILE_SIZE - inset * 2), TILE_SIZE - 4, OUTLINE)
        raster.rect(x + inset + 1, y + 3, max(1, TILE_SIZE - inset * 2 - 2), TILE_SIZE - 6, colour)
    elif tile.kind == TileKind.PRESSURE_PLATE:
        raster.rect(x + 3, y + 3, 8, 8, OUTLINE)
        raster.rect(x + 4, y + 4, 6, 6, colour)
    elif tile.kind == TileKind.LAMP:
        raster.circle(x + 7, y + 7, 4, OUTLINE)
        raster.circle(x + 7, y + 7, 2, (255, 244, 145))
    return colour


def _draw_entity(raster: Raster, x: int, y: int, entity, background: RGB, view: Direction) -> None:
    colour = ENTITY_COLOURS.get(entity.colour, ENTITY_COLOURS["neutral"])
    cx, cy = x + 7, y + 7
    if entity.kind == EntityKind.AGENT:
        raster.circle(cx, cy, 5, OUTLINE)
        raster.circle(cx, cy, 4, colour)
        direction = _relative_direction(entity.direction, view)
        dx, dy = direction.delta
        for distance in range(1, 5):
            spread = max(0, 3 - distance)
            for offset in range(-spread, spread + 1):
                px = cx + dx * distance + (-dy) * offset
                py = cy + dy * distance + dx * offset
                raster.pixel(px, py, background)
    elif entity.kind == EntityKind.ORB:
        raster.circle(cx, cy, 4, OUTLINE)
        raster.circle(cx, cy, 3, colour)
    elif entity.kind == EntityKind.CUBE:
        raster.rect(x + 3, y + 3, 8, 8, OUTLINE)
        raster.rect(x + 4, y + 4, 6, 6, colour)
    elif entity.kind == EntityKind.CRATE:
        raster.rect(x + 2, y + 2, 10, 10, OUTLINE)
        raster.rect(x + 3, y + 3, 8, 8, colour)
        raster.line(x + 4, y + 4, x + 9, y + 9, OUTLINE)
        raster.line(x + 9, y + 4, x + 4, y + 9, OUTLINE)
    elif entity.kind == EntityKind.KEY:
        raster.circle(x + 5, y + 5, 3, OUTLINE)
        raster.circle(x + 5, y + 5, 1, background)
        raster.rect(x + 7, y + 5, 5, 2, colour)
        raster.pixel(x + 10, y + 7, OUTLINE)


def render_rgb(observation: Observation) -> bytes:
    """Render the canonical 128x128, three-channel, row-major RGB frame."""

    if observation.width != 9 or observation.height != 9:
        raise ValueError("canonical renderer requires a 9x9 observation")
    raster = Raster(CANONICAL_SIZE, CANONICAL_SIZE, GRID)
    for row in observation.cells:
        for cell in row:
            x = GRID_OFFSET + cell.local_x * TILE_SIZE
            y = GRID_OFFSET + cell.local_y * TILE_SIZE
            background = _draw_tile(raster, x, y, cell, observation.phase)
            for entity in cell.entities:
                _draw_entity(raster, x, y, entity, background, observation.view_direction)
            raster.rect(x, y, TILE_SIZE, 1, GRID)
            raster.rect(x, y, 1, TILE_SIZE, GRID)
    return bytes(raster.data)


def encode_png(rgb: bytes, width: int = CANONICAL_SIZE, height: int = CANONICAL_SIZE) -> bytes:
    if len(rgb) != width * height * 3:
        raise ValueError("RGB byte count does not match image dimensions")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    rows = b"".join(
        b"\x00" + rgb[y * width * 3 : (y + 1) * width * 3]
        for y in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, level=9))
        + chunk(b"IEND", b"")
    )


def render_png(observation: Observation) -> bytes:
    return encode_png(render_rgb(observation))
