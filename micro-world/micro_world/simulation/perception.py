"""Forward-facing, agent-relative partial observation."""

from __future__ import annotations

from micro_world.protocol import (
    DayPhase,
    Direction,
    Observation,
    ObservedCell,
    TileKind,
    WorldState,
    phase_for_tick,
)


AGENT_LOCAL_POSITION = (4, 8)


def local_to_world(
    agent_x: int,
    agent_y: int,
    direction: Direction,
    local_x: int,
    local_y: int,
) -> tuple[int, int]:
    """Map a 9x9 view cell into the world with forward always pointing up."""

    right = local_x - AGENT_LOCAL_POSITION[0]
    forward = AGENT_LOCAL_POSITION[1] - local_y
    if direction == Direction.NORTH:
        return agent_x + right, agent_y - forward
    if direction == Direction.EAST:
        return agent_x + forward, agent_y + right
    if direction == Direction.SOUTH:
        return agent_x - right, agent_y + forward
    return agent_x - forward, agent_y - right


def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    points: list[tuple[int, int]] = []
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return points
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _line_visible(state: WorldState, start: tuple[int, int], end: tuple[int, int]) -> bool:
    # The target wall or door itself is visible; only intermediate blockers hide it.
    for x, y in _bresenham(start, end)[1:-1]:
        if state.tiles[y][x].blocks_vision:
            return False
    return True


def _lamp_lit(state: WorldState, x: int, y: int) -> bool:
    for lamp_y, row in enumerate(state.tiles):
        for lamp_x, tile in enumerate(row):
            if tile.kind != TileKind.LAMP or not tile.active:
                continue
            if max(abs(x - lamp_x), abs(y - lamp_y)) <= 3 and _line_visible(
                state, (lamp_x, lamp_y), (x, y)
            ):
                return True
    return False


def _within_light(state: WorldState, agent_x: int, agent_y: int, x: int, y: int) -> bool:
    phase = phase_for_tick(state.tick, state.config.day_cycle_ticks)
    if phase == DayPhase.DAY:
        return True
    natural_range = 5 if phase in (DayPhase.DUSK, DayPhase.DAWN) else 3
    if max(abs(x - agent_x), abs(y - agent_y)) <= natural_range:
        return True
    return _lamp_lit(state, x, y)


def observe(state: WorldState, agent_id: str) -> Observation:
    agent = state.entities.get(agent_id)
    if agent is None or agent.kind.value != "agent":
        raise KeyError(f"unknown agent: {agent_id}")

    entities_by_position: dict[tuple[int, int], list] = {}
    for entity in state.entities.values():
        if entity.x >= 0 and entity.y >= 0:
            entities_by_position.setdefault(entity.position, []).append(entity)

    rows: list[list[ObservedCell]] = []
    for local_y in range(state.config.observation_depth):
        row: list[ObservedCell] = []
        for local_x in range(state.config.observation_width):
            world_x, world_y = local_to_world(
                agent.x, agent.y, agent.direction, local_x, local_y
            )
            inside = (
                0 <= world_x < state.config.width
                and 0 <= world_y < state.config.height
            )
            visible = bool(
                inside
                and _within_light(state, agent.x, agent.y, world_x, world_y)
                and _line_visible(state, agent.position, (world_x, world_y))
            )
            row.append(
                ObservedCell(
                    local_x,
                    local_y,
                    world_x if inside else None,
                    world_y if inside else None,
                    visible,
                    state.tiles[world_y][world_x] if visible else None,
                    list(entities_by_position.get((world_x, world_y), []))
                    if visible
                    else [],
                )
            )
        rows.append(row)

    return Observation(
        agent_id=agent_id,
        tick=state.tick,
        phase=phase_for_tick(state.tick, state.config.day_cycle_ticks),
        width=state.config.observation_width,
        height=state.config.observation_depth,
        agent_local_position=AGENT_LOCAL_POSITION,
        view_direction=agent.direction,
        cells=rows,
    )
