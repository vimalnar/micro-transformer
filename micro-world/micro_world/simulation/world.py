"""Authoritative deterministic Micro-World simulation."""

from __future__ import annotations

import copy
from dataclasses import replace

from micro_world.protocol import (
    Action,
    ActionType,
    Direction,
    EntityKind,
    EntityState,
    Event,
    StepResult,
    TileKind,
    WorldConfig,
    WorldState,
    phase_for_tick,
)

from .generation import procedural_world
from .maps import HANDCRAFTED_MAPS
from .perception import observe


PUSH_DIRECTIONS = {
    TileKind.PUSH_NORTH: Direction.NORTH,
    TileKind.PUSH_EAST: Direction.EAST,
    TileKind.PUSH_SOUTH: Direction.SOUTH,
    TileKind.PUSH_WEST: Direction.WEST,
}
MOVABLE_KINDS = {EntityKind.AGENT, EntityKind.CUBE, EntityKind.ORB, EntityKind.CRATE}
PUSHABLE_KINDS = {EntityKind.CUBE, EntityKind.ORB, EntityKind.CRATE}


class World:
    """A mutable session wrapper around serializable world state."""

    def __init__(self, state: WorldState) -> None:
        self.state = state
        self.events: list[Event] = []
        self.action_history: list[dict[str, str | int]] = []

    @classmethod
    def reset(
        cls,
        seed: int = 42,
        configuration: WorldConfig | None = None,
        map_name: str = "physics-lab",
    ) -> "World":
        config = configuration or WorldConfig()
        if map_name == "procedural":
            state = procedural_world(seed, config)
        elif map_name in HANDCRAFTED_MAPS:
            state = HANDCRAFTED_MAPS[map_name](seed, config)
        else:
            raise ValueError(f"unknown map: {map_name}")
        return cls(state)

    def snapshot(self) -> WorldState:
        return copy.deepcopy(self.state)

    def observe(self, agent_id: str = "ava"):
        return observe(self.state, agent_id)

    def entities_at(self, x: int, y: int) -> list[EntityState]:
        return [
            entity
            for entity in self.state.entities.values()
            if entity.x == x and entity.y == y
        ]

    def _emit(
        self,
        events: list[Event],
        tick: int,
        kind: str,
        actor: str | None = None,
        position: tuple[int, int] | None = None,
        **details,
    ) -> None:
        events.append(Event(tick, kind, actor, position, details))

    def _inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.state.config.width and 0 <= y < self.state.config.height

    def _can_enter(self, x: int, y: int, ignore_entity: str | None = None) -> bool:
        if not self._inside(x, y) or self.state.tiles[y][x].blocks_movement:
            return False
        return not any(
            entity.id != ignore_entity and entity.kind in MOVABLE_KINDS
            for entity in self.entities_at(x, y)
        )

    def _move_entity(
        self,
        entity: EntityState,
        direction: Direction,
        tick: int,
        events: list[Event],
        cause: str,
        allow_push: bool = False,
    ) -> bool:
        dx, dy = direction.delta
        target_x, target_y = entity.x + dx, entity.y + dy
        occupants = self.entities_at(target_x, target_y)
        blocking = next((item for item in occupants if item.kind in MOVABLE_KINDS), None)
        if blocking is not None and allow_push and blocking.kind in PUSHABLE_KINDS:
            if not self._move_entity(
                blocking, direction, tick, events, "pushed", allow_push=False
            ):
                blocking = blocking
            else:
                blocking = None
        if blocking is not None or not self._can_enter(target_x, target_y, entity.id):
            self._emit(events, tick, "blocked", entity.id, entity.position, cause=cause)
            entity.sliding = None
            return False
        entity.x, entity.y = target_x, target_y
        self._emit(
            events,
            tick,
            "moved",
            entity.id,
            entity.position,
            direction=direction.name.lower(),
            cause=cause,
        )
        tile = self.state.tiles[entity.y][entity.x]
        if tile.kind == TileKind.ICE:
            entity.sliding = direction
        elif cause != "conveyor":
            entity.sliding = None
        return True

    def _interact(self, agent: EntityState, tick: int, events: list[Event]) -> None:
        dx, dy = agent.direction.delta
        x, y = agent.x + dx, agent.y + dy
        if not self._inside(x, y):
            self._emit(events, tick, "interaction-failed", agent.id, agent.position)
            return
        tile = self.state.tiles[y][x]
        if tile.kind != TileKind.DOOR:
            self._emit(events, tick, "interaction-failed", agent.id, (x, y))
            return
        has_key = any(
            self.state.entities[item].kind == EntityKind.KEY
            for item in agent.inventory
            if item in self.state.entities
        )
        if tile.locked and not has_key:
            self._emit(events, tick, "door-locked", agent.id, (x, y))
            return
        if tile.locked:
            tile.locked = False
            self._emit(events, tick, "door-unlocked", agent.id, (x, y))
        tile.open = not tile.open
        self._emit(
            events,
            tick,
            "door-opened" if tile.open else "door-closed",
            agent.id,
            (x, y),
        )

    def _take(self, agent: EntityState, tick: int, events: list[Event]) -> None:
        dx, dy = agent.direction.delta
        x, y = agent.x + dx, agent.y + dy
        item = next(
            (
                entity
                for entity in self.entities_at(x, y)
                if entity.kind != EntityKind.AGENT
            ),
            None,
        )
        if item is None or len(agent.inventory) >= 1:
            self._emit(events, tick, "take-failed", agent.id, (x, y))
            return
        item.x = item.y = -1
        item.sliding = None
        agent.inventory.append(item.id)
        self._emit(events, tick, "taken", agent.id, agent.position, item=item.id)

    def _drop(self, agent: EntityState, tick: int, events: list[Event]) -> None:
        if not agent.inventory:
            self._emit(events, tick, "drop-failed", agent.id, agent.position)
            return
        dx, dy = agent.direction.delta
        x, y = agent.x + dx, agent.y + dy
        if not self._can_enter(x, y):
            self._emit(events, tick, "drop-failed", agent.id, (x, y))
            return
        item_id = agent.inventory.pop(0)
        item = self.state.entities[item_id]
        item.x, item.y = x, y
        self._emit(events, tick, "dropped", agent.id, (x, y), item=item.id)

    def _automatic_tiles(
        self,
        tick: int,
        events: list[Event],
        manually_moved: set[str],
    ) -> None:
        for entity_id in sorted(self.state.entities):
            entity = self.state.entities[entity_id]
            if entity.kind not in MOVABLE_KINDS or entity.x < 0:
                continue
            tile = self.state.tiles[entity.y][entity.x]
            if tile.kind in PUSH_DIRECTIONS:
                self._move_entity(
                    entity,
                    PUSH_DIRECTIONS[tile.kind],
                    tick,
                    events,
                    "conveyor",
                )
            elif tile.kind == TileKind.ICE and entity.sliding is not None:
                if entity.id not in manually_moved:
                    self._move_entity(
                        entity, entity.sliding, tick, events, "sliding"
                    )
            elif tile.kind == TileKind.SPIN_CLOCKWISE and entity.kind == EntityKind.AGENT:
                entity.direction = entity.direction.right()
                self._emit(events, tick, "rotated", entity.id, entity.position, turn="right")
            elif tile.kind == TileKind.SPIN_ANTICLOCKWISE and entity.kind == EntityKind.AGENT:
                entity.direction = entity.direction.left()
                self._emit(events, tick, "rotated", entity.id, entity.position, turn="left")

    def _pressure_plates(self, tick: int, events: list[Event]) -> None:
        newly_active = False
        for y, row in enumerate(self.state.tiles):
            for x, tile in enumerate(row):
                if tile.kind != TileKind.PRESSURE_PLATE:
                    continue
                occupied = bool(self.entities_at(x, y))
                newly_active = newly_active or (occupied and not tile.active)
                tile.active = occupied
        if not newly_active:
            return
        for y, row in enumerate(self.state.tiles):
            for x, tile in enumerate(row):
                if tile.kind == TileKind.DOOR and not tile.locked and not tile.open:
                    tile.open = True
                    self._emit(events, tick, "door-opened", position=(x, y), cause="plate")

    def _spread_water(self, tick: int, events: list[Event]) -> None:
        if not self.state.config.water_spreads:
            return
        candidates: set[tuple[int, int]] = set()
        for y, row in enumerate(self.state.tiles):
            for x, tile in enumerate(row):
                if tile.kind != TileKind.WATER:
                    continue
                for direction in Direction:
                    dx, dy = direction.delta
                    nx, ny = x + dx, y + dy
                    if self._inside(nx, ny) and self.state.tiles[ny][nx].kind == TileKind.FLOOR:
                        candidates.add((nx, ny))
                    elif (
                        self._inside(nx, ny)
                        and self.state.tiles[ny][nx].kind == TileKind.DOOR
                        and self.state.tiles[ny][nx].open
                    ):
                        beyond_x, beyond_y = nx + dx, ny + dy
                        if (
                            self._inside(beyond_x, beyond_y)
                            and self.state.tiles[beyond_y][beyond_x].kind == TileKind.FLOOR
                        ):
                            candidates.add((beyond_x, beyond_y))
        if candidates:
            x, y = min(candidates, key=lambda point: (point[1], point[0]))
            self.state.tiles[y][x] = replace(self.state.tiles[y][x], kind=TileKind.WATER)
            self._emit(events, tick, "water-spread", position=(x, y))

    def step(self, action: str | ActionType | Action, agent_id: str = "ava") -> StepResult:
        parsed = Action.parse(action)
        agent = self.state.entities.get(agent_id)
        if agent is None or agent.kind != EntityKind.AGENT:
            raise KeyError(f"unknown agent: {agent_id}")

        tick = self.state.tick + 1
        previous_phase = phase_for_tick(self.state.tick, self.state.config.day_cycle_ticks)
        events: list[Event] = []
        manually_moved: set[str] = set()

        if parsed.kind == ActionType.FORWARD:
            if self._move_entity(
                agent, agent.direction, tick, events, "action", allow_push=True
            ):
                manually_moved.add(agent.id)
        elif parsed.kind == ActionType.TURN_LEFT:
            agent.direction = agent.direction.left()
            self._emit(events, tick, "turned", agent.id, agent.position, turn="left")
        elif parsed.kind == ActionType.TURN_RIGHT:
            agent.direction = agent.direction.right()
            self._emit(events, tick, "turned", agent.id, agent.position, turn="right")
        elif parsed.kind == ActionType.INTERACT:
            self._interact(agent, tick, events)
        elif parsed.kind == ActionType.TAKE:
            self._take(agent, tick, events)
        elif parsed.kind == ActionType.DROP:
            self._drop(agent, tick, events)
        else:
            self._emit(events, tick, "waited", agent.id, agent.position)

        self._automatic_tiles(tick, events, manually_moved)
        self._pressure_plates(tick, events)
        self._spread_water(tick, events)
        self.state.tick = tick

        current_phase = phase_for_tick(tick, self.state.config.day_cycle_ticks)
        if current_phase != previous_phase:
            self._emit(events, tick, "phase-changed", phase=current_phase.value)

        self.events.extend(events)
        self.action_history.append(
            {"tick": tick, "agent_id": agent_id, "action": parsed.kind.value}
        )
        return StepResult(self.snapshot(), self.observe(agent_id), events)

    def replay_record(self) -> dict:
        return {
            "seed": self.state.seed,
            "map_name": self.state.map_name,
            "configuration": self.state.config.to_dict(),
            "actions": list(self.action_history),
        }

    @classmethod
    def replay(cls, record: dict) -> "World":
        config = WorldConfig(**record["configuration"])
        world = cls.reset(record["seed"], config, record["map_name"])
        for item in record["actions"]:
            world.step(item["action"], item["agent_id"])
        return world
