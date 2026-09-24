"""Versioned MicroLanguage adapter for bounded Micro-World diagnostics.

The encoding uses only the frozen 128-token vocabulary. Compatibility does not
imply that the released language model learned this observation/action convention.
"""

from __future__ import annotations

from dataclasses import dataclass

from micro_transformer.data.generator import VOCABULARY, VOCABULARY_ORDER
from micro_world.protocol import EntityKind, Observation, TileKind


INTERFACE_VERSION = "micro-world-language-adapter-v1"
OUTPUT_RESERVE = 6

COMMANDS = {
    "ava move .": "forward",
    "ava move one .": "turn-left",
    "ava move two .": "turn-right",
    "ava does .": "interact",
    "ava take .": "take",
    "ava drop .": "drop",
    "ava none .": "wait",
}

TILE_TOKENS = {
    TileKind.FLOOR: "room",
    TileKind.WALL: "closed",
    TileKind.DOOR: "door",
    TileKind.WATER: "river",
    TileKind.ICE: "light",
    TileKind.PUSH_NORTH: "before",
    TileKind.PUSH_EAST: "before",
    TileKind.PUSH_SOUTH: "after",
    TileKind.PUSH_WEST: "after",
    TileKind.SPIN_CLOCKWISE: "again",
    TileKind.SPIN_ANTICLOCKWISE: "again",
    TileKind.PRESSURE_PLATE: "full",
    TileKind.LAMP: "visible",
}

ENTITY_TOKENS = {
    EntityKind.AGENT: "ben",
    EntityKind.KEY: "key",
    EntityKind.CUBE: "cube",
    EntityKind.ORB: "orb",
    EntityKind.CRATE: "box",
}

GOAL_TOKENS = {
    "reach": ("ava", "follow", "lamp", "."),
    "possess": ("ava", "take", "key", "."),
    "deliver": ("ava", "move", "cube", "to", "lamp", "."),
    "open-door": ("ava", "open", "door", "."),
    "activate-plate": ("ava", "move", "cube", "to", "full", "."),
    "visit-return": ("ava", "follow", "lamp", "again", "."),
}


def observation_tokens(observation: Observation) -> list[str]:
    """Encode the complete permitted 9x9 view without world coordinates."""
    values: list[str] = ["all"]
    for row in observation.cells:
        for cell in row:
            if not cell.visible or cell.tile is None:
                values.append("unknown")
                continue
            token = TILE_TOKENS[cell.tile.kind]
            if cell.tile.kind == TileKind.DOOR and cell.tile.open:
                token = "open"
            visible_entities = [
                entity for entity in cell.entities if entity.id != observation.agent_id
            ]
            if visible_entities:
                token = ENTITY_TOKENS[sorted(visible_entities, key=lambda item: item.id)[0].kind]
            values.append(token)
    if len(values) != 82:
        raise ValueError("the adapter requires the fixed 9x9 observation contract")
    return values


def serialize_prompt(observation: Observation, goal_kind: str) -> str:
    if goal_kind not in GOAL_TOKENS:
        raise ValueError(f"unsupported goal kind: {goal_kind}")
    values = [*observation_tokens(observation), ".", *GOAL_TOKENS[goal_kind], "ava"]
    unknown = set(values) - VOCABULARY
    if unknown:
        raise ValueError(f"adapter emitted unsupported vocabulary: {sorted(unknown)}")
    if len(values) > len(VOCABULARY_ORDER) - OUTPUT_RESERVE:
        raise ValueError("observation prompt exceeds the frozen context budget")
    return " ".join(values)


def decode_action(completion: str) -> str | None:
    values = completion.lower().replace("|", " ").split()
    command = " ".join(["ava", *values])
    return COMMANDS.get(command)


@dataclass
class FrozenModelDecision:
    prompt: str
    completion: str
    decoded_action: str | None
    stop_reason: str


class FrozenModelPolicy:
    """Use the frozen language model as an untrained action-continuation probe."""

    name = "frozen-language-model"

    def __init__(self, checkpoint, architecture, device="cpu", seed=42):
        from training.harness import InferenceHarness

        self.harness = InferenceHarness(
            checkpoint,
            device=device,
            architecture=architecture,
            seed=seed,
        )
        self.last_decision: FrozenModelDecision | None = None

    def reset(self, scenario) -> None:
        self.last_decision = None

    def choose(self, observation: Observation, goal) -> str:
        prompt = serialize_prompt(observation, goal.kind)
        result = self.harness.predict(prompt, max_new_tokens=OUTPUT_RESERVE)
        action = decode_action(result["prediction"])
        self.last_decision = FrozenModelDecision(
            prompt,
            result["prediction"],
            action,
            result["stop_reason"],
        )
        return action or "wait"
