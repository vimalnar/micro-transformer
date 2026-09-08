#!/usr/bin/env python3
"""Generate and validate reproducible Micro-Transformer v1 training datasets.

Generation is streaming and dependency-free. It supports exact task quotas,
duplicate rejection, structural splits, difficulty tiers, sharding, resumption,
coverage requirements, disjointness checks, and reproducibility manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable


VOCABULARY_ORDER = (
    ".", "?", "|", "then", "if", "else", "not", "and", "to", "from", "before", "after", "because", "while", "only", "again",
    "move", "take", "drop", "give", "open", "close", "lock", "unlock", "paint", "hide", "find", "build", "put", "remove", "swap", "follow",
    "where", "who", "what", "which", "is", "count", "colour", "yes", "no", "unknown", "does", "can",
    "in", "at", "has", "with", "behind", "beside", "inside", "contains", "sees", "knows", "believes", "tell",
    "red", "blue", "green", "yellow", "small", "heavy", "light", "closed", "locked", "hidden", "new", "old", "empty", "full", "visible", "broken",
    "ava", "ben", "cy", "da", "eri", "fin", "gia", "hal",
    "cube", "key", "orb", "box", "ball", "book", "gem", "tool", "gate", "door", "coin", "flag", "map", "ring", "lamp", "chest",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "garden", "hall", "tower", "cave", "river", "lab", "store", "home", "bridge", "room", "forest", "dock",
    "all", "some", "none", "same",
)
VOCABULARY = frozenset(VOCABULARY_ORDER)
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCABULARY_ORDER)}
assert len(VOCABULARY_ORDER) == len(VOCABULARY) == len(TOKEN_TO_ID) == 128

AGENTS = ("ava", "ben", "cy", "da", "eri", "fin", "gia", "hal")
KINDS = ("cube", "key", "orb", "box", "ball", "book", "gem", "tool", "gate", "door", "coin", "flag", "map", "ring", "lamp", "chest")
IDENTIFIERS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen")
LOCATIONS = ("garden", "hall", "tower", "cave", "river", "lab", "store", "home", "bridge", "room", "forest", "dock")
COLOURS = ("red", "blue", "green", "yellow")
ATTRIBUTES = ("small", "heavy", "light", "closed", "locked", "hidden", "new", "old", "empty", "full", "visible", "broken")
CONTAINERS = ("box", "chest")
NUMBER_TOKENS = ("none", *IDENTIFIERS)
GENERATOR_VERSION = "4.0.2"
SCHEMA_VERSION = "micro-transformer-jsonl-v4"
SPLIT_VERSION = "balanced-entity-matrix-v1"


PROFILES: dict[str, dict[str, int]] = {
    "balanced": {
        "statement_only": 4, "state_tracking": 14, "delayed_recall": 10,
        "ownership": 12, "property": 12, "conditional": 8, "belief": 7,
        "spatial": 6, "quantifier": 7, "temporal": 6, "communication": 5,
        "ability": 5, "composition": 4,
    },
    "reasoning": {
        "statement_only": 2, "state_tracking": 8, "delayed_recall": 18,
        "ownership": 8, "property": 7, "conditional": 15, "belief": 15,
        "spatial": 8, "quantifier": 5, "temporal": 5, "communication": 5,
        "ability": 4,
    },
}

DIFFICULTY = {
    "simple": {"short": (0, 1), "long": (2, 3), "transfers": (1, 2)},
    "mixed": {"short": (1, 3), "long": (4, 8), "transfers": (2, 5)},
    "hard": {"short": (3, 6), "long": (8, 16), "transfers": (5, 9)},
}


@dataclass
class Episode:
    tokens: list[str]
    answer_tokens: list[str]
    task: str
    statement_count: int
    structural_template: str
    split_key: str
    variant: str = "legacy"
    task_version: str = "legacy-v3"
    suite: str = "standard"


class World:
    def __init__(self) -> None:
        self.locations: dict[str, str] = {}
        self.owners: dict[str, str] = {}
        self.colours: dict[str, str] = {}
        self.attributes: dict[str, set[str]] = {}
        self.locked: dict[str, bool] = {}
        self.opened: dict[str, bool] = {}
        self.hidden: dict[str, bool] = {}
        self.containers: dict[str, str] = {}
        self.relations: dict[tuple[str, str], str] = {}
        self.beliefs: dict[str, dict[str, str]] = {agent: {} for agent in AGENTS}
        self.knowledge: dict[str, dict[str, str]] = {agent: {} for agent in AGENTS}
        self.events: list[tuple[str, str]] = []

    def move(self, obj: str, location: str) -> None:
        self.locations[obj] = location
        self.owners.pop(obj, None)
        self.containers.pop(obj, None)
        self.events.append(("move", obj))

    def take(self, agent: str, obj: str) -> None:
        self.owners[obj] = agent
        self.locations.pop(obj, None)
        self.containers.pop(obj, None)
        self.events.append(("take", obj))

    def drop(self, obj: str, location: str) -> None:
        self.owners.pop(obj, None)
        self.locations[obj] = location
        self.events.append(("drop", obj))

    def give(self, giver: str, obj: str, recipient: str) -> None:
        if self.owners.get(obj) != giver:
            raise ValueError("only the current owner can give an object")
        self.owners[obj] = recipient
        self.events.append(("give", obj))

    def paint(self, obj: str, colour: str) -> None:
        self.colours[obj] = colour
        self.events.append(("paint", obj))

    def set_attribute(self, obj: str, attribute: str, value: bool = True) -> None:
        values = self.attributes.setdefault(obj, set())
        values.add(attribute) if value else values.discard(attribute)

    def set_locked(self, obj: str, value: bool) -> None:
        self.locked[obj] = value
        self.set_attribute(obj, "locked", value)

    def set_open(self, obj: str, value: bool) -> None:
        self.opened[obj] = value
        self.set_attribute(obj, "closed", not value)

    def set_hidden(self, obj: str, value: bool) -> None:
        self.hidden[obj] = value
        self.set_attribute(obj, "hidden", value)
        self.set_attribute(obj, "visible", not value)

    def put_inside(self, obj: str, container: str) -> None:
        if obj == container:
            raise ValueError("an object cannot contain itself")
        cursor = container
        while cursor in self.containers:
            cursor = self.containers[cursor]
            if cursor == obj:
                raise ValueError("containment cycle detected")
        self.containers[obj] = container
        self.locations.pop(obj, None)

    def remove_from(self, obj: str, container: str, location: str) -> None:
        if self.containers.get(obj) == container:
            self.containers.pop(obj)
        self.locations[obj] = location

    def where(self, obj: str) -> str:
        return self.locations.get(obj, "unknown")

    def owner(self, obj: str) -> str:
        return self.owners.get(obj, "unknown")

    def colour(self, obj: str) -> str:
        return self.colours.get(obj, "unknown")


@dataclass
class Context:
    rng: random.Random
    split: str
    difficulty: str
    suite: str = "standard"

    def bounds(self, name: str) -> tuple[int, int]:
        return DIFFICULTY[self.difficulty][name]

    def object(self, kinds: tuple[str, ...] = KINDS, excluded: set[str] | None = None) -> tuple[str, list[str]]:
        excluded = excluded or set()
        candidates = [(key, list(tokens)) for key, tokens in entity_pool(tuple(kinds), self.split) if key not in excluded]
        if not candidates:
            raise RuntimeError(f"no {self.split} object is available for the requested kinds")
        return self.rng.choice(candidates)

    def case(self, task: str, kinds: tuple[str, ...] = KINDS) -> tuple[str, list[str], str, str]:
        del task  # Split identity is global and intentionally independent of task.
        obj, tokens = self.object(kinds)
        return obj, tokens, self.rng.choice(LOCATIONS), obj


def assigned_split(key: str) -> str:
    try:
        kind, identifier = key.split(":")
        kind_index, identifier_index = KINDS.index(kind), IDENTIFIERS.index(identifier)
    except (ValueError, AttributeError):
        raise ValueError(f"invalid entity split key: {key!r}")
    if identifier_index == kind_index or (kind_index < 10 and identifier_index == (kind_index + 1) % 16):
        return "validation"
    if identifier_index == (kind_index + 2) % 16 or (kind_index < 9 and identifier_index == (kind_index + 3) % 16):
        return "test"
    return "train"


@lru_cache(maxsize=128)
def entity_pool(kinds: tuple[str, ...], split: str):
    return tuple((f"{kind}:{identifier}", (kind, identifier)) for kind in kinds
                 for identifier in IDENTIFIERS if assigned_split(f"{kind}:{identifier}") == split)


def move(world: World, actor: str, obj: str, ot: list[str], location: str, origin: str | None = None) -> list[str]:
    world.move(obj, location)
    if origin is None:
        return [actor, "move", *ot, "to", location, "."]
    return [actor, "move", *ot, "from", origin, "to", location, "."]


def take(world: World, actor: str, obj: str, ot: list[str]) -> list[str]:
    world.take(actor, obj)
    return [actor, "take", *ot, "."]


def paint(world: World, actor: str, obj: str, ot: list[str], colour: str) -> list[str]:
    world.paint(obj, colour)
    return [actor, "paint", *ot, colour, "."]


def random_statement(world: World, ctx: Context, excluded: set[str] | None = None) -> list[str]:
    rng, excluded = ctx.rng, excluded or set()
    actor = rng.choice(AGENTS)
    obj, ot = ctx.object(excluded=excluded)
    action = rng.choice(("move", "take", "drop", "give", "open", "close", "lock", "unlock", "paint", "hide", "find", "build", "put", "remove", "swap", "follow"))
    location = rng.choice(LOCATIONS)
    if action == "move": return move(world, actor, obj, ot, location)
    if action == "take":
        initial = rng.choice(LOCATIONS); tokens = move(world, actor, obj, ot, initial)
        tokens.extend(take(world, actor, obj, ot)); return tokens
    if action == "drop":
        initial = rng.choice(tuple(place for place in LOCATIONS if place != location)); tokens = move(world, actor, obj, ot, initial)
        tokens.extend(take(world, actor, obj, ot)); world.drop(obj, location)
        tokens.extend([actor, "drop", *ot, "at", location, "."]); return tokens
    if action == "give":
        recipient = rng.choice(tuple(a for a in AGENTS if a != actor))
        tokens = move(world, actor, obj, ot, location); tokens.extend(take(world, actor, obj, ot)); world.give(actor, obj, recipient)
        tokens.extend([actor, "give", *ot, "to", recipient, "."]); return tokens
    if action in {"open", "close", "lock", "unlock"}:
        if action == "open": world.set_open(obj, True)
        elif action == "close": world.set_open(obj, False)
        elif action == "lock": world.set_locked(obj, True)
        else: world.set_locked(obj, False)
        return [actor, action, *ot, "."]
    if action == "paint": return paint(world, actor, obj, ot, rng.choice(COLOURS))
    if action == "hide":
        container, ct = ctx.object(CONTAINERS, {obj, *excluded})
        tokens = move(world, actor, obj, ot, location); world.put_inside(obj, container); world.set_hidden(obj, True)
        tokens.extend([actor, "hide", *ot, "in", *ct, "."]); return tokens
    if action == "find":
        container, ct = ctx.object(CONTAINERS, {obj, *excluded}); world.put_inside(obj, container); world.set_hidden(obj, True)
        tokens = [actor, "hide", *ot, "in", *ct, "."]
        world.set_hidden(obj, False); tokens.extend([actor, "find", *ot, "."]); return tokens
    if action == "build":
        world.move(obj, location); world.set_attribute(obj, "new")
        return [actor, "build", *ot, "at", location, "."]
    if action == "put":
        container, ct = ctx.object(CONTAINERS, {obj, *excluded})
        world.put_inside(obj, container)
        return [actor, "put", *ot, "inside", *ct, "."]
    if action == "remove":
        container, ct = ctx.object(CONTAINERS, {obj, *excluded})
        world.put_inside(obj, container); world.remove_from(obj, container, location)
        return [actor, "put", *ot, "inside", *ct, ".", actor, "remove", *ot, "from", *ct, "to", location, "."]
    if action == "swap":
        other, bt = ctx.object(excluded={obj, *excluded})
        first, second = rng.sample(LOCATIONS, 2)
        world.move(obj, first); world.move(other, second); world.move(obj, second); world.move(other, first)
        return [actor, "move", *ot, "to", first, ".", actor, "move", *bt, "to", second, ".", actor, "swap", *ot, "with", *bt, "."]
    other_agent = rng.choice(tuple(a for a in AGENTS if a != actor))
    world.locations[actor] = location
    return [actor, "follow", other_agent, "to", location, "."]


def add_distractors(tokens: list[str], world: World, ctx: Context, count: int, protected: set[str]) -> None:
    for _ in range(count):
        tokens.extend(random_statement(world, ctx, protected))


def statement_only(ctx: Context, occurrence: int) -> Episode:
    world = World(); obj, ot, location, key = ctx.case("statement_only")
    tokens = move(world, ctx.rng.choice(AGENTS), obj, ot, location)
    count = ctx.rng.randint(*ctx.bounds("long")) + 2
    add_distractors(tokens, world, ctx, count - 1, {obj})
    return Episode(tokens + ["|"], [], "statement_only", tokens.count("."), f"statement_only:{count}", key)


def state_tracking(ctx: Context, occurrence: int) -> Episode:
    world = World(); obj, ot, location, key = ctx.case("state_tracking")
    tokens = move(world, ctx.rng.choice(AGENTS), obj, ot, location)
    distractors = ctx.rng.randint(*ctx.bounds("short")); add_distractors(tokens, world, ctx, distractors, {obj})
    answer = world.where(obj); tokens.extend(["where", *ot, "?", answer, "|"])
    return Episode(tokens, [answer], "state_tracking", tokens.count("."), f"state_tracking:{distractors}", key)


def delayed_recall(ctx: Context, occurrence: int) -> Episode:
    world = World(); obj, ot, location, key = ctx.case("delayed_recall")
    tokens = move(world, ctx.rng.choice(AGENTS), obj, ot, location)
    distractors = ctx.rng.randint(*ctx.bounds("long")); add_distractors(tokens, world, ctx, distractors, {obj})
    answer = world.where(obj); tokens.extend(["where", *ot, "?", answer, "|"])
    return Episode(tokens, [answer], "delayed_recall", tokens.count("."), f"delayed_recall:{distractors}", key)


def ownership(ctx: Context, occurrence: int) -> Episode:
    world = World(); obj, ot, location, key = ctx.case("ownership")
    agents = list(AGENTS); ctx.rng.shuffle(agents); holder = agents[0]
    tokens = move(world, holder, obj, ot, location); tokens.extend(take(world, holder, obj, ot))
    transfers = ctx.rng.randint(*ctx.bounds("transfers"))
    for index in range(transfers):
        recipient = agents[(index + 1) % len(agents)]
        world.give(holder, obj, recipient); tokens.extend([holder, "give", *ot, "to", recipient, "."]); holder = recipient
        if index < transfers - 1: add_distractors(tokens, world, ctx, ctx.rng.randint(0, 2), {obj})
    answer = world.owner(obj); tokens.extend(["who", "has", *ot, "?", answer, "|"])
    return Episode(tokens, [answer], "ownership", tokens.count("."), f"ownership:{transfers}", key)


def property_query(ctx: Context, occurrence: int) -> Episode:
    world = World(); obj, ot, location, key = ctx.case("property"); actor = ctx.rng.choice(AGENTS)
    tokens = move(world, actor, obj, ot, location); variant = occurrence % 3
    if variant == 0:
        updates = ctx.rng.randint(1, 4)
        for _ in range(updates): tokens.extend(paint(world, actor, obj, ot, ctx.rng.choice(COLOURS)))
        add_distractors(tokens, world, ctx, ctx.rng.randint(*ctx.bounds("short")), {obj})
        answer = world.colour(obj); tokens.extend(["what", "colour", *ot, "?", answer, "|"]); template = f"property:colour:{updates}"
    else:
        attribute = ATTRIBUTES[(occurrence // 3) % len(ATTRIBUTES)]; world.set_attribute(obj, attribute)
        tokens.extend([*ot, "is", attribute, "."]); add_distractors(tokens, world, ctx, ctx.rng.randint(*ctx.bounds("short")), {obj})
        answer = "yes"; tokens.extend(["is", *ot, attribute, "?", answer, "|"]); template = f"property:attribute:{attribute}"
    return Episode(tokens, [answer], "property", tokens.count("."), template, key)


def conditional(ctx: Context, occurrence: int) -> Episode:
    world = World(); obj, ot, location, key = ctx.case("conditional", ("gate", "door", "box")); actor = ctx.rng.choice(AGENTS)
    variant = occurrence % 3; tokens = move(world, actor, obj, ot, location)
    add_distractors(tokens, world, ctx, ctx.rng.randint(*ctx.bounds("short")), {obj})
    if variant == 0:
        world.set_locked(obj, True); tokens.extend([actor, "lock", *ot, "."]); world.set_locked(obj, False)
        tokens.extend(["if", *ot, "locked", "then", actor, "unlock", *ot, "else", actor, "open", *ot, ".", "is", *ot, "locked", "?", "no", "|"]); answer = "no"
    elif variant == 1:
        world.set_locked(obj, False); world.set_open(obj, True)
        tokens.extend([actor, "unlock", *ot, ".", "if", *ot, "locked", "then", actor, "close", *ot, "else", actor, "open", *ot, ".", "is", *ot, "closed", "?", "no", "|"]); answer = "no"
    else:
        world.set_locked(obj, False); world.set_open(obj, True)
        tokens.extend([actor, "unlock", *ot, ".", "if", *ot, "not", "locked", "then", actor, "open", *ot, "else", actor, "unlock", *ot, ".", "can", actor, "open", *ot, "?", "yes", "|"]); answer = "yes"
    return Episode(tokens, [answer], "conditional", tokens.count("."), f"conditional:{variant}", key)


def belief(ctx: Context, occurrence: int) -> Episode:
    world = World(); obj, ot, first, key = ctx.case("belief"); mover, observer = ctx.rng.sample(AGENTS, 2)
    second = ctx.rng.choice(tuple(x for x in LOCATIONS if x != first)); tokens = move(world, mover, obj, ot, first)
    world.beliefs[observer][obj] = first; tokens.extend([observer, "sees", mover, "move", *ot, "to", first, "."])
    add_distractors(tokens, world, ctx, ctx.rng.randint(*ctx.bounds("short")), {obj}); tokens.extend(move(world, mover, obj, ot, second))
    answer = world.beliefs[observer][obj]; tokens.extend(["where", observer, "believes", *ot, "?", answer, "|"])
    return Episode(tokens, [answer], "belief", tokens.count("."), "belief:false_belief", key)


def spatial(ctx: Context, occurrence: int) -> Episode:
    world = World(); obj, ot, location, key = ctx.case("spatial"); other, bt = ctx.object(excluded={obj}); variant = occurrence % 5
    tokens = move(world, ctx.rng.choice(AGENTS), obj, ot, location)
    if variant in {0, 1}:
        relation = "behind" if variant == 0 else "beside"; world.relations[(obj, other)] = relation
        tokens.extend([*ot, relation, *bt, ".", "which", *ot, "is", relation, *bt, "?", *ot, "|"]); answer = ot
    elif variant == 2:
        world.relations[(obj, other)] = "with"
        tokens.extend([*ot, "with", *bt, ".", "does", *ot, "with", *bt, "?", "yes", "|"]); answer = ["yes"]
    else:
        container, ct = ctx.object(CONTAINERS, {obj, other}); actor = ctx.rng.choice(AGENTS); world.put_inside(obj, container)
        verb, connector = ("put", "inside") if variant == 3 else ("hide", "in")
        tokens.extend([actor, verb, *ot, connector, *ct, ".", *ct, "contains", *ot, ".", "does", *ct, "contains", *ot, "?", "yes", "|"]); answer = ["yes"]
    return Episode(tokens, answer, "spatial", tokens.count("."), f"spatial:{variant}", key)


def quantifier(ctx: Context, occurrence: int) -> Episode:
    world = World(); obj, ot, location, key = ctx.case("quantifier"); kind = ot[0]; variant = occurrence % 5
    compatible_ids = [
        identifier for identifier in IDENTIFIERS
        if assigned_split(f"{kind}:{identifier}") == ctx.split
    ]
    max_count = min(5, len(compatible_ids))
    count = ctx.rng.randint(1, max_count)
    chosen_ids = [ot[1]]
    chosen_ids.extend(ctx.rng.sample([value for value in compatible_ids if value != ot[1]], count - 1))
    tokens: list[str] = []
    for identifier in chosen_ids:
        current, ct = f"{kind}:{identifier}", [kind, identifier]
        tokens.extend(move(world, ctx.rng.choice(AGENTS), current, ct, location))
    if variant == 0:
        answer = NUMBER_TOKENS[count]; tokens.extend(["count", kind, "at", location, "?", answer, "|"])
    elif variant == 1:
        answer = "yes"; tokens.extend(["is", "some", kind, "at", location, "?", answer, "|"])
    elif variant == 2:
        answer = "yes"; other = ctx.rng.choice(tuple(x for x in LOCATIONS if x != location)); tokens.extend(["is", "none", kind, "at", other, "?", answer, "|"])
    elif variant == 3:
        answer = "yes"; tokens.extend(["is", "all", kind, "at", location, "?", answer, "|"])
    else:
        other, bt = ctx.object(excluded={obj}); colour = ctx.rng.choice(COLOURS)
        tokens.extend(paint(world, ctx.rng.choice(AGENTS), obj, ot, colour)); tokens.extend(paint(world, ctx.rng.choice(AGENTS), other, bt, colour))
        answer = "yes"; tokens.extend(["is", *ot, "same", "colour", *bt, "?", answer, "|"])
    return Episode(tokens, [answer], "quantifier", tokens.count("."), f"quantifier:{variant}:{count}", key)


def temporal(ctx: Context, occurrence: int) -> Episode:
    world = World(); obj, ot, location, key = ctx.case("temporal"); other, bt = ctx.object(excluded={obj}); actor, second_actor = ctx.rng.sample(AGENTS, 2)
    other_location = ctx.rng.choice(tuple(x for x in LOCATIONS if x != location)); variant = occurrence % 7
    if variant == 0:
        world.move(obj, location); world.move(other, other_location); tokens = [actor, "move", *ot, "to", location, "before", second_actor, "move", *bt, "to", other_location, "."]
    elif variant == 1:
        world.move(other, other_location); world.move(obj, location); tokens = [actor, "move", *ot, "to", location, "after", second_actor, "move", *bt, "to", other_location, "."]
    elif variant == 2:
        world.move(obj, location); world.move(other, other_location); tokens = [actor, "move", *ot, "to", location, "while", second_actor, "move", *bt, "to", other_location, "."]
    elif variant == 3:
        world.move(obj, location); world.move(other, other_location); tokens = [actor, "move", *ot, "to", location, "and", second_actor, "move", *bt, "to", other_location, "."]
    elif variant == 4:
        world.move(other, other_location); world.take(actor, other); world.move(obj, location)
        tokens = [actor, "move", *bt, "to", other_location, ".", actor, "take", *bt, ".", actor, "move", *ot, "to", location, "because", actor, "has", *bt, "."]
    elif variant == 5:
        world.move(obj, location); tokens = ["only", actor, "move", *ot, "to", location, "."]
    else:
        origin = ctx.rng.choice(tuple(x for x in LOCATIONS if x != location)); world.move(obj, origin); world.move(obj, location)
        tokens = [actor, "move", *ot, "from", origin, "to", location, "again", "."]
    answer = world.where(obj); tokens.extend(["where", *ot, "?", answer, "|"])
    return Episode(tokens, [answer], "temporal", tokens.count("."), f"temporal:{variant}", key)


def communication(ctx: Context, occurrence: int) -> Episode:
    world = World(); obj, ot, location, key = ctx.case("communication"); speaker, listener = ctx.rng.sample(AGENTS, 2)
    tokens = move(world, speaker, obj, ot, location); world.beliefs[listener][obj] = location; world.knowledge[listener][obj] = location
    tokens.extend([speaker, "tell", listener, *ot, "at", location, ".", listener, "knows", *ot, "at", location, "."])
    add_distractors(tokens, world, ctx, ctx.rng.randint(*ctx.bounds("short")), {obj}); answer = world.knowledge[listener][obj]
    tokens.extend(["where", listener, "knows", *ot, "?", answer, "|"])
    return Episode(tokens, [answer], "communication", tokens.count("."), "communication:tell_knows", key)


def ability(ctx: Context, occurrence: int) -> Episode:
    world = World(); obj, ot, location, key = ctx.case("ability", ("door", "gate", "box")); actor = ctx.rng.choice(AGENTS); variant = occurrence % 4
    tokens = move(world, actor, obj, ot, location); distractors = ctx.rng.randint(*ctx.bounds("short"))
    if variant == 0:
        world.set_locked(obj, True); tokens.extend([actor, "lock", *ot, "."])
        add_distractors(tokens, world, ctx, distractors, {obj}); tokens.extend(["can", actor, "open", *ot, "?", "no", "|"]); answer = "no"
    elif variant == 1:
        world.set_locked(obj, False); tokens.extend([actor, "unlock", *ot, "."])
        add_distractors(tokens, world, ctx, distractors, {obj}); tokens.extend(["can", actor, "open", *ot, "?", "yes", "|"]); answer = "yes"
    elif variant == 2:
        container, ct = ctx.object(CONTAINERS, {obj}); world.put_inside(obj, container); world.set_hidden(obj, True)
        tokens.extend([actor, "hide", *ot, "in", *ct, "."])
        add_distractors(tokens, world, ctx, distractors, {obj, container}); tokens.extend(["is", *ot, "not", "visible", "?", "yes", "|"]); answer = "yes"
    else:
        unknown, ut = ctx.object(excluded={obj}); add_distractors(tokens, world, ctx, distractors, {obj, unknown})
        tokens.extend(["where", *ut, "?", "unknown", "|"]); answer = "unknown"
    return Episode(tokens, [answer], "ability", tokens.count("."), f"ability:{variant}:{distractors}", key)


BUILDERS: dict[str, Callable[[Context, int], Episode]] = {
    "statement_only": statement_only, "state_tracking": state_tracking,
    "delayed_recall": delayed_recall, "ownership": ownership,
    "property": property_query, "conditional": conditional, "belief": belief,
    "spatial": spatial, "quantifier": quantifier, "temporal": temporal,
    "communication": communication, "ability": ability,
}


def quotas(total: int, profile: dict[str, int], seed: int = 0) -> dict[str, int]:
    if sum(profile.values()) != 100: raise ValueError("profile percentages must sum to 100")
    return dict(Counter(task_schedule(total, profile, seed)))


def task_schedule(total: int, profile: dict[str, int], seed: int) -> list[str]:
    """Return a prefix-stable schedule with exact percentages per 100 records."""
    cycle = [task for task in sorted(profile) for _ in range(profile[task])]
    if len(cycle) != 100: raise ValueError("profile percentages must sum to 100")
    cycle_rng = random.Random(f"micro-transformer-task-cycle:{seed}")
    cycle_rng.shuffle(cycle)
    return [cycle[index % len(cycle)] for index in range(total)]


def episode_rng(seed: int, split: str, difficulty: str, record_id: int, attempt: int, suite="standard") -> random.Random:
    material = f"micro-transformer-v4:{seed}:{split}:{difficulty}:{suite}:{record_id}:{attempt}".encode()
    value = int.from_bytes(hashlib.blake2b(material, digest_size=16).digest(), "big")
    return random.Random(value)


def fingerprint(tokens: list[str]) -> bytes:
    return hashlib.blake2b(" ".join(tokens).encode(), digest_size=16).digest()


def validate_tokens(tokens: list[str]) -> None:
    unknown = sorted(set(tokens) - VOCABULARY)
    if unknown: raise ValueError(f"tokens outside Micro-Transformer v1: {unknown}")
    if not tokens or tokens[-1] != "|": raise ValueError("episode does not end with |")
    if tokens.count("|") != 1: raise ValueError("episode contains more than one boundary token")


class InterpreterError(ValueError):
    """Raised when serialized Micro-Transformer text is not valid executable syntax."""


class ReferenceInterpreter:
    """A parser/interpreter independent from the generator's ``World`` object."""

    def __init__(self) -> None:
        self.locations: dict[str, str] = {}
        self.agent_locations: dict[str, str] = {}
        self.owners: dict[str, str] = {}
        self.colours: dict[str, str] = {}
        self.attributes: dict[str, set[str]] = {}
        self.locked: dict[str, bool] = {}
        self.opened: dict[str, bool] = {}
        self.hidden: dict[str, bool] = {}
        self.containers: dict[str, str] = {}
        self.relations: dict[tuple[str, str], str] = {}
        self.beliefs: dict[str, dict[str, str]] = {agent: {} for agent in AGENTS}
        self.knowledge: dict[str, dict[str, str]] = {agent: {} for agent in AGENTS}
        self.known_entities: set[str] = set()

    @staticmethod
    def entity(parts: list[str], index: int) -> str:
        if index + 1 >= len(parts) or parts[index] not in KINDS or parts[index + 1] not in IDENTIFIERS:
            raise InterpreterError(f"expected object at token {index}: {parts}")
        return f"{parts[index]}:{parts[index + 1]}"

    def remember(self, obj: str) -> None:
        self.known_entities.add(obj)

    def set_inside(self, obj: str, container: str) -> None:
        if obj == container:
            raise InterpreterError("an object cannot contain itself")
        cursor = container
        while cursor in self.containers:
            cursor = self.containers[cursor]
            if cursor == obj:
                raise InterpreterError("containment cycle detected")
        self.remember(obj); self.remember(container)
        self.containers[obj] = container
        self.locations.pop(obj, None)
        self.owners.pop(obj, None)

    def set_attribute(self, obj: str, attribute: str, value: bool) -> None:
        values = self.attributes.setdefault(obj, set())
        if value: values.add(attribute)
        else: values.discard(attribute)

    def has_attribute(self, obj: str, attribute: str) -> bool:
        if attribute == "locked": return self.locked.get(obj, False)
        if attribute == "closed": return not self.opened.get(obj, True)
        if attribute == "hidden": return self.hidden.get(obj, False)
        if attribute == "visible": return obj in self.known_entities and not self.hidden.get(obj, False)
        return attribute in self.attributes.get(obj, set())

    def condition(self, parts: list[str]) -> bool:
        obj = self.entity(parts, 0); self.remember(obj)
        if len(parts) == 3:
            return self.has_attribute(obj, parts[2])
        if len(parts) == 4 and parts[2] == "not":
            return not self.has_attribute(obj, parts[3])
        raise InterpreterError(f"invalid condition: {parts}")

    def execute(self, parts: list[str]) -> None:
        if not parts: raise InterpreterError("empty statement")
        if parts[0] == "if":
            if "then" not in parts or "else" not in parts: raise InterpreterError(f"invalid conditional: {parts}")
            then_at, else_at = parts.index("then"), parts.index("else")
            branch = parts[then_at + 1:else_at] if self.condition(parts[1:then_at]) else parts[else_at + 1:]
            self.execute(branch); return
        if parts[0] == "only":
            self.execute(parts[1:]); return
        for connector in ("before", "after", "while", "and"):
            if connector in parts:
                at = parts.index(connector); left, right = parts[:at], parts[at + 1:]
                if connector == "after": self.execute(right); self.execute(left)
                else: self.execute(left); self.execute(right)
                return
        if "because" in parts:
            self.execute(parts[:parts.index("because")]); return

        if parts[0] in AGENTS:
            actor = parts[0]
            if len(parts) < 2: raise InterpreterError(f"incomplete agent statement: {parts}")
            action = parts[1]
            if action == "move":
                obj = self.entity(parts, 2)
                if "to" not in parts: raise InterpreterError(f"move has no destination: {parts}")
                location = parts[parts.index("to") + 1]
                if location not in LOCATIONS: raise InterpreterError(f"invalid destination: {location}")
                self.remember(obj); self.locations[obj] = location
                self.owners.pop(obj, None); self.containers.pop(obj, None); return
            if action == "take":
                obj = self.entity(parts, 2); self.remember(obj); self.owners[obj] = actor
                self.locations.pop(obj, None); self.containers.pop(obj, None); return
            if action == "drop":
                obj = self.entity(parts, 2)
                if len(parts) != 6 or parts[4] != "at" or parts[5] not in LOCATIONS: raise InterpreterError(f"invalid drop: {parts}")
                self.remember(obj); self.owners.pop(obj, None); self.locations[obj] = parts[5]; return
            if action == "give":
                obj = self.entity(parts, 2)
                if len(parts) != 6 or parts[4] != "to" or parts[5] not in AGENTS: raise InterpreterError(f"invalid give: {parts}")
                if self.owners.get(obj) != actor: raise InterpreterError("only the current owner can give an object")
                self.remember(obj); self.owners[obj] = parts[5]; return
            if action in {"open", "close", "lock", "unlock", "find"}:
                obj = self.entity(parts, 2); self.remember(obj)
                if len(parts) != 4: raise InterpreterError(f"invalid {action}: {parts}")
                if action == "open": self.opened[obj] = True; self.set_attribute(obj, "closed", False)
                elif action == "close": self.opened[obj] = False; self.set_attribute(obj, "closed", True)
                elif action == "lock": self.locked[obj] = True; self.set_attribute(obj, "locked", True)
                elif action == "unlock": self.locked[obj] = False; self.set_attribute(obj, "locked", False)
                else: self.hidden[obj] = False; self.set_attribute(obj, "hidden", False); self.set_attribute(obj, "visible", True)
                return
            if action == "paint":
                obj = self.entity(parts, 2)
                if len(parts) != 5 or parts[4] not in COLOURS: raise InterpreterError(f"invalid paint: {parts}")
                self.remember(obj); self.colours[obj] = parts[4]; return
            if action == "hide":
                obj, container = self.entity(parts, 2), self.entity(parts, 5)
                if len(parts) != 7 or parts[4] != "in": raise InterpreterError(f"invalid hide: {parts}")
                self.set_inside(obj, container); self.hidden[obj] = True
                self.set_attribute(obj, "hidden", True); self.set_attribute(obj, "visible", False); return
            if action == "build":
                obj = self.entity(parts, 2)
                if len(parts) != 6 or parts[4] != "at" or parts[5] not in LOCATIONS: raise InterpreterError(f"invalid build: {parts}")
                self.remember(obj); self.locations[obj] = parts[5]; self.set_attribute(obj, "new", True); return
            if action == "put":
                obj, container = self.entity(parts, 2), self.entity(parts, 5)
                if len(parts) != 7 or parts[4] != "inside": raise InterpreterError(f"invalid put: {parts}")
                self.set_inside(obj, container); return
            if action == "remove":
                obj, container = self.entity(parts, 2), self.entity(parts, 5)
                if len(parts) != 9 or parts[4] != "from" or parts[7] != "to" or parts[8] not in LOCATIONS: raise InterpreterError(f"invalid remove: {parts}")
                self.remember(obj); self.remember(container); self.containers.pop(obj, None); self.locations[obj] = parts[8]; return
            if action == "swap":
                first, second = self.entity(parts, 2), self.entity(parts, 5)
                if len(parts) != 7 or parts[4] != "with": raise InterpreterError(f"invalid swap: {parts}")
                self.remember(first); self.remember(second)
                first_location, second_location = self.locations.get(first), self.locations.get(second)
                if first_location is None or second_location is None: raise InterpreterError("swap requires two located objects")
                self.locations[first], self.locations[second] = second_location, first_location; return
            if action == "follow":
                if len(parts) != 5 or parts[2] not in AGENTS or parts[3] != "to" or parts[4] not in LOCATIONS: raise InterpreterError(f"invalid follow: {parts}")
                self.agent_locations[actor] = parts[4]; return
            if action == "tell":
                if len(parts) != 7 or parts[2] not in AGENTS or parts[5] != "at" or parts[6] not in LOCATIONS: raise InterpreterError(f"invalid tell: {parts}")
                obj = self.entity(parts, 3); listener = parts[2]; self.remember(obj)
                self.beliefs[listener][obj] = parts[6]; self.knowledge[listener][obj] = parts[6]; return
            if action == "sees":
                if len(parts) != 8 or parts[2] not in AGENTS or parts[3] != "move" or parts[6] != "to" or parts[7] not in LOCATIONS: raise InterpreterError(f"invalid observation: {parts}")
                obj = self.entity(parts, 4); self.remember(obj); self.beliefs[actor][obj] = parts[7]; return
            if action == "knows":
                if len(parts) != 6 or parts[4] != "at" or parts[5] not in LOCATIONS: raise InterpreterError(f"invalid knowledge statement: {parts}")
                obj = self.entity(parts, 2); self.remember(obj); self.knowledge[actor][obj] = parts[5]; return
            raise InterpreterError(f"unknown action: {parts}")

        if parts[0] in KINDS:
            first = self.entity(parts, 0); self.remember(first)
            if len(parts) == 4 and parts[2] == "is" and parts[3] in ATTRIBUTES:
                self.set_attribute(first, parts[3], True)
                if parts[3] == "locked": self.locked[first] = True
                elif parts[3] == "closed": self.opened[first] = False
                elif parts[3] == "hidden": self.hidden[first] = True
                elif parts[3] == "visible": self.hidden[first] = False
                return
            if len(parts) == 5 and parts[2] in {"with", "behind", "beside"}:
                second = self.entity(parts, 3); self.remember(second); self.relations[(first, second)] = parts[2]; return
            if len(parts) == 5 and parts[2] == "contains":
                second = self.entity(parts, 3); self.set_inside(second, first); return
        raise InterpreterError(f"invalid statement: {parts}")

    def answer(self, question: list[str]) -> list[str]:
        if not question: return []
        if question[0] == "where":
            if len(question) == 3:
                return [self.locations.get(self.entity(question, 1), "unknown")]
            if len(question) == 5 and question[1] in AGENTS and question[2] in {"believes", "knows"}:
                obj = self.entity(question, 3); source = self.beliefs if question[2] == "believes" else self.knowledge
                return [source[question[1]].get(obj, "unknown")]
            raise InterpreterError(f"invalid where question: {question}")
        if question[:2] == ["who", "has"] and len(question) == 4:
            return [self.owners.get(self.entity(question, 2), "unknown")]
        if question[:2] == ["what", "colour"] and len(question) == 4:
            return [self.colours.get(self.entity(question, 2), "unknown")]
        if question[0] == "which" and len(question) == 7 and question[3] == "is":
            first, second, relation = self.entity(question, 1), self.entity(question, 5), question[4]
            return question[1:3] if self.relations.get((first, second)) == relation else ["unknown"]
        if question[0] == "count" and len(question) == 4 and question[2] == "at":
            amount = sum(obj.startswith(question[1] + ":") and location == question[3] for obj, location in self.locations.items())
            if amount >= len(NUMBER_TOKENS): raise InterpreterError("count exceeds vocabulary")
            return [NUMBER_TOKENS[amount]]
        if question[0] == "does" and len(question) == 6:
            first, second, relation = self.entity(question, 1), self.entity(question, 4), question[3]
            if relation == "contains": result = self.containers.get(second) == first
            else: result = self.relations.get((first, second)) == relation
            return ["yes" if result else "no"]
        if question[0] == "can" and len(question) == 5 and question[1] in AGENTS and question[2] == "open":
            return ["no" if self.locked.get(self.entity(question, 3), False) else "yes"]
        if question[0] == "is":
            if len(question) == 5 and question[1] in {"some", "none", "all"} and question[3] == "at":
                members = [obj for obj in self.known_entities if obj.startswith(question[2] + ":")]
                matches = [obj for obj in members if self.locations.get(obj) == question[4]]
                if question[1] == "some": result = bool(matches)
                elif question[1] == "none": result = not matches
                else: result = bool(members) and len(matches) == len(members)
                return ["yes" if result else "no"]
            if len(question) == 7 and question[3:5] == ["same", "colour"]:
                first, second = self.entity(question, 1), self.entity(question, 5)
                first_colour, second_colour = self.colours.get(first), self.colours.get(second)
                return ["yes" if first_colour is not None and first_colour == second_colour else "no"]
            if len(question) in {4, 5}:
                obj = self.entity(question, 1); negated = len(question) == 5 and question[3] == "not"
                attribute = question[4] if negated else question[3]
                if attribute not in ATTRIBUTES: raise InterpreterError(f"invalid attribute question: {question}")
                result = self.has_attribute(obj, attribute)
                if negated: result = not result
                return ["yes" if result else "no"]
        raise InterpreterError(f"invalid question: {question}")


def derive_answer(tokens: list[str]) -> list[str]:
    """Parse and execute serialized tokens using an independent reference interpreter."""
    if not tokens or tokens[-1] != "|": raise InterpreterError("episode does not end with |")
    payload = tokens[:-1]
    question: list[str] = []
    if "?" in payload:
        if payload.count("?") != 1: raise InterpreterError("episode must contain at most one question")
        question_at = payload.index("?")
        if "." in payload[:question_at]: start = len(payload[:question_at]) - 1 - payload[:question_at][::-1].index(".") + 1
        else: start = 0
        question = payload[start:question_at]
        stored = payload[question_at + 1:]
        if not stored: raise InterpreterError("question has no serialized answer")
        statements = payload[:start]
    else:
        statements = payload
    interpreter = ReferenceInterpreter(); current: list[str] = []
    for token in statements:
        if token == ".":
            interpreter.execute(current); current = []
        else: current.append(token)
    if current: raise InterpreterError(f"statement is missing a period: {current}")
    return interpreter.answer(question)


def entity_keys(tokens: list[str]) -> set[str]:
    return {
        f"{tokens[index]}:{tokens[index + 1]}"
        for index in range(len(tokens) - 1)
        if tokens[index] in KINDS and tokens[index + 1] in IDENTIFIERS
    }


def validate_record(record: dict, expected_id: int | None = None) -> list[str]:
    issues: list[str] = []
    required = {"id", "tokens", "answer_tokens", "task", "statement_count", "structural_template", "split", "split_key"}
    if not isinstance(record, dict): return ["record is not a JSON object"]
    if required - record.keys(): return [f"missing fields: {sorted(required - record.keys())}"]
    if not isinstance(record["tokens"], list) or not all(isinstance(token, str) for token in record["tokens"]): return ["tokens must be a list of strings"]
    if not isinstance(record["answer_tokens"], list) or not all(isinstance(token, str) for token in record["answer_tokens"]): return ["answer_tokens must be a list of strings"]
    if not isinstance(record["id"], int): issues.append("id must be an integer")
    if not isinstance(record["task"], str): issues.append("task must be a string")
    if not isinstance(record["split"], str): issues.append("split must be a string")
    if not isinstance(record["split_key"], str): issues.append("split_key must be a string")
    if not isinstance(record["statement_count"], int): issues.append("statement_count must be an integer")
    if expected_id is not None and record["id"] != expected_id: issues.append("non-sequential id")
    try: validate_tokens(record["tokens"])
    except ValueError as error: issues.append(str(error))
    if record.get("schema") == SCHEMA_VERSION:
        if not isinstance(record["task"], str) or not re.fullmatch(r"[a-z][a-z0-9_]*", record["task"]):
            issues.append("invalid task identifier")
        for key in ("variant", "task_version", "suite"):
            if not isinstance(record.get(key), str) or not record[key]: issues.append(f"missing {key}")
        if record.get("suite") not in {"standard", "challenge"}: issues.append("invalid suite")
    elif not isinstance(record["task"], str) or record["task"] not in BUILDERS:
        issues.append("unknown task")
    if not isinstance(record["split"], str) or record["split"] not in {"train", "validation", "test"}: issues.append("unknown split")
    entities = entity_keys(record["tokens"])
    if not isinstance(record["split_key"], str) or record["split_key"] not in entities: issues.append("primary split key is absent from episode tokens")
    try:
        if any(assigned_split(key) != record["split"] for key in entities): issues.append("episode contains an entity assigned to another split")
    except (AttributeError, ValueError) as error: issues.append(f"invalid split metadata: {error}")
    if isinstance(record["statement_count"], int) and record["statement_count"] != record["tokens"].count("."): issues.append("incorrect statement_count")
    answers, tokens = record["answer_tokens"], record["tokens"]
    if answers:
        if "?" not in tokens: issues.append("answer supplied without a question")
        if tokens[-len(answers) - 1:-1] != answers: issues.append("answer not immediately before boundary")
        try:
            derived = derive_answer(tokens)
            if derived != answers: issues.append(f"semantic answer mismatch: derived {derived}, stored {answers}")
        except (IndexError, InterpreterError, KeyError, TypeError, ValueError) as error:
            issues.append(f"interpreter error: {error}")
    elif "?" in tokens: issues.append("question has no answer")
    else:
        try: derive_answer(tokens)
        except (IndexError, InterpreterError, KeyError, TypeError, ValueError) as error: issues.append(f"interpreter error: {error}")
    return issues


class JsonlWriter:
    def __init__(self, output: Path, total: int, shard_size: int, append: bool, existing: int = 0) -> None:
        self.output, self.total, self.shard_size, self.append, self.index = output, total, shard_size, append, existing
        self.handle = None

    def shard_path(self, index: int) -> Path:
        if self.shard_size <= 0: return self.output
        shard = index // self.shard_size
        suffix = self.output.suffix or ".jsonl"; stem = self.output.name[:-len(suffix)] if self.output.name.endswith(suffix) else self.output.name
        return self.output.with_name(f"{stem}-part-{shard + 1:05d}{suffix}")

    def write(self, record: dict) -> None:
        path = self.shard_path(self.index)
        if self.handle is None or Path(self.handle.name) != path:
            if self.handle is not None: self.handle.close()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("a" if self.append and path.exists() else "w", encoding="utf-8")
        self.handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"); self.index += 1

    def close(self) -> None:
        if self.handle is not None: self.handle.close()

    def flush(self) -> None:
        if self.handle is not None:
            self.handle.flush(); os.fsync(self.handle.fileno())


def manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def state_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".state.json")


def working_output(output: Path) -> Path:
    suffix = output.suffix or ".jsonl"
    stem = output.name[:-len(suffix)] if output.name.endswith(suffix) else output.name
    return output.with_name(f".{stem}.building{suffix}")


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_hash() -> str:
    digest = hashlib.sha256()
    for name in ("generator.py", "task_registry.py", "task_families.py", "coverage.py"):
        digest.update(name.encode()); digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_registry(paths=()):
    from .task_registry import TaskRegistry
    from .task_families import register
    registry = TaskRegistry()
    register(registry, sys.modules[__name__])
    for path in paths:
        registry.load(path, sys.modules[__name__])
    return registry


def generation_configuration(args, registry, profile):
    return {
        "format": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION, "generator_sha256": source_hash(),
        "episodes": args.episodes, "seed": args.seed, "split": args.split, "split_version": SPLIT_VERSION,
        "difficulty": args.difficulty, "profile": args.profile, "profile_weights": profile, "shard_size": args.shard_size,
        "max_tokens": args.max_tokens, "minimum_token_occurrences": args.min_token_occurrences,
        "require_full_coverage": args.require_full_coverage,
        "suite": getattr(args, "suite", "standard"),
        "require_capability_coverage": getattr(args, "require_capability_coverage", False),
        "task_modules": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in registry.sources},
        "task_registry": registry.describe(),
        "disjoint_inputs": {str(Path(path).resolve()): file_hash(path) for path in args.disjoint_from},
    }


def matching_outputs(output: Path, shard_size: int) -> list[Path]:
    if shard_size <= 0: return [output] if output.exists() else []
    suffix = output.suffix or ".jsonl"; stem = output.name[:-len(suffix)] if output.name.endswith(suffix) else output.name
    return sorted(output.parent.glob(f"{stem}-part-*{suffix}"))


def all_output_variants(output: Path) -> list[Path]:
    suffix = output.suffix or ".jsonl"; stem = output.name[:-len(suffix)] if output.name.endswith(suffix) else output.name
    paths = list(output.parent.glob(f"{stem}-part-*{suffix}"))
    if output.exists(): paths.append(output)
    return sorted(set(paths))


def remove_outputs(output: Path, shard_size: int) -> None:
    del shard_size
    for path in all_output_variants(output):
        path.unlink()


def repair_incomplete_tail(paths: list[Path]) -> None:
    """Discard only an unterminated final line left by an interrupted writer."""
    if not paths: return
    path = paths[-1]
    with path.open("rb+") as stream:
        stream.seek(0, os.SEEK_END); size = stream.tell()
        if size == 0: return
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b"\n": return
        stream.seek(0); data = stream.read(); boundary = data.rfind(b"\n")
        stream.truncate(boundary + 1 if boundary >= 0 else 0)


def publish_outputs(staged_output: Path, final_output: Path, total: int, shard_size: int) -> list[Path]:
    staged = matching_outputs(staged_output, shard_size)
    if not staged: raise RuntimeError("no staged dataset files were produced")
    if shard_size <= 0:
        final_paths = [final_output]
    else:
        locator = JsonlWriter(final_output, total, shard_size, False)
        final_paths = [locator.shard_path(index * shard_size) for index in range(len(staged))]
    backup = final_output.parent / f".{final_output.name}.backup"
    if backup.exists(): shutil.rmtree(backup)
    backup.mkdir(parents=True)
    old_paths = all_output_variants(final_output)
    moved_old: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for old in old_paths:
            saved = backup / old.name; os.replace(old, saved); moved_old.append((saved, old))
        for source, destination in zip(staged, final_paths):
            destination.parent.mkdir(parents=True, exist_ok=True); os.replace(source, destination); published.append(destination)
    except BaseException:
        for destination in published:
            if destination.exists(): destination.unlink()
        for saved, destination in moved_old:
            if saved.exists(): os.replace(saved, destination)
        raise
    finally:
        if backup.exists(): shutil.rmtree(backup)
    return final_paths


def scan_records(paths: Iterable[Path]) -> tuple[int, Counter[str], set[bytes], Counter[str], Counter[str], Counter[int], int, int, list[str]]:
    count = 0; tasks: Counter[str] = Counter(); seen: set[bytes] = set(); tokens: Counter[str] = Counter(); templates: Counter[str] = Counter(); lengths: Counter[int] = Counter(); statements = 0; questions = 0; issues: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try: record = json.loads(line)
                except json.JSONDecodeError as error:
                    issues.append(f"{path}:{line_number}: invalid JSON: {error}"); continue
                issues.extend(f"{path}:{line_number}: {issue}" for issue in validate_record(record, count)); count += 1
                if not isinstance(record, dict): continue
                task = record.get("task", "unknown"); task = task if isinstance(task, str) else "unknown"
                template = record.get("structural_template", "unknown"); template = template if isinstance(template, str) else "unknown"
                episode_tokens = record.get("tokens", []); episode_tokens = episode_tokens if isinstance(episode_tokens, list) and all(isinstance(token, str) for token in episode_tokens) else []
                tasks[task] += 1; seen.add(fingerprint(episode_tokens)); tokens.update(episode_tokens)
                templates[template] += 1; lengths[len(episode_tokens)] += 1
                statement_count = record.get("statement_count", 0); statements += statement_count if isinstance(statement_count, int) else 0
                answers = record.get("answer_tokens", []); questions += int(isinstance(answers, list) and bool(answers))
    return count, tasks, seen, tokens, templates, lengths, statements, questions, issues


def load_disjoint_fingerprints(paths: Iterable[Path]) -> set[bytes]:
    values: set[bytes] = set()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                record = json.loads(line)
                episode_tokens = record.get("tokens") if isinstance(record, dict) else None
                if not isinstance(episode_tokens, list) or not all(isinstance(token, str) for token in episode_tokens):
                    raise ValueError(f"{path}:{line_number}: disjoint input has invalid tokens")
                values.add(fingerprint(episode_tokens))
    return values


def generate(args: argparse.Namespace) -> dict[str, object]:
    if args.episodes < 1: raise ValueError("--episodes must be at least 1")
    if any(getattr(args, name) < 0 for name in ("shard_size", "max_tokens", "max_duplicate_retries", "min_token_occurrences", "progress_every")):
        raise ValueError("Numeric limits cannot be negative")
    registry = task_registry(getattr(args, "task_module", ()))
    profile = json.loads(args.profile_file.read_text()) if getattr(args, "profile_file", None) else PROFILES[args.profile]
    if not isinstance(profile, dict) or not profile or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in profile.values()):
        raise ValueError("Profiles must map task names to positive integer percentages")
    if set(profile) - registry.tasks.keys(): raise ValueError(f"Unknown tasks: {sorted(set(profile) - registry.tasks.keys())}")
    planned_tasks = task_schedule(args.episodes, profile, args.seed)
    target_counts = dict(Counter(planned_tasks))
    work_output = working_output(args.output); checkpoint_path = state_path(args.output)
    final_manifest_path = args.manifest or manifest_path(args.output)
    configuration = generation_configuration(args, registry, profile)
    if args.resume:
        if not checkpoint_path.exists(): raise ValueError("cannot resume without an unfinished-run state file")
        previous = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        mismatched = {key: (previous.get(key), value) for key, value in configuration.items() if previous.get(key) != value}
        if mismatched: raise ValueError(f"resume configuration differs from unfinished run: {mismatched}")
        existing_paths = matching_outputs(work_output, args.shard_size)
        repair_incomplete_tail(existing_paths)
        existing_count, actual_tasks, seen, token_counts, template_counts, length_counts, statement_total, question_total, issues = scan_records(existing_paths)
        if issues: raise ValueError(f"cannot resume invalid data: {issues[0]}")
        expected_prefix = Counter(planned_tasks[:existing_count])
        if actual_tasks != expected_prefix: raise ValueError("unfinished data does not match the deterministic task schedule")
    else:
        final_outputs = all_output_variants(args.output)
        unfinished_outputs = all_output_variants(work_output)
        if (checkpoint_path.exists() or unfinished_outputs) and not args.overwrite:
            raise ValueError("unfinished output exists; use --resume to continue it or --overwrite to discard it")
        if final_outputs and not args.overwrite: raise ValueError("output already exists; use --overwrite")
        if args.overwrite:
            remove_outputs(work_output, args.shard_size)
            if checkpoint_path.exists(): checkpoint_path.unlink()
        atomic_write_json(checkpoint_path, {**configuration, "status": "generating", "completed_episodes": 0})
        existing_count = 0; actual_tasks = Counter(); seen = set(); token_counts = Counter(); template_counts = Counter(); length_counts = Counter(); statement_total = 0; question_total = 0
    if existing_count > args.episodes: raise ValueError("unfinished output has more episodes than requested")

    from .coverage import Coverage
    coverage = Coverage()
    if args.resume:
        for path in existing_paths:
            with path.open() as stream:
                for line in stream: coverage.add(json.loads(line))
    forbidden = load_disjoint_fingerprints(args.disjoint_from); occurrences = Counter(actual_tasks)
    writer = JsonlWriter(work_output, args.episodes, args.shard_size, args.resume, existing_count)
    started = time.monotonic(); retries = 0; length_rejections = 0; invariant_rejections = 0
    try:
        for record_id in range(existing_count, args.episodes):
            task = planned_tasks[record_id]
            for attempt in range(args.max_duplicate_retries + 1):
                suite = configuration["suite"]
                context = Context(episode_rng(args.seed, args.split, args.difficulty, record_id, attempt, suite), args.split, args.difficulty, suite)
                try:
                    episode = registry.tasks[task].builder(context, occurrences[task]); validate_tokens(episode.tokens)
                except (RuntimeError, ValueError):
                    invariant_rejections += 1; continue
                if episode.task != task or episode.variant not in registry.tasks[task].variants or episode.task_version != registry.tasks[task].version or episode.suite != suite:
                    raise ValueError(f"Task {task} returned incorrect task/version/variant metadata")
                if args.max_tokens and len(episode.tokens) > args.max_tokens:
                    length_rejections += 1; continue
                digest = fingerprint(episode.tokens)
                if digest not in seen and digest not in forbidden: break
            else: raise ValueError(f"could not generate a unique {task} episode after {args.max_duplicate_retries} retries")
            retries += attempt
            record = {"schema": SCHEMA_VERSION, "id": record_id, "tokens": episode.tokens, "answer_tokens": episode.answer_tokens, "task": episode.task, "difficulty": args.difficulty, "statement_count": episode.statement_count, "structural_template": episode.structural_template, "split": args.split, "split_key": episode.split_key,
                      "variant": episode.variant, "task_version": episode.task_version, "suite": suite}
            record_issues = validate_record(record, record_id)
            if record_issues: raise ValueError(f"generated invalid record: {record_issues[0]}")
            writer.write(record); coverage.add(record); seen.add(digest); actual_tasks[task] += 1; occurrences[task] += 1; token_counts.update(episode.tokens); template_counts[episode.structural_template] += 1; length_counts[len(episode.tokens)] += 1; statement_total += episode.statement_count; question_total += int(bool(episode.answer_tokens))
            if args.progress_every and writer.index % args.progress_every == 0:
                writer.flush(); atomic_write_json(checkpoint_path, {**configuration, "status": "generating", "completed_episodes": writer.index})
                elapsed = max(time.monotonic() - started, 0.001); print(f"generated {writer.index}/{args.episodes} episodes ({writer.index / elapsed:,.0f}/s)", file=sys.stderr)
    finally: writer.close()

    unused = sorted(VOCABULARY - set(token_counts)); under_minimum = {token: token_counts[token] for token in sorted(VOCABULARY) if token_counts[token] < args.min_token_occurrences}; coverage_ok = not unused and not under_minimum
    if args.require_full_coverage and not coverage_ok: raise ValueError(f"full vocabulary coverage failed; missing={unused}, under_minimum={under_minimum}")
    capability_coverage = coverage.report(registry, profile)
    if configuration["require_capability_coverage"] and not capability_coverage["capability_coverage_passed"]:
        raise ValueError(f"Capability coverage failed: {capability_coverage}")
    staged_paths = matching_outputs(work_output, args.shard_size)
    verified_count, _, verified_seen, _, _, _, _, _, verification_issues = scan_records(staged_paths)
    if verification_issues: raise ValueError(f"final independent validation failed: {verification_issues[0]}")
    if verified_count != args.episodes or len(verified_seen) != verified_count: raise ValueError("final dataset count or uniqueness check failed")
    total_tokens = sum(length * count for length, count in length_counts.items())
    manifest = {
        "format": SCHEMA_VERSION, "generator": "micro_transformer.data.generator", "generator_version": GENERATOR_VERSION, "generator_sha256": source_hash(), "seed": args.seed,
        "split": args.split, "split_partition": "global balanced kind:identifier matrix (205/26/25 train/validation/test entities); every entity in an episode belongs to its split", "split_version": SPLIT_VERSION,
        "difficulty": args.difficulty, "profile": args.profile, "maximum_tokens_per_episode": args.max_tokens or None,
        "configuration": configuration, "capability_coverage": capability_coverage,
        "vocabulary_version": "micro-transformer-v1", "vocabulary_size": len(VOCABULARY),
        "tokenizer": {"tokens_by_id": list(VOCABULARY_ORDER), "token_to_id": TOKEN_TO_ID, "padding": "use an external masked padding ID if batching requires padding"},
        "episode_count": writer.index, "token_count": total_tokens, "mean_tokens_per_episode": round(total_tokens / writer.index, 3),
        "planned_task_counts": target_counts, "actual_task_counts": dict(sorted(actual_tasks.items())), "statement_count": statement_total, "question_count": question_total,
        "structural_template_counts": dict(sorted(template_counts.items())), "episode_length_histogram": {str(length): count for length, count in sorted(length_counts.items())}, "token_frequency": dict(sorted(token_counts.items())),
        "vocabulary_coverage": {"observed_token_count": len(token_counts), "unobserved_tokens": unused, "minimum_required_per_token": args.min_token_occurrences, "tokens_below_minimum": under_minimum, "full_coverage_requirement_met": coverage_ok},
        "duplicate_control": {"exact_duplicates_written": 0, "rejected_candidates": retries, "over_length_candidates_rejected": length_rejections, "invariant_candidates_rejected": invariant_rejections, "fingerprint": "blake2b-128"},
        "validation": {"all_records_end_with_episode_boundary": True, "all_generated_tokens_are_in_fixed_vocabulary": True, "all_answers_verified_by_fresh_interpreter_replay": True, "separate_semantic_implementation": False, "task_quotas_met_exactly": dict(actual_tasks) == target_counts, "exact_duplicates_written": 0, "global_entity_split_assignment_enforced": True},
        "coverage_note": "Certifies quotas, token/variant coverage, measured boolean balance, exact uniqueness and disjoint entity identities; not exhaustive coverage of the unbounded episode space. Challenge mode is a stress distribution, not automatically disjoint in every structure.",
    }
    atomic_write_json(checkpoint_path, {**configuration, "status": "validated", "completed_episodes": writer.index})
    final_paths = publish_outputs(work_output, args.output, args.episodes, args.shard_size)
    manifest["outputs"] = [str(path) for path in final_paths]
    manifest["output_sha256"] = {str(path): file_hash(path) for path in final_paths}
    atomic_write_json(final_manifest_path, manifest)
    checkpoint_path.unlink()
    return {"output": manifest["outputs"], "manifest": str(final_manifest_path), "episodes": writer.index, "tokens": total_tokens}


def validate_dataset(args: argparse.Namespace) -> dict[str, object]:
    count, tasks, seen, tokens, templates, lengths, _, _, issues = scan_records(args.validate); duplicates = count - len(seen); overlap = len(seen & load_disjoint_fingerprints(args.disjoint_from))
    if duplicates: issues.append(f"{duplicates} exact duplicate token sequences")
    if overlap: issues.append(f"{overlap} sequences overlap --disjoint-from data")
    from .coverage import Coverage
    coverage = Coverage()
    for path in args.validate:
        with path.open() as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                    if not validate_record(record): coverage.add(record)
                except (ValueError, TypeError): pass
    registry = task_registry(getattr(args, "task_module", ()))
    selected = [task for task in tasks if task in registry.tasks]
    if getattr(args, "require_capability_coverage", False):
        expected = json.loads(args.profile_file.read_text()) if getattr(args, "profile_file", None) else PROFILES[args.profile]
        if set(expected) - registry.tasks.keys(): raise ValueError("Profile references unregistered tasks")
        selected = expected
    capability_coverage = coverage.report(registry, selected)
    if getattr(args, "require_capability_coverage", False):
        if set(tasks) - registry.tasks.keys(): issues.append("Load extension task modules to certify their declared variants")
        if not capability_coverage["capability_coverage_passed"]: issues.append("capability coverage failed")
    result = {"valid": not issues, "episodes": count, "tokens": sum(length * amount for length, amount in lengths.items()), "unique_sequences": len(seen), "duplicates": duplicates, "overlap_with_disjoint_inputs": overlap, "observed_vocabulary_tokens": len(tokens), "unobserved_vocabulary_tokens": sorted(VOCABULARY - set(tokens)), "task_counts": dict(sorted(tasks.items())), "template_counts": dict(sorted(templates.items())), "capability_coverage": capability_coverage, "issues": issues[:100]}
    print(json.dumps(result, indent=2, sort_keys=True)); return result


def build_parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__); mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--episodes", type=int); mode.add_argument("--validate", type=Path, nargs="+"); mode.add_argument("--list-tasks", action="store_true")
    command.add_argument("--output", type=Path); command.add_argument("--manifest", type=Path); command.add_argument("--seed", type=int, default=42)
    command.add_argument("--profile", choices=sorted(PROFILES), default="balanced"); command.add_argument("--difficulty", choices=sorted(DIFFICULTY), default="mixed"); command.add_argument("--split", choices=("train", "validation", "test"), default="train")
    command.add_argument("--shard-size", type=int, default=0); lifecycle = command.add_mutually_exclusive_group()
    lifecycle.add_argument("--resume", action="store_true"); lifecycle.add_argument("--overwrite", action="store_true")
    command.add_argument("--max-tokens", type=int, default=128, help="reject and regenerate episodes longer than this; 0 disables the limit")
    command.add_argument("--max-duplicate-retries", type=int, default=1000); command.add_argument("--min-token-occurrences", type=int, default=1); command.add_argument("--require-full-coverage", action="store_true")
    command.add_argument("--disjoint-from", type=Path, action="append", default=[]); command.add_argument("--progress-every", type=int, default=100000)
    command.add_argument("--suite", choices=("standard", "challenge"), default="standard", help="challenge stresses longer target-event chains; it is not a separate identity split")
    command.add_argument("--profile-file", type=Path, help="JSON task percentages summing to 100; can select registered extension tasks")
    command.add_argument("--task-module", type=Path, action="append", default=[], help="Trusted local Python extension exporting register(registry, api)")
    command.add_argument("--require-capability-coverage", action="store_true", help="Require every selected task variant and balanced yes/no in declared boolean variants")
    return command


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.list_tasks:
            print(json.dumps(task_registry(args.task_module).describe(), indent=2)); return 0
        if args.validate:
            result = validate_dataset(args); return 0 if result["valid"] else 1
        if args.output is None: raise ValueError("--output is required with --episodes")
        if args.shard_size < 0 or args.min_token_occurrences < 0 or args.max_tokens < 0: raise ValueError("numeric limits cannot be negative")
        result = generate(args); print(json.dumps(result, sort_keys=True)); return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
