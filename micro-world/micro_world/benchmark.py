"""Deterministic Micro-World v1 scenarios, controls, and benchmark runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time

from micro_world.protocol import Direction, EntityKind, EntityState, TileKind, TileState, WorldConfig, WorldState
from micro_world.simulation.maps import empty_tiles
from micro_world.simulation.world import World


BENCHMARK_VERSION = "micro-world-benchmark-v1"
SCENARIO_IDS = (
    "direct-navigation",
    "detour-navigation",
    "key-door-access",
    "object-delivery",
    "pressure-plate-causality",
    "visit-and-return",
)


@dataclass(frozen=True)
class Goal:
    kind: str
    description: str
    target_position: tuple[int, int] | None = None
    target_entity: str | None = None
    target_door: tuple[int, int] | None = None
    requires_history: bool = False


class Scenario:
    def __init__(self, scenario_id: str, seed: int, world: World, goal: Goal, max_steps: int, oracle_actions):
        self.id = scenario_id
        self.seed = seed
        self.world = world
        self.goal = goal
        self.max_steps = max_steps
        self.oracle_actions = tuple(oracle_actions)
        self.start_position = world.state.entities["ava"].position
        self.visited_target = False
        self.item_acquired = False

    def observe(self):
        return self.world.observe("ava")

    def _update_history(self) -> None:
        if self.goal.kind == "visit-return":
            self.visited_target = self.visited_target or (
                self.world.state.entities["ava"].position == self.goal.target_position
            )

    @property
    def succeeded(self) -> bool:
        agent = self.world.state.entities["ava"]
        if self.goal.kind == "reach":
            return agent.position == self.goal.target_position
        if self.goal.kind == "possess":
            return self.goal.target_entity in agent.inventory
        if self.goal.kind == "deliver":
            item = self.world.state.entities[self.goal.target_entity]
            return (
                self.item_acquired
                and item.position == self.goal.target_position
                and self.goal.target_entity not in agent.inventory
            )
        if self.goal.kind in ("open-door", "activate-plate"):
            x, y = self.goal.target_door
            return self.world.state.tiles[y][x].open
        if self.goal.kind == "visit-return":
            return self.visited_target and agent.position == self.start_position
        raise ValueError(f"unsupported goal kind: {self.goal.kind}")

    @property
    def finished(self) -> bool:
        return self.succeeded or len(self.world.action_history) >= self.max_steps

    def step(self, action: str):
        result = self.world.step(action, "ava")
        self.item_acquired = self.item_acquired or any(
            event.kind == "taken" and event.details.get("item") == self.goal.target_entity
            for event in result.events
        )
        self._update_history()
        return result

    def replay_record(self) -> dict:
        return {
            "format": "micro-world-scenario-replay-v1",
            "scenario_id": self.id,
            "seed": self.seed,
            "actions": [row["action"] for row in self.world.action_history],
        }


def _base_world(seed: int, scenario_id: str, row: int) -> World:
    config = WorldConfig()
    tiles = empty_tiles(config)
    entities = {
        "ava": EntityState("ava", EntityKind.AGENT, 2, row, Direction.EAST, "teal")
    }
    return World(WorldState(seed, f"benchmark-v1/{scenario_id}", 0, config, tiles, entities))


def make_scenario(scenario_id: str, seed: int = 11) -> Scenario:
    if scenario_id not in SCENARIO_IDS:
        raise ValueError(f"unknown benchmark scenario: {scenario_id}")
    row = 2 + seed % 3
    world = _base_world(seed, scenario_id, row)
    state = world.state
    if scenario_id == "direct-navigation":
        distance = 3 + seed % 2
        target = (2 + distance, row)
        state.tiles[target[1]][target[0]] = TileState(TileKind.LAMP, active=True)
        return Scenario(scenario_id, seed, world, Goal("reach", "Reach the visible lamp.", target), 12, ["forward"] * distance)
    if scenario_id == "detour-navigation":
        target = (5, row)
        state.tiles[target[1]][target[0]] = TileState(TileKind.LAMP, active=True)
        state.tiles[row][3] = TileState(TileKind.WALL)
        oracle = ["turn-right", "forward", "turn-left", "forward", "forward", "forward", "turn-left", "forward"]
        return Scenario(scenario_id, seed, world, Goal("reach", "Reach the lamp by navigating around the wall.", target), 20, oracle)
    if scenario_id == "key-door-access":
        door = (5, row)
        target = (7, row)
        state.tiles[door[1]][door[0]] = TileState(TileKind.DOOR, locked=True)
        state.tiles[target[1]][target[0]] = TileState(TileKind.LAMP, active=True)
        state.entities["key-one"] = EntityState("key-one", EntityKind.KEY, 3, row, colour="red")
        oracle = ["take", "forward", "forward", "interact", "forward", "forward", "forward"]
        return Scenario(scenario_id, seed, world, Goal("reach", "Acquire the key, open the locked door, and reach the lamp.", target), 20, oracle)
    if scenario_id == "object-delivery":
        target = (6, row)
        state.tiles[target[1]][target[0]] = TileState(TileKind.LAMP, active=True)
        state.entities["cube-one"] = EntityState("cube-one", EntityKind.CUBE, 3, row, colour="green")
        oracle = ["take", "forward", "forward", "forward", "drop"]
        return Scenario(scenario_id, seed, world, Goal("deliver", "Deliver the cube to the lamp.", target, "cube-one"), 16, oracle)
    if scenario_id == "pressure-plate-causality":
        plate = (5, row)
        door = (7, row)
        state.tiles[plate[1]][plate[0]] = TileState(TileKind.PRESSURE_PLATE)
        state.tiles[door[1]][door[0]] = TileState(TileKind.DOOR)
        state.entities["cube-one"] = EntityState("cube-one", EntityKind.CUBE, 3, row, colour="green")
        return Scenario(scenario_id, seed, world, Goal("activate-plate", "Push the cube onto the plate to open the door.", target_door=door), 12, ["forward", "forward"])
    target = (4 + seed % 2, row)
    state.tiles[target[1]][target[0]] = TileState(TileKind.LAMP, active=True)
    distance = target[0] - 2
    oracle = ["forward"] * distance + ["turn-right", "turn-right"] + ["forward"] * distance
    return Scenario(
        scenario_id,
        seed,
        world,
        Goal("visit-return", "Visit the lamp, then return to the starting cell.", target, requires_history=True),
        18,
        oracle,
    )


def replay_scenario(record: dict) -> Scenario:
    if record.get("format") != "micro-world-scenario-replay-v1":
        raise ValueError("unsupported scenario replay")
    scenario = make_scenario(record["scenario_id"], record["seed"])
    for action in record["actions"]:
        scenario.step(action)
    return scenario


class ScriptedPolicy:
    name = "scripted-feasibility"

    def reset(self, scenario: Scenario) -> None:
        self.actions = list(scenario.oracle_actions)

    def choose(self, observation, goal) -> str:
        return self.actions.pop(0) if self.actions else "wait"


class WaitPolicy:
    name = "wait"

    def reset(self, scenario: Scenario) -> None:
        pass

    def choose(self, observation, goal) -> str:
        return "wait"


class RandomPolicy:
    name = "uniform-random"
    ACTIONS = ("forward", "turn-left", "turn-right", "interact", "take", "drop", "wait")

    def __init__(self, seed=0):
        self.seed = seed

    def reset(self, scenario: Scenario) -> None:
        self.rng = random.Random(f"{self.seed}:{scenario.id}:{scenario.seed}")

    def choose(self, observation, goal) -> str:
        return self.rng.choice(self.ACTIONS)


class ReactivePolicy:
    """A no-history local heuristic using only the permitted observation and goal."""

    name = "reactive-no-history"

    def reset(self, scenario: Scenario) -> None:
        pass

    def choose(self, observation, goal) -> str:
        cells = [cell for row in observation.cells for cell in row if cell.visible]
        targets = []
        for cell in cells:
            if goal.kind in ("reach", "deliver", "visit-return") and cell.tile and cell.tile.kind == TileKind.LAMP:
                targets.append(cell)
            if goal.kind == "possess" and any(entity.id == goal.target_entity for entity in cell.entities):
                targets.append(cell)
            if goal.kind == "open-door" and cell.tile and cell.tile.kind == TileKind.DOOR:
                targets.append(cell)
            if goal.kind == "activate-plate" and cell.tile and cell.tile.kind == TileKind.PRESSURE_PLATE:
                targets.append(cell)
        if targets:
            target = min(targets, key=lambda cell: (8 - cell.local_y, abs(cell.local_x - 4)))
            if target.local_x < 4:
                return "turn-left"
            if target.local_x > 4:
                return "turn-right"
            if target.local_y == 7:
                if goal.kind == "possess":
                    return "take"
                if goal.kind == "open-door":
                    return "interact"
            return "forward"
        ahead = observation.cells[7][4]
        if not ahead.visible or (ahead.tile and ahead.tile.blocks_movement):
            if ahead.tile and ahead.tile.kind == TileKind.DOOR:
                return "interact"
            return "turn-right"
        return "forward"


def _digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_episode(policy, scenario: Scenario) -> tuple[dict, list[dict]]:
    policy.reset(scenario)
    decisions = []
    physical_failures = 0
    parse_valid = 0
    started = time.perf_counter()
    while not scenario.finished:
        observation = scenario.observe()
        action = policy.choose(observation, scenario.goal)
        model_decision = getattr(policy, "last_decision", None)
        if model_decision is None:
            parse_valid += 1
        else:
            parse_valid += int(model_decision.decoded_action is not None)
        result = scenario.step(action)
        failures = [event.kind for event in result.events if event.kind in {
            "blocked", "interaction-failed", "door-locked", "take-failed", "drop-failed"
        }]
        physical_failures += bool(failures)
        decisions.append({
            "step": len(decisions) + 1,
            "scenario_id": scenario.id,
            "scenario_seed": scenario.seed,
            "policy": policy.name,
            "observation_sha256": _digest_json(observation.to_dict()),
            "action": action,
            "parse_valid": model_decision is None or model_decision.decoded_action is not None,
            "physical_failures": failures,
            "events": [event.to_dict() for event in result.events],
            "prompt": model_decision.prompt if model_decision else None,
            "model_completion": model_decision.completion if model_decision else None,
            "decoded_action": model_decision.decoded_action if model_decision else action,
        })
    replay = replay_scenario(scenario.replay_record())
    replay_matches = replay.world.state.to_dict() == scenario.world.state.to_dict() and replay.succeeded == scenario.succeeded
    summary = {
        "scenario_id": scenario.id,
        "scenario_seed": scenario.seed,
        "goal": asdict(scenario.goal),
        "policy": policy.name,
        "completed": scenario.succeeded,
        "steps": len(decisions),
        "max_steps": scenario.max_steps,
        "parse_valid_rate": parse_valid / len(decisions) if decisions else None,
        "physical_failure_rate": physical_failures / len(decisions) if decisions else None,
        "replay_matches": replay_matches,
        "replay": scenario.replay_record(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return summary, decisions


def run_benchmark(policies, cases, report_dir, metadata=None) -> dict:
    destination = Path(report_dir).resolve()
    if destination.exists():
        raise ValueError("Report directory already exists; choose a new path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    model_state_before = {
        policy.name: policy.harness.state_sha256()
        for policy in policies
        if hasattr(policy, "harness")
    }
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".micro-world-benchmark-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        with (stage / "decisions.jsonl").open("w", encoding="utf-8") as stream:
            for policy in policies:
                for scenario_id, seed in cases:
                    summary, decisions = run_episode(policy, make_scenario(scenario_id, seed))
                    summaries.append(summary)
                    for decision in decisions:
                        stream.write(json.dumps(decision, allow_nan=False, sort_keys=True) + "\n")
        by_policy = {}
        for policy in policies:
            rows = [row for row in summaries if row["policy"] == policy.name]
            by_policy[policy.name] = {
                "episodes": len(rows),
                "completed": sum(row["completed"] for row in rows),
                "completion_rate": sum(row["completed"] for row in rows) / len(rows),
                "mean_steps": sum(row["steps"] for row in rows) / len(rows),
                "parse_valid_rate": sum(row["parse_valid_rate"] for row in rows) / len(rows),
                "physical_failure_rate": sum(row["physical_failure_rate"] for row in rows) / len(rows),
                "all_replays_match": all(row["replay_matches"] for row in rows),
            }
        candidate = by_policy.get("frozen-language-model")
        controls = [row for name, row in by_policy.items() if name not in ("frozen-language-model", "scripted-feasibility")]
        if candidate is None:
            outcome = "not_tested"
        elif candidate["completed"] == 0:
            outcome = "negative"
        elif controls and candidate["completion_rate"] <= max(row["completion_rate"] for row in controls):
            outcome = "inconclusive"
        else:
            outcome = "positive_bounded"
        model_checks = {}
        for policy in policies:
            if not hasattr(policy, "harness"):
                continue
            after = policy.harness.state_sha256()
            model_checks[policy.name] = {
                "identity": policy.harness.identity,
                "state_sha256_before": model_state_before[policy.name],
                "state_sha256_after": after,
                "weights_unchanged": after == model_state_before[policy.name],
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in policy.harness.model.parameters()
                    if parameter.requires_grad
                ),
            }
        conformance = (
            all(row["replay_matches"] for row in summaries)
            and all(row["weights_unchanged"] for row in model_checks.values())
        )
        comparisons = {}
        if candidate is not None:
            for name, control in by_policy.items():
                if name in ("frozen-language-model", "scripted-feasibility"):
                    continue
                comparisons[f"frozen-language-model_minus_{name}"] = {
                    "completion_rate_delta": candidate["completion_rate"] - control["completion_rate"],
                    "parse_valid_rate_delta": candidate["parse_valid_rate"] - control["parse_valid_rate"],
                    "physical_failure_rate_delta": candidate["physical_failure_rate"] - control["physical_failure_rate"],
                }
        report = {
            "format": "micro-transformer-experiment-report-v1",
            "experiment_id": "REF-MW-001",
            "title": "Frozen Micro-Transformer in bounded Micro-World scenarios",
            "hypothesis": "The unmodified language checkpoint may produce executable and useful actions under a new versioned observation/action adapter.",
            "benchmark_version": BENCHMARK_VERSION,
            "metadata": metadata or {},
            "states": {
                "execution": "completed",
                "conformance": "passed" if conformance else "failed",
                "validity": "bounded_diagnostic",
                "review": "not_requested",
                "scientific_outcome": outcome,
            },
            "claim_boundary": "This tests one untrained adapter over named deterministic scenarios. It is not evidence of general agency, planning, a learned world model, or consciousness.",
            "policies": by_policy,
            "comparisons": comparisons,
            "model_checks": model_checks,
            "episodes": summaries,
            "controls": [policy.name for policy in policies if policy.name != "frozen-language-model"],
            "limitations": [
                "The frozen model was not trained on the adapter syntax.",
                "Scenario seeds vary placement but do not establish structural generalization.",
                "A successful scripted controller establishes feasibility only.",
            ],
            "elapsed_seconds": time.perf_counter() - started,
        }
        report["decisions_sha256"] = hashlib.sha256((stage / "decisions.jsonl").read_bytes()).hexdigest()
        (stage / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(destination)
    return report
