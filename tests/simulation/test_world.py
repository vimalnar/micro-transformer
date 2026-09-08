import sys
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[2] / "micro-world"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from micro_world.protocol import Direction, TileKind, TileState, WorldConfig, phase_for_tick
from micro_world.simulation import World


class WorldTests(unittest.TestCase):
    def test_replay_reconstructs_identical_state(self):
        world = World.reset(123, map_name="procedural")
        actions = ["forward", "turn-right", "forward", "wait", "turn-left"]
        for action in actions:
            world.step(action)
        replayed = World.replay(world.replay_record())
        self.assertEqual(world.state.to_dict(), replayed.state.to_dict())
        self.assertEqual(world.action_history, replayed.action_history)

    def test_same_seed_produces_same_procedural_world(self):
        first = World.reset(88, map_name="procedural").state.to_dict()
        second = World.reset(88, map_name="procedural").state.to_dict()
        third = World.reset(89, map_name="procedural").state.to_dict()
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_ice_preserves_motion_across_ticks(self):
        world = World.reset(map_name="physics-lab")
        ava = world.state.entities["ava"]
        ava.x, ava.y, ava.direction = 4, 12, Direction.EAST
        world.step("forward")
        self.assertEqual(ava.position, (5, 12))
        world.step("wait")
        self.assertEqual(ava.position, (6, 12))

    def test_conveyor_moves_occupant_one_cell(self):
        world = World.reset(map_name="physics-lab")
        ava = world.state.entities["ava"]
        ava.x, ava.y = 8, 12
        world.step("wait")
        self.assertEqual(ava.position, (9, 12))

    def test_spinner_rotates_agent(self):
        world = World.reset(map_name="physics-lab")
        ava = world.state.entities["ava"]
        ava.x, ava.y, ava.direction = 9, 11, Direction.NORTH
        world.step("wait")
        self.assertEqual(ava.direction, Direction.EAST)

    def test_pressure_plate_opens_unlocked_door(self):
        world = World.reset(map_name="physics-lab")
        door = world.state.tiles[8][8]
        self.assertFalse(door.open)
        ava = world.state.entities["ava"]
        ava.x, ava.y = 3, 13
        world.step("wait")
        self.assertTrue(door.open)

    def test_water_spreads_at_most_one_cell_per_tick(self):
        world = World.reset(
            map_name="physics-lab", configuration=WorldConfig(water_spreads=True)
        )
        before = sum(
            tile.kind == TileKind.WATER for row in world.state.tiles for tile in row
        )
        result = world.step("wait")
        after = sum(
            tile.kind == TileKind.WATER for row in world.state.tiles for tile in row
        )
        self.assertEqual(after, before + 1)
        self.assertEqual(sum(event.kind == "water-spread" for event in result.events), 1)

    def test_water_is_static_in_the_default_world(self):
        world = World.reset(map_name="physics-lab")
        before = sum(
            tile.kind == TileKind.WATER for row in world.state.tiles for tile in row
        )
        events = []
        for _ in range(20):
            events.extend(world.step("wait").events)
        after = sum(
            tile.kind == TileKind.WATER for row in world.state.tiles for tile in row
        )
        self.assertEqual(after, before)
        self.assertFalse(any(event.kind == "water-spread" for event in events))

    def test_water_can_pass_through_an_open_door(self):
        world = World.reset(
            map_name="physics-lab", configuration=WorldConfig(water_spreads=True)
        )
        for row in world.state.tiles:
            for tile in row:
                if tile.kind == TileKind.WATER:
                    tile.kind = TileKind.FLOOR
        world.state.tiles[7][8] = TileState(TileKind.WATER)
        world.state.tiles[6][8] = TileState(TileKind.WALL)
        world.state.tiles[7][7] = TileState(TileKind.WALL)
        world.state.tiles[7][9] = TileState(TileKind.WALL)
        world.state.tiles[8][8].open = True
        world.step("wait")
        self.assertEqual(world.state.tiles[9][8].kind, TileKind.WATER)

    def test_closed_door_blocks_and_interact_opens_it(self):
        world = World.reset(map_name="physics-lab")
        ava = world.state.entities["ava"]
        ava.x, ava.y, ava.direction = 8, 9, Direction.NORTH
        world.step("forward")
        self.assertEqual(ava.position, (8, 9))
        world.step("interact")
        self.assertTrue(world.state.tiles[8][8].open)
        world.step("forward")
        self.assertEqual(ava.position, (8, 8))

    def test_locked_door_requires_carried_key(self):
        world = World.reset(map_name="physics-lab")
        ava = world.state.entities["ava"]
        locked_door = world.state.tiles[4][11]
        ava.x, ava.y, ava.direction = 11, 5, Direction.NORTH
        world.step("interact")
        self.assertTrue(locked_door.locked)
        self.assertFalse(locked_door.open)

        key = world.state.entities["key-one"]
        key.x = key.y = -1
        ava.inventory = [key.id]
        world.step("interact")
        self.assertFalse(locked_door.locked)
        self.assertTrue(locked_door.open)

    def test_day_cycle_boundaries(self):
        self.assertEqual(phase_for_tick(0).value, "day")
        self.assertEqual(phase_for_tick(60).value, "dusk")
        self.assertEqual(phase_for_tick(70).value, "night")
        self.assertEqual(phase_for_tick(90).value, "dawn")
        self.assertEqual(phase_for_tick(100).value, "day")


if __name__ == "__main__":
    unittest.main()
