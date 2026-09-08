import sys
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[2] / "micro-world"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from micro_world.protocol import Direction, EntityKind, EntityState, TileKind, TileState
from micro_world.simulation import World


def visible_entity_ids(observation):
    return {
        entity.id
        for row in observation.cells
        for cell in row
        for entity in cell.entities
    }


class ForwardPerceptionTests(unittest.TestCase):
    def setUp(self):
        self.world = World.reset(seed=7, map_name="physics-lab")
        self.ava = self.world.state.entities["ava"]
        self.ava.x = self.ava.y = 8
        self.ava.direction = Direction.NORTH
        # Remove interior occluders and unrelated entities.
        for y in range(1, 15):
            for x in range(1, 15):
                self.world.state.tiles[y][x] = TileState(TileKind.FLOOR)
        self.world.state.entities = {"ava": self.ava}

    def test_agent_is_at_bottom_centre_of_forward_9x9_view(self):
        observation = self.world.observe("ava")
        self.assertEqual((observation.width, observation.height), (9, 9))
        self.assertEqual(observation.agent_local_position, (4, 8))
        own_cell = observation.cells[8][4]
        self.assertEqual((own_cell.world_x, own_cell.world_y), (8, 8))
        self.assertIn("ava", visible_entity_ids(observation))

    def test_view_contains_ahead_but_not_behind(self):
        self.world.state.entities["ahead"] = EntityState(
            "ahead", EntityKind.ORB, 8, 4, colour="amber"
        )
        self.world.state.entities["behind"] = EntityState(
            "behind", EntityKind.CUBE, 8, 9, colour="green"
        )
        observation = self.world.observe("ava")
        ids = visible_entity_ids(observation)
        self.assertIn("ahead", ids)
        self.assertNotIn("behind", ids)
        self.assertEqual(
            (observation.cells[4][4].world_x, observation.cells[4][4].world_y),
            (8, 4),
        )

    def test_view_rotates_with_agent_heading(self):
        self.ava.direction = Direction.EAST
        self.world.state.entities["ahead"] = EntityState(
            "ahead", EntityKind.ORB, 12, 8, colour="amber"
        )
        observation = self.world.observe("ava")
        self.assertEqual(
            (observation.cells[4][4].world_x, observation.cells[4][4].world_y),
            (12, 8),
        )
        self.assertIn("ahead", visible_entity_ids(observation))

    def test_wall_is_visible_and_occludes_cell_behind_it(self):
        self.world.state.tiles[6][8] = TileState(TileKind.WALL)
        self.world.state.entities["hidden"] = EntityState(
            "hidden", EntityKind.ORB, 8, 5, colour="amber"
        )
        observation = self.world.observe("ava")
        wall_cell = observation.cells[6][4]
        hidden_cell = observation.cells[5][4]
        self.assertTrue(wall_cell.visible)
        self.assertEqual(wall_cell.tile.kind, TileKind.WALL)
        self.assertFalse(hidden_cell.visible)
        self.assertNotIn("hidden", visible_entity_ids(observation))

    def test_night_limits_range_but_lamps_restore_visibility(self):
        self.world.state.tick = 75
        self.world.state.entities["far"] = EntityState(
            "far", EntityKind.ORB, 8, 4, colour="amber"
        )
        self.assertNotIn("far", visible_entity_ids(self.world.observe("ava")))
        self.world.state.tiles[4][8] = TileState(TileKind.LAMP, active=True)
        self.assertIn("far", visible_entity_ids(self.world.observe("ava")))


if __name__ == "__main__":
    unittest.main()

