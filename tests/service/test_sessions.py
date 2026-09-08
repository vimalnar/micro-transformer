import sys
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[2] / "micro-world"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from micro_world.service import SessionManager


class SessionManagerTests(unittest.TestCase):
    def test_sessions_are_independent(self):
        manager = SessionManager()
        first = manager.create(seed=1)
        second = manager.create(seed=1)
        manager.step(first.id, "turn-right")
        self.assertEqual(first.world.state.tick, 1)
        self.assertEqual(second.world.state.tick, 0)

    def test_reset_replaces_world_and_preserves_session_identity(self):
        manager = SessionManager()
        session = manager.create(seed=1)
        manager.step(session.id, "wait")
        reset = manager.reset(session.id, seed=9, map_name="procedural")
        self.assertEqual(reset.id, session.id)
        self.assertEqual(reset.world.state.seed, 9)
        self.assertEqual(reset.world.state.tick, 0)


if __name__ == "__main__":
    unittest.main()
