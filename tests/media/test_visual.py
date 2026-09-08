import hashlib
import sys
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[2] / "micro-world"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from micro_world.media import CANONICAL_SIZE, render_png, render_rgb
from micro_world.simulation import World


class VisualRendererTests(unittest.TestCase):
    def test_canonical_rgb_dimensions_and_repeatability(self):
        observation = World.reset(42).observe("ava")
        first = render_rgb(observation)
        second = render_rgb(observation)
        self.assertEqual(len(first), CANONICAL_SIZE * CANONICAL_SIZE * 3)
        self.assertEqual(hashlib.sha256(first).digest(), hashlib.sha256(second).digest())

    def test_png_is_validly_framed_and_repeatable(self):
        observation = World.reset(42).observe("ava")
        first = render_png(observation)
        second = render_png(observation)
        self.assertTrue(first.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(first.endswith(b"IEND\xaeB`\x82"))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

