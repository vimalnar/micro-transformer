import sys
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[2] / "micro-world"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

try:
    from fastapi.testclient import TestClient
    from micro_world.service.api import create_app
except (ImportError, RuntimeError):
    TestClient = None
    create_app = None


@unittest.skipIf(TestClient is None, "optional web test dependencies are unavailable")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())

    def test_session_step_and_canonical_observation(self):
        response = self.client.post(
            "/sessions", json={"seed": 42, "map_name": "physics-lab"}
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        session_id = payload["session_id"]
        observation = payload["observation"]
        self.assertEqual(observation["agent_local_position"], [4, 8])
        self.assertEqual(len(observation["cells"]), 9)
        self.assertTrue(all(len(row) == 9 for row in observation["cells"]))

        stepped = self.client.post(
            f"/sessions/{session_id}/step",
            json={"action": "forward", "agent_id": "ava"},
        )
        self.assertEqual(stepped.status_code, 200)
        self.assertEqual(stepped.json()["state"]["tick"], 1)

        image = self.client.get(
            f"/sessions/{session_id}/observation/ava/frame.png"
        )
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.headers["content-type"], "image/png")
        self.assertTrue(image.content.startswith(b"\x89PNG"))

    def test_inspector_page_is_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Micro-World Inspector", response.text)
        self.assertIn("only the agent's current field of view is revealed", response.text)
        self.assertIn('id="reveal-world" type="checkbox" checked', response.text)


if __name__ == "__main__":
    unittest.main()
