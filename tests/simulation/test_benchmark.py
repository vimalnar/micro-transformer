import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "micro-world"))

from micro_world.agent_adapter import decode_action, serialize_prompt
from micro_world.benchmark import SCENARIO_IDS, ScriptedPolicy, make_scenario, replay_scenario, run_episode
from micro_transformer.data.generator import VOCABULARY


class BenchmarkScenarioTests(unittest.TestCase):
    def test_scripted_feasibility_and_replay_for_every_family(self):
        for scenario_id in SCENARIO_IDS:
            for seed in (11, 101, 1001):
                with self.subTest(scenario=scenario_id, seed=seed):
                    scenario = make_scenario(scenario_id, seed)
                    summary, decisions = run_episode(ScriptedPolicy(), scenario)
                    self.assertTrue(summary["completed"])
                    self.assertTrue(summary["replay_matches"])
                    self.assertLessEqual(len(decisions), scenario.max_steps)
                    replay = replay_scenario(summary["replay"])
                    self.assertEqual(replay.world.state.to_dict(), scenario.world.state.to_dict())

    def test_adapter_uses_only_vocabulary_and_omits_world_coordinates(self):
        scenario = make_scenario("key-door-access", 11)
        prompt = serialize_prompt(scenario.observe(), scenario.goal.kind)
        tokens = prompt.split()
        self.assertFalse(set(tokens) - VOCABULARY)
        self.assertLessEqual(len(tokens), 122)
        self.assertNotIn(str(scenario.world.state.entities["ava"].x), prompt)
        self.assertNotIn("benchmark-v1", prompt)

    def test_decoder_requires_a_complete_command(self):
        self.assertEqual(decode_action("move . |"), "forward")
        self.assertEqual(decode_action("move one . |"), "turn-left")
        self.assertIsNone(decode_action("move |"))
        self.assertIsNone(decode_action("take key . |"))

    def test_manifest_matches_runtime_scenarios(self):
        manifest = json.loads((ROOT / "benchmarks/v1/manifest.json").read_text())
        self.assertEqual(tuple(manifest["micro_world"]["scenario_families"]), SCENARIO_IDS)
        self.assertEqual(set(manifest["micro_world"]["splits"]), {"development", "validation", "final"})


if __name__ == "__main__":
    unittest.main()
