"""Configuration contracts; no ML dependencies required."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_paper", ROOT / "scripts/train_paper.py")
recipes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipes)


class RecipeTests(unittest.TestCase):
    def test_all_archived_pairs(self):
        count = 0
        for task in recipes.TASKS:
            for method in recipes.METHODS:
                try:
                    cfg = recipes.build_config(task, method, seed=10)
                except ValueError:
                    self.assertIn(task, ("ant", "walker"))
                    self.assertNotIn(method, ("prl", "happo", "happo_deep", "mappo"))
                    continue
                count += 1
                args = cfg["algo_args"]
                self.assertEqual(args["seed"]["seed"], 10)
                self.assertEqual(args["train"]["num_env_steps"], recipes.TASKS[task][2])
                if method in ("prl", "mappo_prl"):
                    self.assertTrue(args["algo"]["use_aux_loss"])
                    self.assertFalse(args["algo"].get("condition_on_pred", False))
                if method == "current_embedding":
                    self.assertEqual(args["algo"]["prl_target"], "current")
                if method == "joint_action":
                    self.assertTrue(args["algo"]["use_action_pred"])
                    self.assertFalse(args["algo"]["use_aux_loss"])
        self.assertEqual(count, 40)


if __name__ == "__main__":
    unittest.main()
