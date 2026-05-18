import json
import tempfile
import unittest
from pathlib import Path

from scripts import run_lab_workflow


class RunLabWorkflowTests(unittest.TestCase):
    def test_build_steps_includes_expected_artifacts(self):
        steps = run_lab_workflow.build_steps("python", Path("outputs/lab_run"))

        self.assertEqual([step["name"] for step in steps], [
            "generate_prompts",
            "evaluate_prompts",
            "bias_detection",
            "train_sentiment_baseline",
        ])
        self.assertIn(Path("outputs/lab_run/generated_prompts.json"), steps[0]["artifacts"])

    def test_build_steps_can_skip_training(self):
        steps = run_lab_workflow.build_steps("python", Path("outputs/lab_run"), skip_training=True)

        self.assertNotIn("train_sentiment_baseline", [step["name"] for step in steps])

    def test_write_manifest_records_steps(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"
            run_lab_workflow.write_manifest(
                manifest_path,
                [{"name": "generate_prompts", "artifacts": ["outputs/generated_prompts.json"]}],
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["steps"][0]["name"], "generate_prompts")
        self.assertIn("generated_at_utc", manifest)
