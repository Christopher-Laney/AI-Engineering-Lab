import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts import run_lab_workflow


class RunLabWorkflowTests(unittest.TestCase):
    def test_build_steps_includes_expected_artifacts(self):
        steps = run_lab_workflow.build_steps("python", Path("outputs/lab_run"))

        self.assertEqual([step["name"] for step in steps], [
            "generate_prompts",
            "evaluate_prompts",
            "bias_detection",
            "train_sentiment_baseline",
            "generate_model_card",
        ])
        self.assertIn(Path("outputs/lab_run/generated_prompts.json"), steps[0]["artifacts"])

    def test_build_steps_can_skip_training(self):
        steps = run_lab_workflow.build_steps("python", Path("outputs/lab_run"), skip_training=True)

        self.assertNotIn("train_sentiment_baseline", [step["name"] for step in steps])
        self.assertNotIn("generate_model_card", [step["name"] for step in steps])

    def test_write_manifest_records_steps(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"
            run_lab_workflow.write_manifest(
                manifest_path,
                [{"name": "generate_prompts", "artifacts": ["outputs/generated_prompts.json"]}],
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["steps"][0]["name"], "generate_prompts")
        self.assertEqual(manifest["status"], "passed")
        self.assertIn("generated_at_utc", manifest)

    def test_validate_artifacts_reports_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact = Path(tmp_dir) / "artifact.json"
            artifact.write_text("{}", encoding="utf-8")

            artifacts = run_lab_workflow.validate_artifacts([artifact])

        self.assertEqual(artifacts[0]["path"], str(artifact))
        self.assertTrue(artifacts[0]["exists"])
        self.assertGreater(artifacts[0]["bytes"], 0)

    def test_validate_artifacts_fails_for_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing.json"

            with self.assertRaisesRegex(FileNotFoundError, "Expected artifact"):
                run_lab_workflow.validate_artifacts([missing])

    def test_build_plan_serializes_commands_and_artifacts(self):
        steps = run_lab_workflow.build_steps("python", Path("outputs/lab_run"), skip_training=True)

        plan = run_lab_workflow.build_plan(steps)

        self.assertEqual(plan[0]["name"], "generate_prompts")
        self.assertEqual(plan[0]["command"][0], "python")
        self.assertIn("outputs/lab_run/generated_prompts.json", plan[0]["artifacts"])

    def test_print_plan_lists_commands_and_artifacts(self):
        plan = [
            {
                "name": "generate_prompts",
                "command": ["python", "scripts/generate_prompts.py"],
                "artifacts": ["outputs/lab_run/generated_prompts.json"],
            }
        ]

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            run_lab_workflow.print_plan(plan)

        output = stdout.getvalue()
        self.assertIn("Lab workflow dry run", output)
        self.assertIn("command: python scripts/generate_prompts.py", output)
        self.assertIn("outputs/lab_run/generated_prompts.json", output)
