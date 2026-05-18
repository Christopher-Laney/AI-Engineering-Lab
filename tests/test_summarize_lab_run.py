import json
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_lab_run


class SummarizeLabRunTests(unittest.TestCase):
    def test_render_markdown_handles_skipped_training(self):
        markdown = summarize_lab_run.render_markdown(
            {
                "manifest": {
                    "status": "passed",
                    "generated_at_utc": "2026-01-01T00:00:00Z",
                    "steps": [{"name": "generate_prompts", "artifacts": []}],
                },
                "prompt_eval": None,
                "bias_report": None,
                "model_summary": None,
            }
        )

        self.assertIn("Status: `passed`", markdown)
        self.assertIn("Training was skipped", markdown)

    def test_build_summary_loads_neighbor_reports(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "generated_at_utc": "2026-01-01T00:00:00Z",
                        "steps": [{"name": "evaluate_prompts", "artifacts": []}],
                    }
                ),
                encoding="utf-8",
            )
            (root / "prompt_evaluation.json").write_text(
                json.dumps({"summary": {"records": 2, "average_score": 3.5, "grades": {"strong": 2}}}),
                encoding="utf-8",
            )

            summary = summarize_lab_run.build_summary(manifest_path)

        self.assertEqual(summary["prompt_eval"]["summary"]["records"], 2)

    def test_artifact_path_finds_matching_suffix(self):
        path = summarize_lab_run.artifact_path(
            {"artifacts": [{"path": "outputs/lab_run/prompt_evaluation.json"}]},
            "prompt_evaluation.json",
        )

        self.assertEqual(path, Path("outputs/lab_run/prompt_evaluation.json"))
