import json
import tempfile
import unittest
from pathlib import Path

from scripts import evaluate_prompts


class EvaluatePromptsTests(unittest.TestCase):
    def test_load_prompt_records_accepts_strings_and_metadata_objects(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "prompts.json"
            prompt_path.write_text(
                json.dumps(["Explain caching.", {"prompt": "Use exactly 5 bullets.", "topic": "Caching"}]),
                encoding="utf-8",
            )

            records = evaluate_prompts.load_prompt_records(prompt_path)

        self.assertEqual(records[0]["prompt"], "Explain caching.")
        self.assertEqual(records[1]["topic"], "Caching")

    def test_score_prompt_marks_structured_audience_targeted_prompts_strong(self):
        scored = evaluate_prompts.score_prompt(
            {
                "prompt": "Create a step-by-step checklist. Include metrics for a junior dev.",
                "audience": "junior dev",
            }
        )

        self.assertEqual(scored["score"], 4)
        self.assertEqual(scored["grade"], "strong")

    def test_summarize_reports_grade_distribution(self):
        summary = evaluate_prompts.summarize(
            [
                {"prompt": "a", "score": 4, "grade": "strong"},
                {"prompt": "b", "score": 2, "grade": "usable"},
                {"prompt": "c", "score": 1, "grade": "thin"},
            ]
        )

        self.assertEqual(summary["records"], 3)
        self.assertEqual(summary["average_score"], 2.333)
        self.assertEqual(summary["grades"], {"strong": 1, "thin": 1, "usable": 1})
