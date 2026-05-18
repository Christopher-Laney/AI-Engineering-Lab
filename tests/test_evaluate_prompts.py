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
        self.assertEqual(scored["recommendations"], [])

    def test_score_prompt_recommends_missing_dimensions(self):
        scored = evaluate_prompts.score_prompt({"prompt": "Explain caching."})

        self.assertEqual(scored["grade"], "thin")
        self.assertEqual(len(scored["recommendations"]), 4)
        self.assertIn("Name the target audience", scored["recommendations"][2])

    def test_summarize_reports_grade_distribution(self):
        summary = evaluate_prompts.summarize(
            [
                {
                    "prompt": "a",
                    "score": 4,
                    "grade": "strong",
                    "rubric": {"specificity": 1, "structure": 1, "audience": 1, "evaluation_ready": 1},
                    "recommendations": [],
                },
                {
                    "prompt": "b",
                    "score": 2,
                    "grade": "usable",
                    "rubric": {"specificity": 1, "structure": 1, "audience": 0, "evaluation_ready": 0},
                    "recommendations": ["Name the target audience."],
                },
                {
                    "prompt": "c",
                    "score": 1,
                    "grade": "thin",
                    "rubric": {"specificity": 1, "structure": 0, "audience": 0, "evaluation_ready": 0},
                    "recommendations": ["Ask for a clear output structure."],
                },
            ]
        )

        self.assertEqual(summary["records"], 3)
        self.assertEqual(summary["average_score"], 2.333)
        self.assertEqual(summary["grades"], {"strong": 1, "thin": 1, "usable": 1})
        self.assertEqual(summary["rubric_coverage"]["specificity"], 1.0)
        self.assertEqual(summary["rubric_coverage"]["evaluation_ready"], 0.333)
        self.assertEqual(summary["needs_attention"][0]["prompt"], "c")

    def test_write_markdown_includes_recommendations(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "report.md"
            summary = evaluate_prompts.summarize(
                [
                    evaluate_prompts.score_prompt({"prompt": "Explain caching.", "topic": "Caching"}),
                    evaluate_prompts.score_prompt(
                        {
                            "prompt": "Create a checklist for a junior dev. Include metrics.",
                            "topic": "Caching",
                        }
                    ),
                ]
            )

            evaluate_prompts.write_markdown(report_path, summary)
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("## Rubric Coverage", report)
        self.assertIn("## Needs Attention", report)
        self.assertIn("Name the target audience", report)
