import unittest

from scripts import generate_model_card


class GenerateModelCardTests(unittest.TestCase):
    def test_render_model_card_includes_required_sections(self):
        markdown = generate_model_card.render_model_card(
            {
                "run_id": "run-123",
                "timestamp_utc": "2026-01-01T00:00:00Z",
                "data_path": "datasets/sentiment_training.json",
                "data_sha256": "abc123",
                "rows": 40,
                "class_dist": {"0": 20, "1": 20},
                "splits": {"train": 32, "val": 0, "test": 8},
                "test": {"roc_auc": 0.5, "confusion_matrix": [[2, 2], [2, 2]]},
                "artifacts": {"model_path": "outputs/sentiment_model.joblib"},
            }
        )

        self.assertIn("# Sentiment Baseline Model Card", markdown)
        self.assertIn("## Intended Use", markdown)
        self.assertIn("abc123", markdown)
        self.assertIn("## Limitations", markdown)
