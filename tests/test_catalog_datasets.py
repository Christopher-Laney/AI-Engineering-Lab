import json
import tempfile
import unittest
from pathlib import Path

from scripts import catalog_datasets


class CatalogDatasetsTests(unittest.TestCase):
    def test_catalog_csv_records_rows_and_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.csv"
            path.write_text("id,text\n1,hello\n2,world\n", encoding="utf-8")

            item = catalog_datasets.catalog_csv(path)

        self.assertEqual(item["rows"], 2)
        self.assertEqual(item["columns"], ["id", "text"])
        self.assertEqual(item["format"], "csv")

    def test_catalog_sentiment_json_records_label_counts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sentiment_training.json"
            path.write_text(
                json.dumps(
                    {
                        "description": "sample",
                        "classes": {"0": "negative", "1": "positive"},
                        "text": ["bad", "good", "great"],
                        "label": [0, 1, 1],
                    }
                ),
                encoding="utf-8",
            )

            item = catalog_datasets.catalog_sentiment_json(path)

        self.assertEqual(item["rows"], 3)
        self.assertEqual(item["label_counts"], {"0": 1, "1": 2})

    def test_render_markdown_includes_hashes_and_notes(self):
        markdown = catalog_datasets.render_markdown(
            [
                {
                    "path": "datasets/sample.csv",
                    "format": "csv",
                    "rows": 1,
                    "columns": ["text"],
                    "sha256": "abc123",
                }
            ]
        )

        self.assertIn("# Dataset Catalog", markdown)
        self.assertIn("abc123", markdown)
        self.assertIn("synthetic or instructional", markdown)

    def test_catalog_check_accepts_current_committed_catalog(self):
        catalog = catalog_datasets.build_catalog(Path("datasets"))
        rendered = catalog_datasets.render_markdown(catalog)
        current = Path("docs/dataset_catalog.md").read_text(encoding="utf-8")

        self.assertEqual(current, rendered)
