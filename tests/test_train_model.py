import json
import tempfile
import unittest
from pathlib import Path

from scripts import train_model


class TrainModelTests(unittest.TestCase):
    def test_load_json_dataset_reads_standard_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "dataset.json"
            data_path.write_text(
                json.dumps({"text": ["good", "bad"], "label": [1, 0]}),
                encoding="utf-8",
            )

            frame = train_model.load_json_dataset(data_path)

        self.assertEqual(frame["text"].tolist(), ["good", "bad"])
        self.assertEqual(frame["label"].tolist(), [1, 0])

    def test_load_json_dataset_renames_prompt_column_for_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "dataset.jsonl"
            data_path.write_text(
                '{"prompt":"good","label":1}\n{"prompt":"bad","label":0}\n',
                encoding="utf-8",
            )

            frame = train_model.load_json_dataset(data_path)

        self.assertEqual(frame["text"].tolist(), ["good", "bad"])
        self.assertEqual(frame["label"].tolist(), [1, 0])

    def test_validate_dataset_rejects_non_binary_labels(self):
        frame = train_model.pd.DataFrame(
            {"text": ["a", "b", "c"], "label": [0, 1, 2]}
        )

        with self.assertRaisesRegex(ValueError, "binary labels 0 and 1"):
            train_model.validate_dataset(frame)

    def test_validate_dataset_requires_two_rows_per_class(self):
        frame = train_model.pd.DataFrame(
            {"text": ["good", "bad", "great"], "label": [1, 0, 1]}
        )

        with self.assertRaisesRegex(ValueError, "at least 2 rows"):
            train_model.validate_dataset(frame)

    def test_sha256_file_returns_stable_digest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "dataset.json"
            data_path.write_text('{"text":["good","bad"],"label":[1,0]}', encoding="utf-8")

            digest = train_model.sha256_file(data_path)

        self.assertEqual(
            digest,
            "2e2610caa83cc8019f9a433bdfe01c8bb8b7c8e0116a2516f834e37bb98f30f8",
        )
