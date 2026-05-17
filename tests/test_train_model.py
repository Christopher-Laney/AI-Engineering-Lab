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
