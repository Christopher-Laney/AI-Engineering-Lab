import json
import tempfile
import unittest
from pathlib import Path

from scripts import validate_notebooks


class ValidateNotebooksTests(unittest.TestCase):
    def test_validate_notebook_accepts_markdown_and_code_cells(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            notebook = Path(tmp_dir) / "sample.ipynb"
            notebook.write_text(
                json.dumps(
                    {
                        "cells": [
                            {"cell_type": "markdown", "source": ["# Demo"]},
                            {"cell_type": "code", "source": ["x = 1\n", "print(x)"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = validate_notebooks.validate_notebook(notebook)

        self.assertEqual(result["code_cells"], 1)
        self.assertEqual(result["markdown_cells"], 1)

    def test_validate_notebook_rejects_invalid_code_syntax(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            notebook = Path(tmp_dir) / "broken.ipynb"
            notebook.write_text(
                json.dumps(
                    {
                        "cells": [
                            {"cell_type": "markdown", "source": ["# Demo"]},
                            {"cell_type": "code", "source": ["def broken(:\n", "    pass"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "invalid Python syntax"):
                validate_notebooks.validate_notebook(notebook)

    def test_validate_directory_checks_committed_notebooks(self):
        results = validate_notebooks.validate_directory(Path("notebooks"))

        self.assertEqual(len(results), 3)
        self.assertTrue(all(result["code_cells"] > 0 for result in results))
