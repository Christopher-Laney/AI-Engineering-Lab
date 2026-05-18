#!/usr/bin/env python3
"""
Validate notebook structure and Python code-cell syntax without executing cells.
"""

import argparse
import ast
import json
from pathlib import Path


def validate_notebook(path: Path):
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid notebook JSON: {exc}") from exc

    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"{path}: notebook must contain a cells list.")

    code_cells = 0
    markdown_cells = 0
    for index, cell in enumerate(cells, start=1):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        if cell_type == "code":
            code_cells += 1
            try:
                ast.parse(source)
            except SyntaxError as exc:
                raise ValueError(f"{path}: code cell {index} has invalid Python syntax: {exc}") from exc
        elif cell_type == "markdown":
            markdown_cells += 1

    if code_cells == 0:
        raise ValueError(f"{path}: notebook must contain at least one code cell.")
    if markdown_cells == 0:
        raise ValueError(f"{path}: notebook must contain at least one markdown cell.")

    return {
        "path": str(path),
        "code_cells": code_cells,
        "markdown_cells": markdown_cells,
    }


def validate_directory(path: Path):
    notebooks = sorted(path.glob("*.ipynb"))
    if not notebooks:
        raise ValueError(f"No notebooks found in {path}.")
    return [validate_notebook(notebook) for notebook in notebooks]


def main():
    parser = argparse.ArgumentParser(description="Validate notebook JSON and code-cell syntax.")
    parser.add_argument("--notebook-dir", default="notebooks", help="Directory containing .ipynb files.")
    args = parser.parse_args()

    results = validate_directory(Path(args.notebook_dir))
    for result in results:
        print(
            f"Validated {result['path']}: "
            f"{result['code_cells']} code cell(s), {result['markdown_cells']} markdown cell(s)"
        )


if __name__ == "__main__":
    main()
