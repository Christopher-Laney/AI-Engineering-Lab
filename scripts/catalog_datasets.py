#!/usr/bin/env python3
"""
Generate a lightweight datasheet-style catalog for bundled sample datasets.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def catalog_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return {
        "path": str(path),
        "format": "csv",
        "sha256": sha256_file(path),
        "rows": len(rows),
        "columns": reader.fieldnames or [],
    }


def catalog_sentiment_json(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("label", [])
    texts = payload.get("text", [])
    return {
        "path": str(path),
        "format": "json",
        "sha256": sha256_file(path),
        "rows": len(texts),
        "columns": ["text", "label"],
        "description": payload.get("description"),
        "classes": payload.get("classes", {}),
        "label_counts": {str(label): count for label, count in sorted(Counter(labels).items())},
    }


def catalog_dataset(path: Path):
    if path.suffix.lower() == ".csv":
        return catalog_csv(path)
    if path.name == "sentiment_training.json":
        return catalog_sentiment_json(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def build_catalog(dataset_dir: Path):
    paths = sorted(
        path for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".json"}
    )
    return [catalog_dataset(path) for path in paths]


def render_markdown(catalog: list[dict]):
    lines = [
        "# Dataset Catalog",
        "",
        "This catalog documents the bundled sample datasets used by the local lab workflow.",
        "",
        "| Dataset | Format | Rows | Columns | SHA-256 |",
        "|---|---|---:|---|---|",
    ]
    for item in catalog:
        columns = ", ".join(item["columns"])
        lines.append(f"| `{item['path']}` | {item['format']} | {item['rows']} | {columns} | `{item['sha256']}` |")

    lines += [
        "",
        "## Dataset Notes",
        "",
        "- All bundled datasets are synthetic or instructional samples.",
        "- They are intended for workflow demonstration and validation, not production modeling.",
        "- Hashes help reviewers confirm which dataset version produced a lab run.",
    ]

    for item in catalog:
        if item.get("label_counts"):
            lines += [
                "",
                f"### `{item['path']}`",
                "",
                f"- Description: {item.get('description') or 'Not provided'}",
                f"- Classes: `{item.get('classes')}`",
                f"- Label counts: `{item['label_counts']}`",
            ]

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate a dataset catalog for bundled lab datasets.")
    parser.add_argument("--dataset-dir", default="datasets", help="Directory containing sample datasets.")
    parser.add_argument("--output", default="docs/dataset_catalog.md", help="Markdown catalog output path.")
    args = parser.parse_args()

    catalog = build_catalog(Path(args.dataset_dir))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(catalog), encoding="utf-8")
    print(f"Dataset catalog written to {output_path}")


if __name__ == "__main__":
    main()
