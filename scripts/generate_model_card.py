#!/usr/bin/env python3
"""
Generate a lightweight model card from a training summary JSON file.
"""

import argparse
import json
from pathlib import Path


def load_summary(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def render_model_card(summary: dict):
    test = summary.get("test", {})
    splits = summary.get("splits", {})
    artifacts = summary.get("artifacts", {})
    lines = [
        "# Sentiment Baseline Model Card",
        "",
        "## Model Details",
        "",
        "- Model family: `TF-IDF + Logistic Regression`",
        f"- Run ID: `{summary.get('run_id', 'unknown')}`",
        f"- Created: `{summary.get('timestamp_utc', 'unknown')}`",
        f"- Model artifact: `{artifacts.get('model_path', 'unknown')}`",
        "",
        "## Intended Use",
        "",
        "This model is a small instructional baseline for the AI Engineering Lab sentiment workflow.",
        "Use it to verify the training pipeline, inspect saved metrics, and compare later experiments against a simple starting point.",
        "",
        "## Data",
        "",
        f"- Source: `{summary.get('data_path', 'unknown')}`",
        f"- SHA-256: `{summary.get('data_sha256', 'unknown')}`",
        f"- Rows: `{summary.get('rows', 'unknown')}`",
        f"- Class distribution: `{summary.get('class_dist', {})}`",
        f"- Split sizes: train `{splits.get('train', 0)}`, validation `{splits.get('val', 0)}`, test `{splits.get('test', 0)}`",
        "",
        "## Evaluation",
        "",
        f"- Test ROC-AUC: `{test.get('roc_auc')}`",
        f"- Confusion matrix: `{test.get('confusion_matrix')}`",
        "",
        "## Limitations",
        "",
        "- The bundled dataset is intentionally tiny and synthetic.",
        "- Metrics are useful for workflow validation, not production performance claims.",
        "- The model should not be deployed for real sentiment decisions without larger representative data and additional evaluation.",
        "",
        "## Responsible AI Notes",
        "",
        "- Treat outputs as experimental.",
        "- Review dataset provenance before replacing the sample data.",
        "- Compare future models against this baseline using consistent splits and documented run summaries.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate a model card from a sentiment training summary.")
    parser.add_argument("--summary", required=True, help="Path to sentiment_model_summary.json")
    parser.add_argument("--output", required=True, help="Markdown model-card output path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_model_card(load_summary(Path(args.summary))), encoding="utf-8")
    print(f"Model card written to {output_path}")


if __name__ == "__main__":
    main()
