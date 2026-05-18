#!/usr/bin/env python3
"""
Summarize a completed local lab run into a readable Markdown report.
"""

import argparse
import json
from pathlib import Path


def load_json_if_exists(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_step(manifest: dict, name: str):
    for step in manifest.get("steps", []):
        if step.get("name") == name:
            return step
    return None


def artifact_path(step: dict | None, suffix: str):
    if not step:
        return None
    for artifact in step.get("artifacts", []):
        path = artifact["path"] if isinstance(artifact, dict) else artifact
        if path.endswith(suffix):
            return Path(path)
    return None


def build_summary(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent

    prompt_eval = load_json_if_exists(
        artifact_path(find_step(manifest, "evaluate_prompts"), "prompt_evaluation.json")
        or base_dir / "prompt_evaluation.json"
    )
    bias_report = load_json_if_exists(
        artifact_path(find_step(manifest, "bias_detection"), "sample_bias_report.json")
        or base_dir / "sample_bias_report.json"
    )
    model_summary = load_json_if_exists(
        artifact_path(find_step(manifest, "train_sentiment_baseline"), "sentiment_model_summary.json")
        or base_dir / "sentiment_model_summary.json"
    )

    return {
        "manifest": manifest,
        "prompt_eval": prompt_eval,
        "bias_report": bias_report,
        "model_summary": model_summary,
    }


def render_markdown(summary: dict):
    manifest = summary["manifest"]
    lines = [
        "# Lab Run Summary",
        "",
        f"- Status: `{manifest.get('status', 'unknown')}`",
        f"- Generated: `{manifest.get('generated_at_utc', 'unknown')}`",
        f"- Steps: `{len(manifest.get('steps', []))}`",
        "",
        "## Workflow Steps",
        "",
        "| Step | Artifacts |",
        "|---|---:|",
    ]

    for step in manifest.get("steps", []):
        lines.append(f"| {step.get('name')} | {len(step.get('artifacts', []))} |")

    prompt_eval = summary["prompt_eval"]
    if prompt_eval:
        prompt_summary = prompt_eval["summary"]
        lines += [
            "",
            "## Prompt Evaluation",
            "",
            f"- Records: `{prompt_summary['records']}`",
            f"- Average score: `{prompt_summary['average_score']}`",
            f"- Grades: `{prompt_summary['grades']}`",
        ]

    bias_report = summary["bias_report"]
    if bias_report:
        bias_summary = bias_report["summary"]
        lines += [
            "",
            "## Bias Review",
            "",
            f"- Rows: `{bias_summary['rows']}`",
            f"- Flagged: `{bias_summary['flagged_count']}`",
            f"- Flagged rate: `{bias_summary['flagged_rate']}`",
            f"- Categories present: `{', '.join(bias_summary['categories_present'])}`",
        ]

    model_summary = summary["model_summary"]
    if model_summary:
        test = model_summary["test"]
        lines += [
            "",
            "## Sentiment Baseline",
            "",
            f"- Rows: `{model_summary['rows']}`",
            f"- Data SHA-256: `{model_summary['data_sha256']}`",
            f"- Test ROC-AUC: `{test['roc_auc']}`",
            f"- Confusion matrix: `{test['confusion_matrix']}`",
        ]
    else:
        lines += [
            "",
            "## Sentiment Baseline",
            "",
            "_Training was skipped for this lab run._",
        ]

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize a completed AI Engineering Lab run.")
    parser.add_argument("--manifest", required=True, help="Path to a lab-run manifest JSON.")
    parser.add_argument("--output", default=None, help="Markdown output path. Defaults next to manifest.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output) if args.output else manifest_path.with_name("summary.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(build_summary(manifest_path)), encoding="utf-8")
    print(f"Lab summary written to {output_path}")


if __name__ == "__main__":
    main()
