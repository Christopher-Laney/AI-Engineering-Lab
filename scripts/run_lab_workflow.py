#!/usr/bin/env python3
"""
Run the no-API AI Engineering Lab workflow end to end.

This script stitches together the repository's local experiments so maintainers
can prove the sample lab still works without copying commands from the docs.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[1]


def build_steps(python_executable: str, output_dir: Path, skip_training: bool = False):
    prompt_output = output_dir / "generated_prompts.json"
    prompt_eval_base = output_dir / "prompt_evaluation"
    bias_base = output_dir / "sample_bias_report"
    model_path = output_dir / "sentiment_model.joblib"
    summary_path = output_dir / "sentiment_model_summary.json"

    steps = [
        {
            "name": "generate_prompts",
            "command": [
                python_executable,
                "scripts/generate_prompts.py",
                "--input",
                "datasets/sample_prompts.csv",
                "--output",
                str(prompt_output),
                "--with-metadata",
                "--dedupe",
                "--seed",
                "42",
            ],
            "artifacts": [prompt_output],
        },
        {
            "name": "evaluate_prompts",
            "command": [
                python_executable,
                "scripts/evaluate_prompts.py",
                "--input",
                str(prompt_output),
                "--output-base",
                str(prompt_eval_base),
            ],
            "artifacts": [prompt_eval_base.with_suffix(".json"), prompt_eval_base.with_suffix(".md")],
        },
        {
            "name": "bias_detection",
            "command": [
                python_executable,
                "scripts/bias_detection.py",
                "--data",
                "datasets/sample_bias_texts.csv",
                "--text-col",
                "text",
                "--id-col",
                "id",
                "--group-col",
                "group",
                "--output-base",
                str(bias_base),
            ],
            "artifacts": [bias_base.with_suffix(".json"), bias_base.with_suffix(".csv"), bias_base.with_suffix(".md")],
        },
    ]

    if not skip_training:
        steps.append(
            {
                "name": "train_sentiment_baseline",
                "command": [
                    python_executable,
                    "scripts/train_model.py",
                    "--data",
                    "datasets/sentiment_training.json",
                    "--model",
                    str(model_path),
                    "--summary",
                    str(summary_path),
                    "--seed",
                    "42",
                ],
                "artifacts": [model_path, summary_path],
            }
        )

    return steps


def run_step(step: dict, root: Path):
    completed = subprocess.run(
        step["command"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return {
        "name": step["name"],
        "command": step["command"],
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "artifacts": [str(path) for path in step["artifacts"]],
    }


def write_manifest(path: Path, results: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "steps": results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Run the local AI Engineering Lab workflow.")
    parser.add_argument("--output-dir", default="outputs/lab_run", help="Directory for workflow artifacts.")
    parser.add_argument("--manifest", default=None, help="Path to write the run manifest JSON.")
    parser.add_argument("--skip-training", action="store_true", help="Run prompt and bias steps without model training.")
    args = parser.parse_args()

    root = repo_root()
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.json"
    steps = build_steps(sys.executable, output_dir, skip_training=args.skip_training)

    results = []
    for step in steps:
        print(f"Running {step['name']}...")
        results.append(run_step(step, root))

    write_manifest(manifest_path, results)
    print(f"Lab workflow complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
