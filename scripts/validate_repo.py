#!/usr/bin/env python3
"""
Run the repository's local validation checks in the same order as CI.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[1]


def build_checks(python_executable: str, include_lab: bool = True):
    checks = [
        {
            "name": "unit_tests",
            "command": [python_executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        },
        {
            "name": "dataset_catalog",
            "command": [
                python_executable,
                "scripts/catalog_datasets.py",
                "--dataset-dir",
                "datasets",
                "--output",
                "docs/dataset_catalog.md",
                "--check",
            ],
        },
        {
            "name": "notebooks",
            "command": [
                python_executable,
                "scripts/validate_notebooks.py",
                "--notebook-dir",
                "notebooks",
            ],
        },
    ]

    if include_lab:
        checks.append(
            {
                "name": "lab_smoke",
                "command": [
                    python_executable,
                    "scripts/run_lab_workflow.py",
                    "--output-dir",
                    "outputs/ci_lab_run",
                    "--skip-training",
                ],
            }
        )

    return checks


def run_check(check: dict, root: Path):
    print(f"Running {check['name']}...", flush=True)
    subprocess.run(check["command"], cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run AI Engineering Lab validation checks.")
    parser.add_argument("--skip-lab", action="store_true", help="Skip the integrated lab smoke test.")
    args = parser.parse_args()

    root = repo_root()
    for check in build_checks(sys.executable, include_lab=not args.skip_lab):
        run_check(check, root)

    print("Repository validation passed.")


if __name__ == "__main__":
    main()
