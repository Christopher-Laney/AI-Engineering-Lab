#!/usr/bin/env python3
"""
Evaluate generated prompts with a deterministic lightweight rubric.

The scorer is intentionally simple and transparent. It is useful for comparing
prompt batches before any model API call is introduced.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


STRUCTURE_TERMS = ("checklist", "bullets", "sections", "outline", "matrix", "step-by-step")
EVALUATION_TERMS = ("criteria", "metrics", "assessment", "risks", "pitfalls", "mitigations")


def load_prompt_records(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Prompt input must be a JSON array.")

    records = []
    for item in payload:
        if isinstance(item, str):
            records.append({"prompt": item})
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            records.append(item)
        else:
            raise ValueError("Each prompt item must be a string or object with a string 'prompt'.")
    return records


def score_prompt(record: dict):
    prompt = record["prompt"]
    lower = prompt.lower()
    dimensions = {
        "specificity": int(any(token in lower for token in ("exactly", "include", "with constraints", "vs alternatives"))),
        "structure": int(any(token in lower for token in STRUCTURE_TERMS)),
        "audience": int(bool(record.get("audience")) or "for a " in lower or "aimed at a " in lower),
        "evaluation_ready": int(any(token in lower for token in EVALUATION_TERMS)),
    }
    total = sum(dimensions.values())
    return {
        **record,
        "rubric": dimensions,
        "score": total,
        "grade": "strong" if total >= 3 else "usable" if total >= 2 else "thin",
    }


def summarize(scored_records):
    by_grade = Counter(record["grade"] for record in scored_records)
    avg_score = round(sum(record["score"] for record in scored_records) / max(1, len(scored_records)), 3)
    return {
        "records": len(scored_records),
        "average_score": avg_score,
        "grades": dict(sorted(by_grade.items())),
        "top_prompts": [
            {
                "topic": record.get("topic"),
                "prompt": record["prompt"],
                "score": record["score"],
                "grade": record["grade"],
            }
            for record in sorted(scored_records, key=lambda row: (-row["score"], row["prompt"]))[:5]
        ],
    }


def write_markdown(path: Path, summary: dict):
    lines = [
        "# Prompt Evaluation Report",
        "",
        f"- Records: `{summary['records']}`",
        f"- Average score: `{summary['average_score']}`",
        "",
        "## Grade Distribution",
        "",
        "| Grade | Count |",
        "|---|---:|",
    ]
    for grade, count in summary["grades"].items():
        lines.append(f"| {grade} | {count} |")

    lines += ["", "## Top Prompts", ""]
    for record in summary["top_prompts"]:
        topic = record["topic"] or "Unlabeled"
        lines.append(f"- **{topic}** (`{record['score']}` / {record['grade']}): {record['prompt']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Score generated prompts with a deterministic rubric.")
    parser.add_argument("--input", required=True, help="JSON array from generate_prompts.py")
    parser.add_argument("--output-base", default="outputs/prompt_evaluation", help="Base path for JSON and Markdown outputs")
    args = parser.parse_args()

    records = load_prompt_records(Path(args.input))
    scored = [score_prompt(record) for record in records]
    summary = summarize(scored)

    base = Path(args.output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "records": scored}, indent=2),
        encoding="utf-8",
    )
    write_markdown(base.with_suffix(".md"), summary)

    print(f"Wrote JSON -> {base.with_suffix('.json')}")
    print(f"Wrote Markdown -> {base.with_suffix('.md')}")


if __name__ == "__main__":
    main()
