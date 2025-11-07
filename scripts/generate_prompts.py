#!/usr/bin/env python3
"""
generate_prompts.py

Generate structured prompts from a CSV of topics (and optional extra columns).
- Supports built-in templates or custom templates from JSON/YAML.
- Can produce JSON array (default) or JSONL, with or without metadata objects.
- Permutations across styles, audiences, difficulty levels; dedupe + limit.
- Deterministic mode via --seed.

Input CSV (minimum):
  topic

Optional extra columns to use in templates:
  goal, domain, constraint, format, examples, audience

Examples
  python scripts/generate_prompts.py --input datasets/sample_prompts.csv \
    --output datasets/generated_prompts.json --with-metadata --styles concise bullet \
    --levels easy pro --audiences "junior dev" "executive"

  python scripts/generate_prompts.py --input datasets/sample_prompts.csv \
    --config config/prompt_templates.yaml --jsonl --limit-per-topic 20 --seed 42
"""

import argparse, csv, json, random, sys, re
from pathlib import Path
from itertools import product

# Optional YAML support
try:
    import yaml   # pip install pyyaml (optional)
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

# ------------------------- Built-in templates -------------------------
BUILTIN_TEMPLATES = [
    "Explain {topic} to a {audience} in a {style} style. Keep it {length}.",
    "Create a step-by-step checklist to implement {topic} for {audience}. Include pitfalls and mitigations.",
    "Summarize pros and cons of {topic} in exactly 5 bullets for a {audience}.",
    "Provide a practical mini-guide to {topic} with sections: Overview, Use-Cases, Risks, Metrics. Keep tone {style}.",
    "Draft a comparison: {topic} vs alternatives. Include selection criteria and decision matrix.",
    "Design a quickstart for {topic} in the {domain} domain with constraints: {constraint}.",
    "Give 3 common failure modes for {topic} and a mitigation plan for each, aimed at a {audience}.",
    "Produce an outline to teach {topic} to a {audience}. Include assessment questions."
]

DEFAULT_STYLES = ["concise", "practical", "bullet", "checklist"]
DEFAULT_AUDIENCES = ["senior engineer", "junior dev", "product manager"]
DEFAULT_LEVELS = ["easy", "intermediate", "pro"]

# ------------------------- Utilities -------------------------
PLACEHOLDER_RE = re.compile(r"{([a-zA-Z0-9_]+)}")

def load_templates(config_path: str | None):
    """Load templates from JSON or YAML; fall back to built-in."""
    if not config_path:
        return BUILTIN_TEMPLATES
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if p.suffix.lower() in (".yml", ".yaml"):
        if not HAVE_YAML:
            raise RuntimeError("YAML config requested but PyYAML not installed. Install pyyaml or use JSON.")
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
    # Accept {"templates":[...]} or a bare list
    if isinstance(data, dict) and "templates" in data:
        templates = data["templates"]
    elif isinstance(data, list):
        templates = data
    else:
        raise ValueError("Invalid config format. Use a list or {'templates':[...]}")

    # Basic validation
    if not all(isinstance(t, str) and t.strip() for t in templates):
        raise ValueError("Templates must be a non-empty list of strings.")
    return templates

def csv_rows(path: str):
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            yield {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}

def required_fields_for(template: str):
    """Return the set of {placeholders} used by a template."""
    return set(PLACEHOLDER_RE.findall(template))

def fill_template(template: str, row: dict):
    """Safely format a template with missing-field passthrough."""
    # Provide common defaults
    defaults = {
        "style": row.get("style") or "concise",
        "audience": row.get("audience") or "general reader",
        "length": row.get("length") or "short",
        "domain": row.get("domain") or "general",
        "constraint": row.get("constraint") or "none",
        "examples": row.get("examples") or "",
        "goal": row.get("goal") or "",
    }
    payload = defaults | row
    # Replace only known placeholders; leave unknowns untouched
    try:
        return template.format(**payload)
    except KeyError as e:
        missing = str(e).strip("'")
        # Replace unknown with empty string and retry
        payload[missing] = ""
        return template.format(**payload)

def build_permutations(row, styles, audiences, levels):
    """Yield dicts with combined permutations for a row."""
    row_topic = row.get("topic", "").strip()
    if not row_topic:
        return
    for style, audience, level in product(styles, audiences, levels):
        yield row | {
            "topic": row_topic,
            "style": style,
            "audience": audience,
            "level": level,
        }

def dedupe_keep_order(seq):
    seen = set()
    out = []
    for s in seq:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out

# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate prompts from topics CSV with templates and permutations.")
    ap.add_argument("--input", required=True, help="CSV with header: topic (and optional: goal, domain, constraint, format, examples, audience)")
    ap.add_argument("--output", default="datasets/generated_prompts.json", help="Output path (.json or .jsonl)")
    ap.add_argument("--config", help="JSON/YAML file with templates. Defaults to built-in templates.")
    ap.add_argument("--styles", nargs="*", default=DEFAULT_STYLES, help="Style list (e.g., concise bullet checklist)")
    ap.add_argument("--audiences", nargs="*", default=DEFAULT_AUDIENCES, help="Audience list (e.g., 'junior dev' 'exec')")
    ap.add_argument("--levels", nargs="*", default=DEFAULT_LEVELS, help="Difficulty levels (e.g., easy pro)")
    ap.add_argument("--with-metadata", action="store_true", help="Emit objects with metadata instead of plain strings.")
    ap.add_argument("--jsonl", action="store_true", help="Write JSONL lines instead of JSON array (overrides extension).")
    ap.add_argument("--limit-per-topic", type=int, default=20, help="Cap number of prompts per topic after dedupe (default 20).")
    ap.add_argument("--dedupe", action="store_true", help="Remove duplicate prompts.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle prompts before writing.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for determinism.")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    templates = load_templates(args.config)

    rows = list(csv_rows(args.input))
    if not rows:
        print("No rows found in input.", file=sys.stderr)
        sys.exit(1)

    all_outputs = []
    for row in rows:
        per_topic = []
        for combo in build_permutations(row, args.styles, args.audiences, args.levels):
            for tmpl in templates:
                txt = fill_template(tmpl, combo)
                if args.with_metadata:
                    per_topic.append({
                        "topic": combo.get("topic"),
                        "prompt": txt,
                        "style": combo.get("style"),
                        "audience": combo.get("audience"),
                        "level": combo.get("level"),
                        "source_template": tmpl
                    })
                else:
                    per_topic.append(txt)

        # Deduplicate within-topic
        if args.dedupe:
            if args.with_metadata:
                # dedupe by prompt text
                seen = set()
                deduped = []
                for obj in per_topic:
                    p = obj["prompt"]
                    if p in seen:
                        continue
                    seen.add(p)
                    deduped.append(obj)
                per_topic = deduped
            else:
                per_topic = dedupe_keep_order(per_topic)

        # Limit per topic
        if args.limit_per_topic and len(per_topic) > args.limit_per_topic:
            per_topic = per_topic[:args.limit_per_topic]

        all_outputs.extend(per_topic)

    if args.shuffle:
        random.shuffle(all_outputs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # JSONL vs JSON
    if args.jsonl or out_path.suffix.lower() == ".jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            if args.with_metadata:
                for obj in all_outputs:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                for s in all_outputs:
                    f.write(json.dumps({"prompt": s}, ensure_ascii=False) + "\n")
        print(f"Wrote {len(all_outputs)} lines → {out_path}")
    else:
        payload = all_outputs
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(all_outputs)} prompts → {out_path}")

if __name__ == "__main__":
    main()
