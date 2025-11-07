#!/usr/bin/env python3
"""
bias_detection.py
Lightweight lexical bias signal detector with group-wise metrics and rich reports.

Features
- Built-in lexicon with categories (gender, religion, age, etc.) + weights.
- Optional custom lexicon (JSON/YAML) and allow/deny lists.
- Category-weighted score per row, hit extraction with context window.
- Group metrics (flag rate by subgroup), top flagged terms.
- Outputs: JSON (detailed), CSV (flagged rows), Markdown (summary).
- Configurable thresholds, columns, and case normalization.

Examples
  python bias_detection.py --data datasets/sample_texts.csv --text-col text \
    --group-col group --output-base outputs/bias_report

  python bias_detection.py --data data.csv --text-col note \
    --min-score 1.0 --custom-lexicon config/lexicon.json
"""

import argparse, json, re, sys, os
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd

# ------------------------- Built-in lexicon -------------------------
# Each term maps to (category, weight). You can override/extend via --custom-lexicon.
BUILTIN_TERMS = {
    # gender / sex
    "pregnant": ("gender", 1.0),
    "female": ("gender", 0.5),
    "male": ("gender", 0.5),
    "she": ("gender", 0.25),
    "he": ("gender", 0.25),

    # age
    "age 62": ("age", 1.0),
    "elderly": ("age", 0.75),
    "too young": ("age", 0.75),

    # religion
    "religion": ("religion", 0.75),
    "christian": ("religion", 1.0),
    "muslim": ("religion", 1.0),
    "jewish": ("religion", 1.0),
    "hindu": ("religion", 1.0),
    "atheist": ("religion", 0.5),

    # race/ethnicity (placeholder generic terms; adapt per policy)
    "race": ("race", 0.75),
    "ethnicity": ("race", 0.75),

    # other sensitive
    "disability": ("disability", 1.0),
    "pregnancy": ("gender", 1.0),
    "nationality": ("nationality", 0.75),
}

# ------------------------- Helpers -------------------------
def load_custom_lexicon(path: str):
    """
    Load custom lexicon from JSON or YAML.
    Supported JSON formats:
      {"terms": [{"pattern":"pregnant","category":"gender","weight":1.0}, ...]}
      or shorthand: {"pregnant":{"category":"gender","weight":1.0}, ...}
    YAML uses the same shape (requires PyYAML if you use YAML).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Custom lexicon not found: {path}")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # optional
        except ImportError as e:
            raise RuntimeError("YAML lexicon requested but PyYAML not installed. Use JSON or install pyyaml.") from e
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    else:
        data = json.loads(p.read_text(encoding="utf-8"))

    merged = {}
    if isinstance(data, dict) and "terms" in data and isinstance(data["terms"], list):
        for item in data["terms"]:
            pat = item.get("pattern") or item.get("term")
            if not pat: continue
            merged[pat] = (item.get("category","uncat"), float(item.get("weight",1.0)))
    elif isinstance(data, dict):
        for pat, spec in data.items():
            if isinstance(spec, dict):
                merged[pat] = (spec.get("category","uncat"), float(spec.get("weight",1.0)))
    else:
        raise ValueError("Unsupported custom lexicon format.")
    return merged

def compile_patterns(terms_dict, case_insensitive=True):
    """Compile regex patterns with word-boundary-ish defaults."""
    flags = re.IGNORECASE if case_insensitive else 0
    # naive word boundary for phrases: use re.escape and allow spaces
    compiled = []
    for pat, (cat, w) in terms_dict.items():
        esc = re.escape(pat)
        # allow partials when term contains space; otherwise require word boundaries
        if " " in pat:
            rx = re.compile(rf"{esc}", flags)
        else:
            rx = re.compile(rf"\b{esc}\b", flags)
        compiled.append((pat, cat, w, rx))
    return compiled

def extract_hits(text, patterns, whitelist=None, window=36):
    """
    Return list of hits: [{term, category, weight, start, end, context}]
    Whitelist terms are ignored (case-insensitive exact substring match).
    """
    hits = []
    if not isinstance(text, str): 
        return hits
    t = text
    low = t.lower()
    if whitelist:
        for wl in whitelist:
            low = low.replace(wl.lower(), " ")  # blunt removal of whitelisted substrings

    for term, cat, w, rx in patterns:
        for m in rx.finditer(low):
            s, e = m.start(), m.end()
            c_start = max(0, s - window)
            c_end = min(len(t), e + window)
            ctx = t[c_start:c_end]
            hits.append({
                "term": term,
                "category": cat,
                "weight": w,
                "start": int(s),
                "end": int(e),
                "context": ctx
            })
    return hits

def score_row(hits, category_caps=None):
    """
    Sum weights across hits; optionally cap by category to avoid runaway scores.
    category_caps = {"gender": 2.0, "religion": 2.0, ...}
    """
    total = 0.0
    per_cat = defaultdict(float)
    for h in hits:
        per_cat[h["category"]] += h["weight"]
    if category_caps:
        for cat, val in per_cat.items():
            total += min(val, float(category_caps.get(cat, val)))
    else:
        total = sum(per_cat.values())
    return float(total), dict(per_cat)

def safe_mkdir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Lexical bias signal detector with reporting.")
    ap.add_argument("--data", required=True, help="CSV path with at least a text column.")
    ap.add_argument("--text-col", default="text", help="Column containing free text. Default: text")
    ap.add_argument("--id-col", default=None, help="Optional ID column for traceability.")
    ap.add_argument("--group-col", default=None, help="Optional group column for subgroup metrics.")
    ap.add_argument("--custom-lexicon", default=None, help="JSON/YAML lexicon to extend/override built-ins.")
    ap.add_argument("--allowlist", default=None, help="Path to newline-delimited allowlist terms to ignore.")
    ap.add_argument("--denylist", default=None, help="Path to newline-delimited denylist terms to force-flag.")
    ap.add_argument("--min-score", type=float, default=0.5, help="Minimum total score to count a row as flagged. Default: 0.5")
    ap.add_argument("--case-insensitive", action="store_true", help="Case-insensitive matching (default on).")
    ap.add_argument("--case-sensitive", dest="case_insensitive", action="store_false", help="Case-sensitive matching.")
    ap.set_defaults(case_insensitive=True)
    ap.add_argument("--context-window", type=int, default=36, help="Characters around a hit to include in context.")
    ap.add_argument("--output-base", default="outputs/bias_report", help="Base path for outputs (without extension).")
    ap.add_argument("--json-only", action="store_true", help="Only write JSON report (skip CSV/Markdown).")
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    if args.text_col not in df.columns:
        raise SystemExit(f"Missing --text-col '{args.text_col}' in CSV.")
    text_series = df[args.text_col].astype(str)

    # Load allow/deny lists
    allowlist = []
    if args.allowlist and Path(args.allowlist).exists():
        allowlist = [ln.strip() for ln in Path(args.allowlist).read_text(encoding="utf-8").splitlines() if ln.strip()]
    denylist = []
    if args.denylist and Path(args.denylist).exists():
        denylist = [ln.strip() for ln in Path(args.denylist).read_text(encoding="utf-8").splitlines() if ln.strip()]

    # Build lexicon
    terms = dict(BUILTIN_TERMS)
    if args.custom_lexicon:
        custom = load_custom_lexicon(args.custom_lexicon)
        terms.update(custom)

    patterns = compile_patterns(terms, case_insensitive=args.case_insensitive)

    # Category caps (optional safety to prevent runaway scores)
    category_caps = {
        "gender": 3.0, "religion": 3.0, "age": 3.0, "race": 3.0,
        "disability": 3.0, "nationality": 3.0, "uncat": 2.0
    }

    # Process rows
    flagged_rows = []
    all_term_counts = Counter()
    rows_output = []
    for idx, text in enumerate(text_series):
        hits = extract_hits(text, patterns, whitelist=allowlist, window=args.context_window)
        # Force-flag denylist terms (adds weight if present)
        for d in denylist:
            if d and d.lower() in text.lower():
                hits.append({"term": d, "category":"denylist", "weight": 1.0, "start": -1, "end": -1, "context": ""})
        score, per_cat = score_row(hits, category_caps=category_caps)

        row_obj = {
            "index": int(idx),
            "id": df[args.id_col].iloc[idx] if args.id_col and args.id_col in df.columns else None,
            "text": text,
            "score": score,
            "per_category": per_cat,
            "hits": hits
        }

        # Collect counters
        for h in hits:
            all_term_counts[h["term"]] += 1

        rows_output.append(row_obj)
        if score >= args.min_score:
            flagged_rows.append(row_obj)

    # Group metrics (if provided)
    group_stats = []
    if args.group_col and args.group_col in df.columns:
        g = df[args.group_col]
        total_by_group = g.value_counts(dropna=False).to_dict()
        flagged_by_group = Counter()
        for r in flagged_rows:
            grp_val = df[args.group_col].iloc[r["index"]]
            flagged_by_group[grp_val] += 1
        for grp, total in total_by_group.items():
            f = int(flagged_by_group.get(grp, 0))
            rate = float(f) / float(total) if total else 0.0
            group_stats.append({
                "group": None if pd.isna(grp) else grp,
                "total": int(total),
                "flagged": int(f),
                "flag_rate": round(rate, 4)
            })

    # Build summary
    summary = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "source_csv": str(Path(args.data).resolve()),
        "rows": int(len(df)),
        "text_col": args.text_col,
        "id_col": args.id_col,
        "group_col": args.group_col,
        "min_score": args.min_score,
        "case_insensitive": args.case_insensitive,
        "context_window": args.context_window,
        "flagged_count": len(flagged_rows),
        "flagged_rate": round((len(flagged_rows) / max(1, len(df))), 4),
        "top_terms": [{"term": t, "count": c} for t, c in all_term_counts.most_common(20)],
        "group_stats": group_stats,
        "categories_present": sorted({h["category"] for r in rows_output for h in r["hits"]}),
    }

    # Write outputs
    base = Path(args.output_base)
    safe_mkdir(base.with_suffix(".json"))

    # JSON (detailed)
    json_payload = {
        "summary": summary,
        "rows": rows_output
    }
    base.with_suffix(".json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    print(f"Wrote JSON → {base.with_suffix('.json')}")

    if not args.json_only:
        # CSV (flagged rows only)
        flagged_csv_rows = []
        for r in flagged_rows:
            flagged_csv_rows.append({
                "index": r["index"],
                "id": r["id"],
                "score": r["score"],
                "per_category": json.dumps(r["per_category"]),
                "text": r["text"]
            })
        flagged_df = pd.DataFrame(flagged_csv_rows)
        flagged_df.to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8")
        print(f"Wrote CSV → {base.with_suffix('.csv')}")

        # Markdown summary
        md_lines = [
            f"# Bias Report",
            "",
            f"- Generated: `{summary['generated_at_utc']}`",
            f"- Source: `{summary['source_csv']}`",
            f"- Rows: `{summary['rows']}`  |  Flagged: `{summary['flagged_count']}`  |  Rate: `{summary['flagged_rate']}`",
            f"- Min score: `{summary['min_score']}`  |  Case-insensitive: `{summary['case_insensitive']}`",
            "",
            "## Top Terms",
            "",
        ]
        if summary["top_terms"]:
            for t in summary["top_terms"]:
                md_lines.append(f"- {t['term']}: {t['count']}")
        else:
            md_lines.append("_No terms flagged._")

        if group_stats:
            md_lines += ["", "## Group Metrics", "", "| Group | Total | Flagged | Flag Rate |", "|---|---:|---:|---:|"]
            for g in group_stats:
                md_lines.append(f"| {g['group']} | {g['total']} | {g['flagged']} | {g['flag_rate']:.4f} |")

        base.with_suffix(".md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print(f"Wrote Markdown → {base.with_suffix('.md')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
