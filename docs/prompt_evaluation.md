# Prompt Evaluation

`scripts/evaluate_prompts.py` provides a transparent, deterministic rubric for generated prompt batches.

## What It Scores

Each prompt receives one point for:

- specificity
- structure
- audience targeting
- evaluation readiness

Grades are:

- `strong`: score `3-4`
- `usable`: score `2`
- `thin`: score `0-1`

The rubric is intentionally simple. It is meant for quick local comparison before introducing model calls, human review, or richer evaluation systems.
The built-in prompt templates are designed to produce structured, audience-aware, evaluation-ready prompts by default, so generated batches start from a useful baseline before teams customize them.

## Example Workflow

Generate a prompt batch:

```bash
python scripts/generate_prompts.py \
  --input datasets/sample_prompts.csv \
  --output outputs/generated_prompts.json \
  --with-metadata \
  --dedupe \
  --seed 42
```

Score it:

```bash
python scripts/evaluate_prompts.py \
  --input outputs/generated_prompts.json \
  --output-base outputs/prompt_evaluation
```

Outputs:

- `outputs/prompt_evaluation.json`
- `outputs/prompt_evaluation.md`

## How To Use It

- compare batches after changing templates
- spot prompt families that are consistently thin
- keep prompt experiments reviewable before involving an LLM API

This is a heuristic quality screen, not a substitute for task-grounded evaluation against real model outputs.
