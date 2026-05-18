# Guided Lab Walkthrough

This walkthrough gives the repository one compact, no-API path from prompt generation to responsible-AI review to classical model training.

## Run Everything

Use the runner when you want to execute the whole local lab:

```bash
python scripts/run_lab_workflow.py --output-dir outputs/lab_run
```

Expected result:

- generated prompt batch
- prompt evaluation report
- bias review report
- sentiment baseline model and summary
- `outputs/lab_run/manifest.json`

The manifest records each step, captured command output, and whether every expected artifact was created.

Generate a readable summary from the completed run:

```bash
python scripts/summarize_lab_run.py \
  --manifest outputs/lab_run/manifest.json \
  --output outputs/lab_run/summary.md
```

## 1. Generate Prompt Variants

Create structured prompt examples from the sample topic list:

```bash
python scripts/generate_prompts.py \
  --input datasets/sample_prompts.csv \
  --output outputs/generated_prompts.json \
  --with-metadata \
  --dedupe \
  --seed 42
```

Expected result:

- `outputs/generated_prompts.json`
- `20` prompt records per topic by default after the per-topic cap

## 2. Evaluate Prompt Quality

Score the generated prompt batch before moving to downstream experiments:

```bash
python scripts/evaluate_prompts.py \
  --input outputs/generated_prompts.json \
  --output-base outputs/prompt_evaluation
```

Expected result:

- `outputs/prompt_evaluation.json`
- `outputs/prompt_evaluation.md`

## 3. Run A Bias Review

Use the sanitized sample review text to exercise the lexical bias detector:

```bash
python scripts/bias_detection.py \
  --data datasets/sample_bias_texts.csv \
  --text-col text \
  --id-col id \
  --group-col group \
  --output-base outputs/sample_bias_report
```

Expected result:

- `outputs/sample_bias_report.json`
- `outputs/sample_bias_report.csv`
- `outputs/sample_bias_report.md`
- flagged terms across age, gender, religion, disability, and nationality examples

The sample rows are intentionally synthetic and compact. They are meant to demonstrate the review workflow, not to stand in for a production fairness dataset.

## 4. Train The Sentiment Baseline

Train the sample TF-IDF + Logistic Regression classifier:

```bash
python scripts/train_model.py \
  --data datasets/sentiment_training.json \
  --model outputs/sentiment_model.joblib \
  --summary outputs/sentiment_model_summary.json \
  --seed 42
```

Expected result:

- `outputs/sentiment_model.joblib`
- `outputs/sentiment_model_summary.json`

## 5. Validate The Repo

Run the lightweight regression suite before changing scripts:

```bash
pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

## What This Demonstrates

- prompt generation with deterministic examples
- prompt comparison with a transparent deterministic rubric
- responsible-AI review with traceable flagged rows and group metrics
- a reproducible baseline training run with saved artifacts
- a small local validation loop suitable for pull requests

The bundled sample datasets are documented in `docs/dataset_catalog.md`.
Check that the catalog matches the current dataset files with:

```bash
python scripts/catalog_datasets.py \
  --dataset-dir datasets \
  --output docs/dataset_catalog.md \
  --check
```
