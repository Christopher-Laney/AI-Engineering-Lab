# Sentiment Baseline

The repository includes a small classical baseline in `scripts/train_model.py`:

- TF-IDF text features
- Logistic Regression classifier
- stratified train/test splitting
- JSON summary output with metrics and artifact paths

## Purpose

This baseline is intentionally simple. It is useful for demonstrating:

- how to load a text classification dataset
- how to record a reproducible training run
- how to compare later experiments against a known starting point

It is not intended to represent production-grade sentiment modeling or state-of-the-art performance.

## Dataset Contract

The input dataset must contain:

- `text`: free-text examples
- `label`: binary labels using `0` and `1`

The training script validates that both classes are present and that each class has at least two rows before it attempts stratified splitting.

The sample dataset in `datasets/sentiment_training.json` is intentionally tiny and synthetic. Its role is to exercise the workflow, not to produce a strong model.

## Reproducibility

Each summary JSON records:

- a generated `run_id`
- `timestamp_utc`
- the source `data_path`
- `data_sha256`
- class distribution and split sizes
- selected parameters
- test metrics and confusion matrix

Generate a lightweight model card from a completed training summary:

```bash
python scripts/generate_model_card.py \
  --summary outputs/sentiment_model_summary.json \
  --output outputs/sentiment_model_card.md
```

Run the baseline with a fixed seed:

```bash
python scripts/train_model.py \
  --data datasets/sentiment_training.json \
  --model outputs/sentiment_model.joblib \
  --summary outputs/sentiment_model_summary.json \
  --seed 42
```

## Interpreting Results

Because the bundled dataset is deliberately small, metrics may look weak or unstable. Treat the output as a workflow demonstration:

- confirm the training path runs
- inspect the saved summary
- generate a model card for review
- use the baseline as a comparison point for richer datasets or later models
