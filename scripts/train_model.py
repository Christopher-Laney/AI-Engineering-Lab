#!/usr/bin/env python3
"""
train_model.py

TF-IDF + Logistic Regression sentiment classifier with:
- Stratified train/val/test split
- Optional small hyperparameter search
- Class weighting for imbalance
- Metrics: accuracy, precision/recall/F1, ROC-AUC, confusion matrix
- Saved artifacts: model (.joblib) and summary (.json)

Input:
  JSON with keys: text (list[str]), label (list[int])
  (Also works with JSONL via --jsonl)

Examples:
  python scripts/train_model.py --data datasets/sentiment_training.json \
    --model outputs/sentiment_model.joblib --summary outputs/sentiment_model_summary.json

  python scripts/train_model.py --data datasets/sentiment_training.json \
    --search --ngram-max 2 --min-df 2 --class-weight balanced --seed 42
"""

import argparse, json, os, sys, uuid
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import joblib


def load_json_dataset(path: Path, jsonl: bool = False):
    if jsonl or path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(rows)
        if "prompt" in df.columns and "label" in df.columns and "text" not in df.columns:
            df = df.rename(columns={"prompt": "text"})
        return df[["text", "label"]]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "text" in payload and "label" in payload:
        return pd.DataFrame({"text": payload["text"], "label": payload["label"]})
    raise ValueError("Unsupported JSON format. Expect keys: 'text' and 'label'.")


def main(data_path, model_path, summary_path, test_size, val_size, seed, ngram_max,
         min_df, max_features, class_weight, C, penalty, solver, lowercase, stop_words,
         search, n_jobs):
    rng = np.random.RandomState(seed)

    data_path = Path(data_path)
    if not data_path.exists():
        sys.exit(f"Data not found: {data_path}")

    # Load
    df = load_json_dataset(data_path, jsonl=False)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    y = df["label"].astype(int).values
    X = df["text"].values

    # Split (train/val/test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=seed, stratify=y
    )
    if val_size > 0:
        rel_val = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=1 - rel_val, random_state=seed, stratify=y_tmp
        )
    else:
        X_val, y_val = np.array([]), np.array([])
        X_test, y_test = X_tmp, y_tmp

    # Pipeline
    vec = TfidfVectorizer(
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_features=None if max_features <= 0 else max_features,
        lowercase=lowercase,
        stop_words=stop_words if stop_words != "none" else None
    )
    lr = LogisticRegression(
        max_iter=2000,
        class_weight=None if class_weight == "none" else class_weight,
        C=C,
        penalty=penalty,
        solver=solver
    )
    pipe = Pipeline([("tfidf", vec), ("lr", lr)])

    # Optional small search (overrides some params)
    if search:
        param_grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2, 3],
            "lr__C": [0.5, 1.0, 2.0],
        }
        searcher = GridSearchCV(
            pipe, param_grid, scoring="f1", cv=5, n_jobs=n_jobs, verbose=0
        )
        searcher.fit(X_train, y_train)
        best = searcher.best_estimator_
        best_params = searcher.best_params_
    else:
        best = pipe.fit(X_train, y_train)
        best_params = {
            "tfidf__ngram_range": (1, ngram_max),
            "tfidf__min_df": min_df,
            "lr__C": C,
            "lr__penalty": penalty,
            "lr__solver": solver,
            "lr__class_weight": class_weight,
        }

    # Validate (if any)
    val_report_text = None
    val_auc = None
    if val_size > 0 and len(np.unique(y_val)) > 1:
        val_pred = best.predict(X_val)
        val_report_text = classification_report(y_val, val_pred, digits=3)
        if hasattr(best, "predict_proba"):
            val_proba = best.predict_proba(X_val)[:, 1]
            val_auc = float(roc_auc_score(y_val, val_proba))

    # Test
    test_pred = best.predict(X_test)
    test_report_dict = classification_report(y_test, test_pred, output_dict=True, digits=3)
    test_report_text = classification_report(y_test, test_pred, digits=3)
    cm = confusion_matrix(y_test, test_pred).tolist()

    test_auc = None
    if hasattr(best, "predict_proba") and len(np.unique(y_test)) == 2:
        test_proba = best.predict_proba(X_test)[:, 1]
        test_auc = float(roc_auc_score(y_test, test_proba))

    # Save model
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, model_path)

    # Summary
    run_id = str(uuid.uuid4())
    summary = {
        "run_id": run_id,
        "timestamp_utc": pd.Timestamp.utcnow().isoformat() + "Z",
        "data_path": str(data_path),
        "rows": int(len(df)),
        "class_dist": {int(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()},
        "splits": {
            "train": int(len(X_train)),
            "val": int(len(X_val)) if val_size > 0 else 0,
            "test": int(len(X_test)),
            "test_size": test_size,
            "val_size": val_size
        },
        "best_params": best_params,
        "validation": {
            "report_text": val_report_text,
            "roc_auc": val_auc
        } if val_size > 0 else None,
        "test": {
            "report_dict": test_report_dict,
            "report_text": test_report_text,
            "confusion_matrix": cm,
            "roc_auc": test_auc
        },
        "artifacts": {
            "model_path": str(model_path)
        }
    }

    summary_path = Path(summary_path) if summary_path else model_path.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Model saved →", model_path)
    print("Summary saved →", summary_path)
    if val_report_text:
        print("\nValidation report:\n", val_report_text)
    print("\nTest report:\n", test_report_text)
    if test_auc is not None:
        print("Test ROC-AUC:", round(test_auc, 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a TF-IDF + Logistic Regression sentiment model.")
    ap.add_argument("--data", required=True, help="Path to JSON with fields: text,label")
    ap.add_argument("--model", default="outputs/sentiment_model.joblib")
    ap.add_argument("--summary", default="outputs/sentiment_model_summary.json")

    # Splits & seed
    ap.add_argument("--test-size", type=float, default=0.2, help="Proportion for test split (default 0.2)")
    ap.add_argument("--val-size", type=float, default=0.0, help="Proportion for validation split (default 0.0)")
    ap.add_argument("--seed", type=int, default=42)

    # Vectorizer params
    ap.add_argument("--ngram-max", type=int, default=1)
    ap.add_argument("--min-df", type=int, default=1)
    ap.add_argument("--max-features", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--lowercase", action="store_true", default=True)
    ap.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    ap.add_argument("--stop-words", default="none", choices=["none", "english"])

    # LogisticRegression params
    ap.add_argument("--class-weight", default="none", choices=["none", "balanced"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--penalty", default="l2", choices=["l2", "l1"])
    ap.add_argument("--solver", default="liblinear", choices=["liblinear", "lbfgs", "saga"])

    # Search
    ap.add_argument("--search", action="store_true", help="Run a small grid search over C/ngram/min_df.")
    ap.add_argument("--n-jobs", type=int, default=-1)

    args = ap.parse_args()
    main(
        data_path=args.data,
        model_path=args.model,
        summary_path=args.summary,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_features=args.max_features,
        class_weight=args.class_weight,
        C=args.C,
        penalty=args.penalty,
        solver=args.solver,
        lowercase=args.lowercase,
        stop_words=args.stop_words,
        search=args.search,
        n_jobs=args.n_jobs
    )