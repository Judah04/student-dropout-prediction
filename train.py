"""Train the final student-outcome model and save it.

Uses the configuration selected in notebook.ipynb (gradient boosting, class_weight="balanced"),
the same stratified 80/20 split (random_state=42), and reports the same metrics on the held-out test set.

Usage:
    uv run python train.py
    uv run python train.py --data data/data.csv --out-dir artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common import DATA_PATH, RANDOM_STATE, TARGET, feature_lists, load_data, make_preprocessor

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "artifacts"

# Best parameters found by the grid search in notebook.ipynb (section 5).
MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 3,
    "max_iter": 200,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
    "random_state": 1,
}


def build_pipeline(categorical: list[str], numerical: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor(categorical, numerical, scale=False)),
            ("model", HistGradientBoostingClassifier(**MODEL_PARAMS)),
        ]
    )


def train(data_path: Path, out_dir: Path) -> dict:
    df = load_data(data_path)
    target = TARGET.lower()
    X, y = df.drop(columns=target), df[target]
    categorical, numerical = feature_lists(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline(categorical, numerical)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    metrics = {
        "test_macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "per_class": {
            label: {k: round(report[label][k], 4) for k in ("precision", "recall", "f1-score", "support")}
            for label in pipeline.classes_
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    joblib.dump(
        {"pipeline": pipeline, "feature_names": list(X.columns), "classes": list(pipeline.classes_)},
        model_path,
    )

    metadata = {
        "model": "HistGradientBoostingClassifier",
        "params": MODEL_PARAMS,
        "classes": list(pipeline.classes_),
        "n_features": X.shape[1],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "scikit_learn": sklearn.__version__,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "metrics": metrics,
    }
    (out_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"\nSaved model to {model_path}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the student outcome model")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    train(args.data, args.out_dir)
