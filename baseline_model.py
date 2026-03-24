from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ARTIFACT_PATH = Path("ProcessAnalyses.pkl")
DATASET_PATH = Path("CombinedSets.csv")
MODEL_VERSION = "0.2.0"
TARGET_COLUMN = "label"
DROP_COLUMNS = ["ts", "EXC", "type", "CPUNR", "PID"]
THRESHOLD_CANDIDATES = np.linspace(0.10, 0.90, 17)


def load_baseline_dataset(csv_path: str | Path = DATASET_PATH) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    dataframe = pd.read_csv(csv_path)
    dataframe = dataframe.drop(columns=[column for column in DROP_COLUMNS if column in dataframe.columns])

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    feature_columns = [column for column in dataframe.columns if column != TARGET_COLUMN]
    features = dataframe[feature_columns].copy()
    target = dataframe[TARGET_COLUMN].copy()
    return features, target, feature_columns


def build_preprocessor(features: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_features = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [column for column in features.columns if column not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor, categorical_features, numeric_features


def build_baseline_pipeline(features: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
    preprocessor, categorical_features, numeric_features = build_preprocessor(features)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return pipeline, categorical_features, numeric_features


def split_baseline_dataset(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    train_valid_features, test_features, train_valid_target, test_target = train_test_split(
        features,
        target,
        test_size=0.10,
        random_state=42,
        stratify=target,
    )
    train_features, validation_features, train_target, validation_target = train_test_split(
        train_valid_features,
        train_valid_target,
        test_size=1 / 9,
        random_state=42,
        stratify=train_valid_target,
    )
    return (
        train_features,
        validation_features,
        test_features,
        train_target,
        validation_target,
        test_target,
    )


def select_best_threshold(target: pd.Series, probabilities: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    threshold_results: list[dict[str, float]] = []
    best_threshold = 0.50
    best_score = (-1.0, -1.0, -1.0)

    for threshold in THRESHOLD_CANDIDATES:
        predictions = (probabilities >= threshold).astype(int)
        f1 = f1_score(target, predictions, zero_division=0)
        precision = precision_score(target, predictions, zero_division=0)
        recall = recall_score(target, predictions, zero_division=0)
        threshold_results.append(
            {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
        )
        score = (float(f1), float(precision), float(recall))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, threshold_results


def evaluate_probabilities(target: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(target, probabilities)),
        "confusion_matrix": confusion_matrix(target, predictions).tolist(),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(target, probabilities))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


def train_and_save_baseline_model(
    dataset_path: str | Path = DATASET_PATH,
    artifact_path: str | Path = ARTIFACT_PATH,
) -> dict[str, Any]:
    features, target, feature_columns = load_baseline_dataset(dataset_path)
    pipeline, categorical_features, numeric_features = build_baseline_pipeline(features)
    (
        train_features,
        validation_features,
        test_features,
        train_target,
        validation_target,
        test_target,
    ) = split_baseline_dataset(features, target)

    pipeline.fit(train_features, train_target)

    validation_probabilities = pipeline.predict_proba(validation_features)[:, 1]
    best_threshold, threshold_results = select_best_threshold(validation_target, validation_probabilities)
    test_probabilities = pipeline.predict_proba(test_features)[:, 1]

    artifact = {
        "model": pipeline,
        "metadata": {
            "version": MODEL_VERSION,
            "model_type": "RandomForestClassifier",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_columns": feature_columns,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "threshold": best_threshold,
            "metrics": {
                "validation": evaluate_probabilities(validation_target, validation_probabilities, best_threshold),
                "test": evaluate_probabilities(test_target, test_probabilities, best_threshold),
                "threshold_sweep": threshold_results,
            },
        },
    }
    joblib.dump(artifact, artifact_path)
    return artifact


def load_model_artifact(artifact_path: str | Path = ARTIFACT_PATH) -> dict[str, Any]:
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict) or "model" not in artifact or "metadata" not in artifact:
        raise ValueError("Unexpected model artifact format")
    return artifact