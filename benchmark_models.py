from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from baseline_model import (
    build_preprocessor,
    evaluate_probabilities,
    load_baseline_dataset,
    select_best_threshold,
    split_baseline_dataset,
)

try:
    from xgboost import XGBClassifier
    from xgboost.core import XGBoostError

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBoostError = Exception


RESULTS_JSON_PATH = Path("benchmark_results.json")
RESULTS_CSV_PATH = Path("benchmark_results.csv")
TOP_K_FEATURE_LATENCY_LOOPS = 5


@dataclass
class Candidate:
    name: str
    estimator: ClassifierMixin


def build_candidates(profile: str, use_gpu: bool) -> list[Candidate]:
    rf_edge_estimators = {
        "quick": 180,
        "standard": 300,
        "full": 300,
    }[profile]
    rf_quality_estimators = {
        "quick": 450,
        "standard": 700,
        "full": 1200,
    }[profile]
    xgb_edge_estimators = {
        "quick": 220,
        "standard": 350,
        "full": 350,
    }[profile]
    xgb_quality_estimators = {
        "quick": 500,
        "standard": 700,
        "full": 900,
    }[profile]

    candidates: list[Candidate] = [
        Candidate(
            name="RandomForest_Edge",
            estimator=RandomForestClassifier(
                n_estimators=rf_edge_estimators,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        Candidate(
            name="DecisionTree_Fast",
            estimator=DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced",
            ),
        ),
        Candidate(
            name="LogisticRegression_Fast",
            estimator=LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            ),
        ),
    ]

    if profile in {"standard", "full"}:
        candidates.append(
            Candidate(
                name="RandomForest_MaxQuality",
                estimator=RandomForestClassifier(
                    n_estimators=rf_quality_estimators,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        )

    if HAS_XGBOOST:
        xgb_tree_method = "hist"
        xgb_device_kwargs: dict[str, Any] = {"device": "cuda"} if use_gpu else {}
        candidates.append(
            Candidate(
                name="XGBoost_Edge",
                estimator=XGBClassifier(
                    n_estimators=xgb_edge_estimators,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    eval_metric="logloss",
                    tree_method=xgb_tree_method,
                    **xgb_device_kwargs,
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        )

        if profile in {"standard", "full"}:
            candidates.append(
                Candidate(
                    name="XGBoost_MaxQuality",
                    estimator=XGBClassifier(
                        n_estimators=xgb_quality_estimators,
                        max_depth=7,
                        learning_rate=0.03,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.5,
                        eval_metric="logloss",
                        tree_method=xgb_tree_method,
                        **xgb_device_kwargs,
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            )

    return candidates


def _estimate_artifact_size_mb(pipeline: Pipeline, metadata: dict[str, Any]) -> float:
    artifact = {"model": pipeline, "metadata": metadata}
    with tempfile.NamedTemporaryFile(suffix=".pkl") as temporary_file:
        joblib.dump(artifact, temporary_file.name)
        size_bytes = Path(temporary_file.name).stat().st_size
    return float(size_bytes / (1024 * 1024))


def _benchmark_predict_latency(pipeline: Pipeline, test_features: pd.DataFrame) -> tuple[float, float]:
    latencies_seconds: list[float] = []
    for _ in range(TOP_K_FEATURE_LATENCY_LOOPS):
        start = time.perf_counter()
        _ = pipeline.predict_proba(test_features)
        latencies_seconds.append(time.perf_counter() - start)

    average_batch_seconds = float(np.mean(latencies_seconds))
    per_sample_ms = average_batch_seconds * 1000 / max(1, len(test_features))
    return average_batch_seconds, per_sample_ms


def _confusion_stats(confusion: list[list[int]]) -> tuple[float, float]:
    if len(confusion) != 2 or len(confusion[0]) != 2 or len(confusion[1]) != 2:
        return float("nan"), float("nan")

    tn, fp = confusion[0]
    fn, tp = confusion[1]

    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    return float(false_positive_rate), float(false_negative_rate)


def _deployment_score(test_metrics: dict[str, Any], per_sample_ms: float, artifact_size_mb: float) -> float:
    quality = (
        0.55 * float(test_metrics["pr_auc"])
        + 0.30 * float(test_metrics["f1"])
        + 0.15 * float(test_metrics["recall"])
    )
    latency_penalty = 0.01 * per_sample_ms
    size_penalty = 0.0025 * artifact_size_mb
    return float(quality - latency_penalty - size_penalty)


def evaluate_candidates(profile: str, sample_frac: float, random_state: int, use_gpu: bool) -> dict[str, Any]:
    features, target, feature_columns = load_baseline_dataset()

    if sample_frac < 1.0:
        sampled = features.copy()
        sampled["label"] = target
        sampled = sampled.sample(frac=sample_frac, random_state=random_state)
        target = sampled["label"].copy()
        features = sampled.drop(columns=["label"])

    (
        train_features,
        validation_features,
        test_features,
        train_target,
        validation_target,
        test_target,
    ) = split_baseline_dataset(features, target)

    candidates = build_candidates(profile, use_gpu)
    results: list[dict[str, Any]] = []

    for candidate in candidates:
        preprocessor, categorical_features, numeric_features = build_preprocessor(train_features)
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", candidate.estimator),
            ]
        )

        fit_started = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter("always")
                pipeline.fit(train_features, train_target)
            fit_seconds = float(time.perf_counter() - fit_started)
        except XGBoostError as error:
            results.append(
                {
                    "model_name": candidate.name,
                    "status": "failed",
                    "error": str(error),
                    "deployment_score": float("-inf"),
                }
            )
            continue
        except Exception as error:
            results.append(
                {
                    "model_name": candidate.name,
                    "status": "failed",
                    "error": str(error),
                    "deployment_score": float("-inf"),
                }
            )
            continue

        validation_probabilities = pipeline.predict_proba(validation_features)[:, 1]
        threshold, threshold_sweep = select_best_threshold(validation_target, validation_probabilities)
        test_probabilities = pipeline.predict_proba(test_features)[:, 1]

        validation_metrics = evaluate_probabilities(validation_target, validation_probabilities, threshold)
        test_metrics = evaluate_probabilities(test_target, test_probabilities, threshold)

        average_batch_seconds, per_sample_ms = _benchmark_predict_latency(pipeline, test_features)
        artifact_size_mb = _estimate_artifact_size_mb(
            pipeline,
            {
                "model_name": candidate.name,
                "feature_columns": feature_columns,
                "categorical_features": categorical_features,
                "numeric_features": numeric_features,
                "threshold": threshold,
            },
        )

        false_positive_rate, false_negative_rate = _confusion_stats(test_metrics["confusion_matrix"])
        score = _deployment_score(test_metrics, per_sample_ms, artifact_size_mb)

        results.append(
            {
                "model_name": candidate.name,
                "status": "ok",
                "threshold": float(threshold),
                "fit_seconds": fit_seconds,
                "predict_batch_seconds": average_batch_seconds,
                "predict_per_sample_ms": per_sample_ms,
                "artifact_size_mb": artifact_size_mb,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
                "deployment_score": score,
                "threshold_sweep": threshold_sweep,
                "warnings": [str(warning.message) for warning in captured_warnings],
            }
        )

    ranked_results = sorted(results, key=lambda item: item["deployment_score"], reverse=True)
    successful = [result for result in ranked_results if result.get("status") == "ok"]
    return {
        "summary": {
            "candidate_count": len(ranked_results),
            "successful_count": len(successful),
            "profile": profile,
            "sample_frac": sample_frac,
            "use_gpu": use_gpu,
            "ranking_metric": "deployment_score = quality(pr_auc,f1,recall) - latency_penalty - size_penalty",
            "best_model": successful[0]["model_name"] if successful else None,
        },
        "results": ranked_results,
    }


def save_results(report: dict[str, Any]) -> None:
    RESULTS_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    flattened_rows: list[dict[str, Any]] = []
    for result in report["results"]:
        if result.get("status") != "ok":
            flattened_rows.append(
                {
                    "model_name": result["model_name"],
                    "status": "failed",
                    "error": result.get("error", "unknown"),
                }
            )
            continue

        test_metrics = result["test_metrics"]
        flattened_rows.append(
            {
                "model_name": result["model_name"],
                "status": "ok",
                "deployment_score": result["deployment_score"],
                "threshold": result["threshold"],
                "fit_seconds": result["fit_seconds"],
                "predict_batch_seconds": result["predict_batch_seconds"],
                "predict_per_sample_ms": result["predict_per_sample_ms"],
                "artifact_size_mb": result["artifact_size_mb"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
                "false_positive_rate": result["false_positive_rate"],
                "false_negative_rate": result["false_negative_rate"],
            }
        )

    pd.DataFrame(flattened_rows).to_csv(RESULTS_CSV_PATH, index=False)


def print_summary(report: dict[str, Any]) -> None:
    print(f"Candidates evaluated: {report['summary']['candidate_count']}")
    print(f"Candidates successful: {report['summary']['successful_count']}")
    print(f"Profile: {report['summary']['profile']}")
    print(f"Sample fraction: {report['summary']['sample_frac']}")
    print(f"Best model: {report['summary']['best_model']}")
    print("\nTop models:")
    successful_results = [result for result in report["results"] if result.get("status") == "ok"]
    for rank, result in enumerate(successful_results[:3], start=1):
        metrics = result["test_metrics"]
        print(
            f"{rank}. {result['model_name']} | score={result['deployment_score']:.4f} | "
            f"pr_auc={metrics['pr_auc']:.4f} | f1={metrics['f1']:.4f} | "
            f"latency={result['predict_per_sample_ms']:.6f}ms/sample | size={result['artifact_size_mb']:.2f}MB"
        )

    failed_results = [result for result in report["results"] if result.get("status") == "failed"]
    if failed_results:
        print("\nFailed candidates:")
        for result in failed_results:
            print(f"- {result['model_name']}: {result.get('error', 'unknown error')}")

    print(f"\nWrote: {RESULTS_JSON_PATH}")
    print(f"Wrote: {RESULTS_CSV_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark edge-deployable malware classifiers.")
    parser.add_argument(
        "--profile",
        choices=["quick", "standard", "full"],
        default="standard",
        help="Controls model search breadth and training cost.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of dataset rows to use for faster exploratory runs (0 < frac <= 1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for dataset sampling.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration for XGBoost candidates when available.",
    )
    args = parser.parse_args()

    if not 0 < args.sample_frac <= 1.0:
        raise ValueError("--sample-frac must be in the range (0, 1].")

    return args


def main() -> None:
    args = parse_args()
    report = evaluate_candidates(args.profile, args.sample_frac, args.random_state, args.use_gpu)
    save_results(report)
    print_summary(report)


if __name__ == "__main__":
    main()