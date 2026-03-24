from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from baseline_model import (
    build_preprocessor,
    evaluate_probabilities,
    load_baseline_dataset,
    select_best_threshold,
    split_baseline_dataset,
)

from xgboost import XGBClassifier


DEFAULT_OUTPUT_ARTIFACT = Path("ProcessAnalyses_xgb_tuned.pkl")
DEFAULT_OUTPUT_REPORT = Path("xgb_tuning_results.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune XGBoost for edge malware process detection.")
    parser.add_argument("--trials", type=int, default=30, help="Number of random hyperparameter trials.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--sample-frac", type=float, default=1.0, help="Fraction of dataset rows to use.")
    parser.add_argument("--use-gpu", action="store_true", help="Use CUDA for XGBoost training.")
    parser.add_argument(
        "--output-artifact",
        type=Path,
        default=DEFAULT_OUTPUT_ARTIFACT,
        help="Path to write tuned model artifact.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT,
        help="Path to write tuning report JSON.",
    )
    args = parser.parse_args()

    if args.trials <= 0:
        raise ValueError("--trials must be > 0")
    if not 0 < args.sample_frac <= 1.0:
        raise ValueError("--sample-frac must be in the range (0, 1]")

    return args


def sample_hyperparameters(rng: random.Random, use_gpu: bool) -> dict[str, Any]:
    tree_method = "hist"
    device_kwargs: dict[str, Any] = {"device": "cuda"} if use_gpu else {}

    return {
        "n_estimators": rng.choice([200, 300, 450, 600, 800, 1000, 1300]),
        "max_depth": rng.choice([3, 4, 5, 6, 7, 8, 10]),
        "learning_rate": rng.choice([0.01, 0.02, 0.03, 0.05, 0.08]),
        "subsample": rng.choice([0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": rng.choice([0.6, 0.75, 0.9, 1.0]),
        "min_child_weight": rng.choice([1, 2, 4, 6, 10]),
        "gamma": rng.choice([0.0, 0.1, 0.3, 0.5, 1.0]),
        "reg_alpha": rng.choice([0.0, 0.01, 0.1, 0.5, 1.0]),
        "reg_lambda": rng.choice([0.5, 1.0, 1.5, 2.0, 3.0]),
        "tree_method": tree_method,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        **device_kwargs,
    }


def tuning_score(metrics: dict[str, Any], per_sample_ms: float, artifact_size_mb: float) -> float:
    quality = (
        0.60 * float(metrics["pr_auc"])
        + 0.25 * float(metrics["f1"])
        + 0.15 * float(metrics["recall"])
    )
    latency_penalty = 0.01 * per_sample_ms
    size_penalty = 0.0025 * artifact_size_mb
    return float(quality - latency_penalty - size_penalty)


def benchmark_predict_latency_ms(pipeline: Pipeline, test_features, loops: int = 5) -> float:
    latencies = []
    for _ in range(loops):
        start = time.perf_counter()
        _ = pipeline.predict_proba(test_features)
        latencies.append(time.perf_counter() - start)
    return float(np.mean(latencies) * 1000 / max(1, len(test_features)))


def estimate_artifact_size_mb(artifact: dict[str, Any]) -> float:
    temp_path = Path(".tmp_tuned_model.pkl")
    try:
        joblib.dump(artifact, temp_path)
        return float(temp_path.stat().st_size / (1024 * 1024))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.random_state)

    features, target, feature_columns = load_baseline_dataset()
    if args.sample_frac < 1.0:
        sampled = features.copy()
        sampled["label"] = target
        sampled = sampled.sample(frac=args.sample_frac, random_state=args.random_state)
        target = sampled["label"]
        features = sampled.drop(columns=["label"])

    (
        train_features,
        validation_features,
        test_features,
        train_target,
        validation_target,
        test_target,
    ) = split_baseline_dataset(features, target)

    preprocessor, categorical_features, numeric_features = build_preprocessor(train_features)

    trial_results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_pipeline: Pipeline | None = None

    print(f"Starting XGBoost tuning with {args.trials} trials")
    for trial_index in range(1, args.trials + 1):
        params = sample_hyperparameters(rng, args.use_gpu)
        candidate = XGBClassifier(**params)
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", candidate),
            ]
        )

        trial_start = time.perf_counter()
        pipeline.fit(train_features, train_target)
        fit_seconds = float(time.perf_counter() - trial_start)

        if args.use_gpu:
            pipeline.named_steps["model"].set_params(device="cpu")

        validation_probabilities = pipeline.predict_proba(validation_features)[:, 1]
        threshold, threshold_sweep = select_best_threshold(validation_target, validation_probabilities)

        test_probabilities = pipeline.predict_proba(test_features)[:, 1]
        validation_metrics = evaluate_probabilities(validation_target, validation_probabilities, threshold)
        test_metrics = evaluate_probabilities(test_target, test_probabilities, threshold)

        artifact_probe = {
            "model": pipeline,
            "metadata": {
                "feature_columns": feature_columns,
                "categorical_features": categorical_features,
                "numeric_features": numeric_features,
                "threshold": threshold,
            },
        }
        artifact_size_mb = estimate_artifact_size_mb(artifact_probe)
        per_sample_ms = benchmark_predict_latency_ms(pipeline, test_features)
        score = tuning_score(test_metrics, per_sample_ms, artifact_size_mb)

        result = {
            "trial": trial_index,
            "params": params,
            "fit_seconds": fit_seconds,
            "threshold": float(threshold),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "predict_per_sample_ms": per_sample_ms,
            "artifact_size_mb": artifact_size_mb,
            "score": score,
            "threshold_sweep": threshold_sweep,
        }
        trial_results.append(result)

        print(
            f"[{trial_index}/{args.trials}] score={score:.4f} | "
            f"f1={test_metrics['f1']:.4f} | pr_auc={test_metrics['pr_auc']:.4f} | "
            f"latency={per_sample_ms:.6f}ms/sample | fit={fit_seconds:.2f}s"
        )

        if best_result is None or score > float(best_result["score"]):
            best_result = result
            best_pipeline = pipeline
            print(f"New best trial: {trial_index} (score={score:.4f})")

    if best_result is None or best_pipeline is None:
        raise RuntimeError("No successful tuning trials completed.")

    model_version = f"xgb-tuned-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    final_artifact = {
        "model": best_pipeline,
        "metadata": {
            "version": model_version,
            "model_type": "XGBClassifier",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_columns": feature_columns,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "threshold": best_result["threshold"],
            "best_params": best_result["params"],
            "metrics": {
                "validation": best_result["validation_metrics"],
                "test": best_result["test_metrics"],
                "threshold_sweep": best_result["threshold_sweep"],
            },
            "deployment": {
                "predict_per_sample_ms": best_result["predict_per_sample_ms"],
                "artifact_size_mb": best_result["artifact_size_mb"],
                "score": best_result["score"],
            },
            "search": {
                "trials": args.trials,
                "sample_frac": args.sample_frac,
                "use_gpu": args.use_gpu,
            },
        },
    }

    args.output_artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_artifact, args.output_artifact)

    sorted_trials = sorted(trial_results, key=lambda item: float(item["score"]), reverse=True)
    report = {
        "best_trial": best_result,
        "top_trials": sorted_trials[:10],
        "summary": {
            "trials": len(trial_results),
            "best_score": best_result["score"],
            "best_threshold": best_result["threshold"],
            "artifact": str(args.output_artifact),
            "report": str(args.output_report),
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTuning complete")
    print(f"Best score: {best_result['score']:.4f}")
    print(f"Best params: {best_result['params']}")
    print(f"Saved artifact: {args.output_artifact}")
    print(f"Saved report: {args.output_report}")


if __name__ == "__main__":
    main()