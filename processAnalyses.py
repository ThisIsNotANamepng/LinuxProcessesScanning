import time

from baseline_model import ARTIFACT_PATH, train_and_save_baseline_model


def main() -> None:
    start = time.time()
    artifact = train_and_save_baseline_model()
    metadata = artifact["metadata"]
    validation_metrics = metadata["metrics"]["validation"]
    test_metrics = metadata["metrics"]["test"]

    print(f"Saved model artifact to {ARTIFACT_PATH}")
    print(f"Model version: {metadata['version']}")
    print(f"Selected threshold: {metadata['threshold']:.2f}")
    print(
        "Validation metrics: "
        f"accuracy={validation_metrics['accuracy']:.4f}, "
        f"precision={validation_metrics['precision']:.4f}, "
        f"recall={validation_metrics['recall']:.4f}, "
        f"f1={validation_metrics['f1']:.4f}, "
        f"pr_auc={validation_metrics['pr_auc']:.4f}, "
        f"roc_auc={validation_metrics['roc_auc']:.4f}"
    )
    print(
        "Test metrics: "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, "
        f"recall={test_metrics['recall']:.4f}, "
        f"f1={test_metrics['f1']:.4f}, "
        f"pr_auc={test_metrics['pr_auc']:.4f}, "
        f"roc_auc={test_metrics['roc_auc']:.4f}"
    )
    print(f"Test confusion matrix: {test_metrics['confusion_matrix']}")
    print(f"Time taken: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
