"""
MEDSCOPE AI — Main Training Script
Run from the MEDSCOPE_AI/ directory:
  python train_all.py
  python train_all.py --task heart_disease
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import TASKS
from src.data_loader import load_dataset
from src.data_validation import validate_dataset
from src.evaluate import (
    evaluate_all_models,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curve,
    plot_pr_curve,
    save_metrics_csv,
)
from src.explainability import explain_all
from src.preprocessing import get_feature_columns
from src.reporting import generate_text_report, generate_data_quality_report
from src.train import train_all_models, load_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_task(task: str) -> None:
    logger.info("=" * 60)
    logger.info("TASK: %s", task.upper())
    logger.info("=" * 60)

    # Stage 3: Load & validate
    from config.settings import DATASET_INFO
    info = DATASET_INFO[task]
    df = load_dataset(task)
    rpt = validate_dataset(df, task=task, target_col=info["target_column"])
    generate_data_quality_report(task, rpt)
    logger.info(rpt.summary())

    if not rpt.passed:
        logger.error("[%s] Validation failed — aborting.", task)
        return

    # Stage 5-6: Train & evaluate
    train_output = train_all_models(task)
    eval_results = evaluate_all_models(train_output)
    metadata = load_metadata(task)

    # Save CSV metrics
    save_metrics_csv(eval_results, task)

    # Generate figures
    from config.settings import DATASET_INFO
    info = DATASET_INFO[task]
    from src.preprocessing import FEATURE_SCHEMAS

    labels = [info["negative_label"], info["positive_label"]]
    for model_name, m in eval_results.items():
        plot_confusion_matrix(m, class_labels=labels)

    plot_roc_curve(eval_results, train_output["y_test"], task)
    plot_pr_curve(eval_results, train_output["y_test"], task)
    plot_model_comparison(eval_results, task)

    # Stage 7: Explainability for best model
    best_name = train_output["best_model_name"]
    best_pipeline = train_output["best_pipeline"]
    feature_names = get_feature_columns(task)
    explain_all(
        pipeline=best_pipeline,
        feature_names=feature_names,
        task=task,
        model_name=best_name,
        X_test=train_output["X_test"],
        y_test=train_output["y_test"],
    )

    # Stage 10: Text report
    generate_text_report(task, eval_results, best_name, metadata)

    logger.info("[%s] [OK] Complete. Best model: %s", task, best_name)
    # Print summary
    best_metrics = eval_results.get(best_name, {})
    logger.info(
        "[%s] Test metrics — Acc: %.4f | F1: %.4f | AUC: %s",
        task,
        best_metrics.get("accuracy", 0),
        best_metrics.get("f1", 0),
        f"{best_metrics['roc_auc']:.4f}" if best_metrics.get("roc_auc") else "N/A",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MEDSCOPE AI — Train all disease models")
    parser.add_argument(
        "--task",
        choices=TASKS + ["all"],
        default="all",
        help="Disease task to train (default: all).",
    )
    args = parser.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    failed = []
    for task in tasks:
        try:
            run_task(task)
        except Exception as exc:
            logger.exception("[%s] FAILED: %s", task, exc)
            failed.append(task)

    if failed:
        logger.error("Tasks that failed: %s", failed)
        sys.exit(1)
    else:
        logger.info("All tasks completed successfully.")


if __name__ == "__main__":
    main()
