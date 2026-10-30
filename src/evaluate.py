"""
MEDSCOPE AI — Model Evaluation
Computes all required metrics from actual predictions and generates report artefacts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from config.settings import FIGURES_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)


def compute_metrics(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
    model_name: str,
    positive_class: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate *pipeline* on the held-out test set.

    Returns a metrics dictionary with all required evaluation values.
    """
    y_pred = pipeline.predict(X_test)

    metrics: Dict[str, Any] = {
        "task": task,
        "model": model_name,
        "n_test": len(y_test),
        "class_distribution": dict(zip(*np.unique(y_test, return_counts=True))),
    }

    # ── Core classification metrics ───────────────────────────────────────────
    metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
    metrics["precision"] = round(float(
        precision_score(y_test, y_pred, pos_label=positive_class, zero_division=0)
    ), 4)
    metrics["recall"] = round(float(
        recall_score(y_test, y_pred, pos_label=positive_class, zero_division=0)
    ), 4)
    metrics["f1"] = round(float(
        f1_score(y_test, y_pred, pos_label=positive_class, zero_division=0)
    ), 4)

    # ── Specificity ───────────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else None
        metrics["specificity"] = round(float(specificity), 4) if specificity is not None else None
    else:
        metrics["specificity"] = None

    metrics["confusion_matrix"] = cm.tolist()

    # ── ROC-AUC ───────────────────────────────────────────────────────────────
    if hasattr(pipeline, "predict_proba") and len(np.unique(y_test)) == 2:
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
            metrics["avg_precision"] = round(float(
                average_precision_score(y_test, y_prob, pos_label=positive_class)
            ), 4)
            metrics["_y_prob"] = y_prob  # retained for plotting; not serialised
        except Exception as exc:
            logger.warning("[%s/%s] ROC-AUC failed: %s", task, model_name, exc)
            metrics["roc_auc"] = None
            metrics["avg_precision"] = None
    else:
        metrics["roc_auc"] = None
        metrics["avg_precision"] = None

    # ── Classification report ─────────────────────────────────────────────────
    metrics["classification_report"] = classification_report(
        y_test, y_pred, zero_division=0, output_dict=True
    )

    logger.info(
        "[%s/%s] Acc=%.4f | F1=%.4f | AUC=%s",
        task, model_name,
        metrics["accuracy"], metrics["f1"],
        f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] else "N/A",
    )
    return metrics


def evaluate_all_models(train_output: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Evaluate all trained pipelines on the shared test set.
    Returns a dict keyed by model name.
    """
    task = train_output["task"]
    X_test = train_output["X_test"]
    y_test = train_output["y_test"]
    results = train_output["results"]

    from config.settings import DATASET_INFO
    info = DATASET_INFO[task]
    positive_class = info["positive_class"]
    if isinstance(positive_class, str):
        # Will have been label-encoded; look up numeric value
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit([info["negative_label"].split(":")[0], info["positive_class"]])
        positive_class = int(le.transform([info["positive_class"]])[0])

    eval_results: Dict[str, Dict] = {}
    for name, r in results.items():
        metrics = compute_metrics(
            pipeline=r["pipeline"],
            X_test=X_test,
            y_test=y_test,
            task=task,
            model_name=name,
            positive_class=int(positive_class) if not isinstance(positive_class, str) else 1,
        )
        metrics["cv_score_mean"] = r.get("cv_score_mean")
        metrics["cv_scores"] = r.get("cv_scores", [])
        metrics["train_time_s"] = r.get("train_time_s")
        eval_results[name] = metrics

    return eval_results


# ── Figure generation ─────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, task: str, name: str) -> Path:
    out_dir = FIGURES_DIR / task
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def plot_confusion_matrix(
    metrics: Dict[str, Any],
    class_labels: Optional[List[str]] = None,
) -> Optional[Path]:
    cm = metrics.get("confusion_matrix")
    if cm is None:
        return None
    cm_array = np.array(cm)
    labels = class_labels or [str(i) for i in range(cm_array.shape[0])]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_array, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)

    thresh = cm_array.max() / 2.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(
                j, i, str(cm_array[i, j]),
                ha="center", va="center",
                color="white" if cm_array[i, j] > thresh else "black",
            )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(f"Confusion Matrix — {metrics['model']}")
    fig.tight_layout()
    return _save_fig(fig, metrics["task"], f"cm_{metrics['model']}")


def plot_roc_curve(eval_results: Dict[str, Dict], y_test: pd.Series, task: str) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(6, 5))
    any_plotted = False
    for model_name, metrics in eval_results.items():
        y_prob = metrics.get("_y_prob")
        if y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {task.replace('_', ' ').title()}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return _save_fig(fig, task, "roc_curves")


def plot_pr_curve(eval_results: Dict[str, Dict], y_test: pd.Series, task: str) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=(6, 5))
    any_plotted = False
    for model_name, metrics in eval_results.items():
        y_prob = metrics.get("_y_prob")
        if y_prob is None:
            continue
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, label=f"{model_name} (AP={ap:.3f})")
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — {task.replace('_', ' ').title()}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return _save_fig(fig, task, "pr_curves")


def plot_model_comparison(eval_results: Dict[str, Dict], task: str) -> Optional[Path]:
    rows = []
    for name, m in eval_results.items():
        rows.append({
            "Model": name,
            "Accuracy": m.get("accuracy"),
            "F1": m.get("f1"),
            "ROC-AUC": m.get("roc_auc"),
            "Recall": m.get("recall"),
        })
    df = pd.DataFrame(rows).set_index("Model")
    df = df.dropna(axis=1, how="all")

    fig, ax = plt.subplots(figsize=(8, 4))
    df.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white")
    ax.set_title(f"Model Comparison — {task.replace('_', ' ').title()}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xticklabels(df.index, rotation=25, ha="right")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return _save_fig(fig, task, "model_comparison")


def save_metrics_csv(eval_results: Dict[str, Dict], task: str) -> Path:
    out_dir = REPORTS_DIR / task
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, m in eval_results.items():
        rows.append({
            "model": name,
            "accuracy": m.get("accuracy"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1": m.get("f1"),
            "specificity": m.get("specificity"),
            "roc_auc": m.get("roc_auc"),
            "avg_precision": m.get("avg_precision"),
            "cv_score_mean": m.get("cv_score_mean"),
            "train_time_s": m.get("train_time_s"),
        })
    path = out_dir / "metrics_comparison.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Saved metrics CSV: %s", path)
    return path
