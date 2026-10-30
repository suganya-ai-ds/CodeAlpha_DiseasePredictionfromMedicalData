"""
MEDSCOPE AI — Explainability Module
Generates feature importance plots for supported models.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from config.settings import FIGURES_DIR

logger = logging.getLogger(__name__)

# ── Optional: SHAP ────────────────────────────────────────────────────────────
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    logger.info("SHAP not installed; SHAP explanations will be skipped.")


def _save_fig(fig: plt.Figure, task: str, name: str) -> Path:
    out_dir = FIGURES_DIR / task
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def get_feature_importances(
    pipeline: Pipeline,
    feature_names: List[str],
    task: str,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extract feature importances from a trained pipeline.

    Returns a dict with 'importances' Series (sorted descending) and 'method' str,
    or None if the model type does not support a simple extraction.
    """
    clf = pipeline.named_steps.get("classifier")
    if clf is None:
        return None

    clf_type = type(clf).__name__

    # ── Random Forest / tree-based: built-in importances ─────────────────────
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        # feature_names may not align after ColumnTransformer — use index
        n = len(importances)
        names = feature_names[:n] if len(feature_names) >= n else [f"feat_{i}" for i in range(n)]
        series = pd.Series(importances, index=names).sort_values(ascending=False)
        return {"importances": series, "method": "Gini Impurity (tree built-in)"}

    # ── Logistic Regression: coefficient-based ────────────────────────────────
    if hasattr(clf, "coef_") and clf.coef_ is not None:
        coef = clf.coef_.ravel()
        n = len(coef)
        names = feature_names[:n] if len(feature_names) >= n else [f"feat_{i}" for i in range(n)]
        series = pd.Series(np.abs(coef), index=names).sort_values(ascending=False)
        return {
            "importances": series,
            "method": "Absolute Logistic Regression Coefficients (magnitude only; not a causal measure)",
        }

    # ── Permutation importance (fallback for SVM, etc.) ───────────────────────
    if X_test is not None and y_test is not None:
        try:
            result = permutation_importance(
                pipeline, X_test, y_test,
                n_repeats=10, random_state=42, n_jobs=-1,
            )
            n = len(result.importances_mean)
            names = feature_names[:n] if len(feature_names) >= n else [f"feat_{i}" for i in range(n)]
            series = pd.Series(
                result.importances_mean, index=names
            ).sort_values(ascending=False)
            return {
                "importances": series,
                "method": f"Permutation Importance ({clf_type})",
            }
        except Exception as exc:
            logger.warning("Permutation importance failed: %s", exc)

    return None


def plot_feature_importance(
    importance_dict: Dict[str, Any],
    task: str,
    model_name: str,
    top_n: int = 15,
) -> Optional[Path]:
    """Plot a horizontal bar chart of feature importances and save it."""
    if importance_dict is None:
        return None
    series: pd.Series = importance_dict["importances"].head(top_n)
    method: str = importance_dict["method"]

    fig, ax = plt.subplots(figsize=(8, max(4, len(series) * 0.35)))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(series)))[::-1]
    series[::-1].plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance — {model_name}\n({method})", fontsize=10)
    ax.axvline(0, color="gray", linewidth=0.6)
    fig.tight_layout()
    return _save_fig(fig, task, f"feature_importance_{model_name}")


def explain_all(
    pipeline: Pipeline,
    feature_names: List[str],
    task: str,
    model_name: str,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Generate all available explanations for a given pipeline.

    Returns a dict with paths to saved figures and importance data.
    """
    output: Dict[str, Any] = {
        "model_name": model_name,
        "task": task,
        "figures": {},
        "importances": None,
    }

    imp = get_feature_importances(pipeline, feature_names, task, X_test, y_test)
    if imp:
        output["importances"] = imp
        fig_path = plot_feature_importance(imp, task, model_name)
        if fig_path:
            output["figures"]["feature_importance"] = str(fig_path)

    # SHAP — optional
    if _HAS_SHAP and X_test is not None:
        try:
            preprocessed = pipeline.named_steps["preprocessor"].transform(X_test)
            clf = pipeline.named_steps["classifier"]
            explainer = shap.Explainer(clf, preprocessed)
            shap_values = explainer(preprocessed, check_additivity=False)

            shap.summary_plot(shap_values, preprocessed, show=False)
            fig = plt.gcf()
            path = _save_fig(fig, task, f"shap_summary_{model_name}")
            output["figures"]["shap_summary"] = str(path)
            logger.info("SHAP summary saved: %s", path)
        except Exception as exc:
            logger.warning("SHAP explanation failed: %s", exc)

    return output
