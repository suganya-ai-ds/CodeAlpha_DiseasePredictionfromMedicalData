"""
MEDSCOPE AI — Model Training Engine
Trains Logistic Regression, SVM, Random Forest (and XGBoost if available)
for a given disease task and saves the best pipeline artifact.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

import config.settings as settings
from config.settings import (
    CV_FOLDS,
    DATASET_INFO,
    PARAM_GRIDS,
    RANDOM_SEED,
    TEST_SIZE,
)
from src.data_loader import load_dataset
from src.data_validation import validate_dataset
from src.preprocessing import build_preprocessor, get_feature_columns, prepare_xy

logger = logging.getLogger(__name__)

# ── Optional: XGBoost ─────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False
    logger.info("XGBoost not installed; XGBoost experiments will be skipped.")


def _get_classifiers(random_state: int) -> Dict[str, Any]:
    clfs = {
        "logistic_regression": LogisticRegression(random_state=random_state),
        "svm": SVC(probability=True, random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state),
    }
    if _HAS_XGBOOST:
        clfs["xgboost"] = XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
        )
    return clfs


def _build_pipeline(preprocessor, classifier) -> Pipeline:
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def train_all_models(
    task: str,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Train and evaluate all classifiers for *task*.

    Returns a dict with keys:
      - 'best_model_name'
      - 'best_pipeline'
      - 'results': Dict[model_name, metrics_dict]
      - 'X_test', 'y_test'
      - 'task_meta'
    """
    artifact_dir = settings.MODELS_DIR / task
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "summary.json"

    # ── Load & validate data ──────────────────────────────────────────────────
    info = DATASET_INFO[task]
    target_col = info["target_column"]

    logger.info("[%s] Loading dataset…", task)
    df = load_dataset(task)

    rpt = validate_dataset(
        df, task=task, target_col=target_col,
        required_features=get_feature_columns(task),
    )
    if not rpt.passed:
        raise RuntimeError(f"[{task}] Validation failed:\n" + rpt.summary())

    # Drop duplicate rows
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < n_before:
        logger.warning("[%s] Dropped %d duplicate rows.", task, n_before - len(df))

    X, y = prepare_xy(df, task, target_col)

    # ── Stratified train/test split ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    logger.info(
        "[%s] Split — train: %d, test: %d | classes: %s",
        task, len(X_train), len(X_test), sorted(y.unique().tolist()),
    )

    preprocessor = build_preprocessor(task)
    classifiers = _get_classifiers(RANDOM_SEED)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    results: Dict[str, Any] = {}

    for name, clf in classifiers.items():
        logger.info("[%s] Training %s…", task, name)
        pipeline = _build_pipeline(preprocessor, clf)
        param_grid = PARAM_GRIDS.get(name, {})

        t0 = time.perf_counter()
        if param_grid:
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            cv_scores = search.cv_results_["mean_test_score"]
            best_cv_score = search.best_score_
            best_params = search.best_params_
        else:
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline
            best_params = {}
            best_cv_score = np.nan
            cv_scores = []

        train_time = time.perf_counter() - t0

        results[name] = {
            "pipeline": best_pipeline,
            "best_params": best_params,
            "cv_score_mean": float(best_cv_score) if not np.isnan(best_cv_score) else None,
            "cv_scores": cv_scores.tolist() if hasattr(cv_scores, "tolist") else list(cv_scores),
            "train_time_s": round(train_time, 3),
        }

        logger.info(
            "[%s] %s — best CV ROC-AUC: %.4f (%.1f s)",
            task, name,
            best_cv_score if not np.isnan(best_cv_score) else 0,
            train_time,
        )

    # ── Select best model by CV ROC-AUC ──────────────────────────────────────
    scored = {
        n: r["cv_score_mean"]
        for n, r in results.items()
        if r["cv_score_mean"] is not None
    }
    if scored:
        best_name = max(scored, key=lambda k: scored[k])
    else:
        best_name = list(results.keys())[0]

    best_pipeline = results[best_name]["pipeline"]
    logger.info("[%s] Selected best model: %s", task, best_name)

    # ── Save artifacts ────────────────────────────────────────────────────────
    pipeline_path = artifact_dir / f"best_pipeline.joblib"
    joblib.dump(best_pipeline, pipeline_path)

    for name, r in results.items():
        joblib.dump(r["pipeline"], artifact_dir / f"{name}_pipeline.joblib")

    # Save metadata JSON
    metadata = {
        "task": task,
        "dataset": info["name"],
        "source": info["url"],
        "target_column": target_col,
        "target_definition": info["description"],
        "feature_schema": get_feature_columns(task),
        "algorithm": best_name,
        "best_params": results[best_name]["best_params"],
        "cv_folds": CV_FOLDS,
        "cv_roc_auc": results[best_name]["cv_score_mean"],
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "random_seed": RANDOM_SEED,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().isoformat(),
        "known_limitations": info["limitations"],
    }

    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("[%s] Artifacts saved to %s", task, artifact_dir)

    return {
        "task": task,
        "best_model_name": best_name,
        "best_pipeline": best_pipeline,
        "results": results,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
        "task_meta": metadata,
    }


def load_best_pipeline(task: str) -> Pipeline:
    """Load the saved best pipeline for *task*."""
    path = settings.MODELS_DIR / task / "best_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found for task '{task}'. "
            f"Run train.py first. (Expected: {path})"
        )
    return joblib.load(path)


def load_pipeline(task: str, model_name: str) -> Pipeline:
    """Load a specific named pipeline for *task*."""
    path = settings.MODELS_DIR / task / f"{model_name}_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline not found: {path}")
    return joblib.load(path)


def load_metadata(task: str) -> Dict[str, Any]:
    """Load training metadata JSON for *task*."""
    path = settings.MODELS_DIR / task / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    with open(path) as f:
        return json.load(f)
