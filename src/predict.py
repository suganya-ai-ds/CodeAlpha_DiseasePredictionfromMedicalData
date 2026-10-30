"""
MEDSCOPE AI — Prediction Module
Runs inference using saved model artifacts and validates input schema.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import DATASET_INFO
from src.preprocessing import FEATURE_SCHEMAS, get_feature_columns
from src.train import load_best_pipeline

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """Raised when input validation or inference fails."""


def validate_input(
    input_dict: Dict[str, Any],
    task: str,
) -> pd.DataFrame:
    """
    Validate and convert a user-supplied feature dict into a single-row DataFrame.

    Raises PredictionError with an actionable message on any issue.
    """
    expected_cols = get_feature_columns(task)
    schema = FEATURE_SCHEMAS[task]

    # Check for missing required features
    missing = [c for c in expected_cols if c not in input_dict]
    if missing:
        raise PredictionError(f"Missing required features: {missing}")

    # Build DataFrame
    row = {col: [input_dict[col]] for col in expected_cols}
    X = pd.DataFrame(row)

    # Check numeric columns for NaN / inf
    for col in schema["numeric"]:
        if col not in X.columns:
            continue
        val = X[col].iloc[0]
        if pd.isna(val):
            raise PredictionError(f"Feature '{col}' must not be empty.")
        try:
            val = float(val)
        except (TypeError, ValueError):
            raise PredictionError(f"Feature '{col}' must be a number; got '{val}'.")
        if np.isinf(val):
            raise PredictionError(f"Feature '{col}' must be a finite number.")
        X[col] = val

    return X


def predict(
    task: str,
    input_dict: Dict[str, Any],
    pipeline=None,
) -> Dict[str, Any]:
    """
    Run a prediction for *task* using *input_dict*.

    Parameters
    ----------
    task        : Disease task name.
    input_dict  : {feature_name: value} mapping for all required features.
    pipeline    : Optional pre-loaded pipeline (avoids repeated I/O).

    Returns
    -------
    dict with keys:
      - 'predicted_class': int
      - 'predicted_label': str
      - 'probability': Optional[float]  (probability of positive class)
      - 'class_labels': {0: str, 1: str}
    """
    if pipeline is None:
        pipeline = load_best_pipeline(task)

    X = validate_input(input_dict, task)

    predicted_class = int(pipeline.predict(X)[0])

    info = DATASET_INFO[task]
    # Determine class labels — handle both int and string positive_class definitions
    positive_class = info["positive_class"]
    if isinstance(positive_class, str):
        positive_class = 1  # after label encoding, Positive → 1 for diabetes
    negative_class = 1 - positive_class

    class_labels = {
        positive_class: info["positive_label"],
        negative_class: info["negative_label"],
    }
    predicted_label = class_labels.get(predicted_class, f"Class {predicted_class}")

    # Probability
    probability: Optional[float] = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)[0]
            # proba index aligns with pipeline.classes_
            classes = list(pipeline.classes_)
            if positive_class in classes:
                probability = float(proba[classes.index(positive_class)])
        except Exception as exc:
            logger.warning("[%s] predict_proba failed: %s", task, exc)

    result = {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "probability": round(probability, 4) if probability is not None else None,
        "class_labels": {str(k): v for k, v in class_labels.items()},
        "task": task,
        "model": "best_pipeline",
    }
    logger.info(
        "[%s] Prediction: %s | prob=%.4f",
        task, predicted_label,
        probability if probability is not None else 0,
    )
    return result
