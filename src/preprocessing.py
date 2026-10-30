"""
MEDSCOPE AI — Preprocessing Pipelines
Builds a disease-specific sklearn Pipeline + ColumnTransformer.
All fitting happens exclusively on training data.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ── Feature schemas (drives ColumnTransformer) ────────────────────────────────

FEATURE_SCHEMAS: Dict[str, Dict[str, List[str]]] = {
    "heart_disease": {
        "numeric": [
            "age", "trestbps", "chol", "thalach", "oldpeak",
        ],
        "categorical": [
            "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
        ],
    },
    "diabetes": {
        "numeric": ["age"],
        "categorical": [
            "gender", "polyuria", "polydipsia", "sudden_weight_loss",
            "weakness", "polyphagia", "genital_thrush", "visual_blurring",
            "itching", "irritability", "delayed_healing", "partial_paresis",
            "muscle_stiffness", "alopecia", "obesity",
        ],
    },
    "breast_cancer": {
        "numeric": [
            "mean radius", "mean texture", "mean perimeter", "mean area",
            "mean smoothness", "mean compactness", "mean concavity",
            "mean concave points", "mean symmetry", "mean fractal dimension",
            "radius error", "texture error", "perimeter error", "area error",
            "smoothness error", "compactness error", "concavity error",
            "concave points error", "symmetry error", "fractal dimension error",
            "worst radius", "worst texture", "worst perimeter", "worst area",
            "worst smoothness", "worst compactness", "worst concavity",
            "worst concave points", "worst symmetry", "worst fractal dimension",
        ],
        "categorical": [],
    },
}

# Human-readable feature labels for the Streamlit UI
FEATURE_UI_META: Dict[str, Dict[str, Dict]] = {
    "heart_disease": {
        "age":      {"label": "Age", "type": "numeric", "unit": "years",   "min": 20, "max": 100, "help": "Patient's age in years"},
        "sex":      {"label": "Sex", "type": "categorical", "options": {"0": "Female", "1": "Male"}},
        "cp":       {"label": "Chest Pain Type", "type": "categorical",
                     "options": {"0": "Typical Angina", "1": "Atypical Angina", "2": "Non-Anginal Pain", "3": "Asymptomatic"}},
        "trestbps": {"label": "Resting Blood Pressure", "type": "numeric", "unit": "mm Hg", "min": 80, "max": 220},
        "chol":     {"label": "Serum Cholesterol", "type": "numeric", "unit": "mg/dl", "min": 100, "max": 600},
        "fbs":      {"label": "Fasting Blood Sugar > 120 mg/dl", "type": "categorical",
                     "options": {"0": "No (≤120 mg/dl)", "1": "Yes (>120 mg/dl)"}},
        "restecg":  {"label": "Resting ECG Results", "type": "categorical",
                     "options": {"0": "Normal", "1": "ST-T Abnormality", "2": "Left Ventricular Hypertrophy"}},
        "thalach":  {"label": "Max Heart Rate Achieved", "type": "numeric", "unit": "bpm", "min": 60, "max": 220},
        "exang":    {"label": "Exercise-Induced Angina", "type": "categorical",
                     "options": {"0": "No", "1": "Yes"}},
        "oldpeak":  {"label": "ST Depression (Exercise vs Rest)", "type": "numeric", "unit": "mm", "min": 0.0, "max": 10.0},
        "slope":    {"label": "Slope of Peak Exercise ST Segment", "type": "categorical",
                     "options": {"0": "Upsloping", "1": "Flat", "2": "Downsloping"}},
        "ca":       {"label": "Major Vessels Coloured (Fluoroscopy)", "type": "categorical",
                     "options": {"0": "0", "1": "1", "2": "2", "3": "3"}},
        "thal":     {"label": "Thalassemia", "type": "categorical",
                     "options": {"1": "Normal", "2": "Fixed Defect", "3": "Reversable Defect"}},
    },
    "diabetes": {
        "age":                 {"label": "Age", "type": "numeric", "unit": "years", "min": 16, "max": 90},
        "gender":              {"label": "Gender", "type": "categorical", "options": {"Male": "Male", "Female": "Female"}},
        "polyuria":            {"label": "Polyuria (Excessive Urination)", "type": "categorical", "options": {"Yes": "Yes", "No": "No"}},
        "polydipsia":          {"label": "Polydipsia (Excessive Thirst)", "type": "categorical", "options": {"Yes": "Yes", "No": "No"}},
        "sudden_weight_loss":  {"label": "Sudden Weight Loss", "type": "categorical", "options": {"Yes": "Yes", "No": "No"}},
        "weakness":            {"label": "Weakness", "type": "categorical", "options": {"Yes": "Yes", "No": "No"}},
        "polyphagia":          {"label": "Polyphagia (Excessive Hunger)", "type": "categorical", "options": {"Yes": "Yes", "No": "No"}},
        "genital_thrush":      {"label": "Genital Thrush", "type": "categorical", "options": {"Yes": "Yes", "No": "No"}},
        "visual_blurring":     {"label": "Visual Blurring", "type": "categorical", "options": {"Yes": "Yes", "No": "No"}},
        "itching":             {"label": "Itching", "type": "categorical", "options": {"Yes": "Yes", "No": "No"}},
        "irritability":        {"label": "Irritability", "type": "categorical", "options": {"Yes": "Yes", "No": "No"}},
        "delayed_healing":     {"label": "Delayed Healing", "type": "categorical", "options": {"Yes": "Yes", "No": "No"}},
        "partial_paresis":     {"label": "Partial Paresis (Muscle Weakness)", "type": "categorical", "options": {"Yes": "Yes", "No": "No"}},
        "muscle_stiffness":    {"label": "Muscle Stiffness", "type": "categorical", "options": {"Yes": "Yes", "No": "No"}},
        "alopecia":            {"label": "Alopecia (Hair Loss)", "type": "categorical", "options": {"Yes": "Yes", "No": "No"}},
        "obesity":             {"label": "Obesity", "type": "categorical", "options": {"Yes": "Yes", "No": "No"}},
    },
    "breast_cancer": {
        # 30 numeric features — provide a generic meta; real labels come from sklearn feature_names
        **{
            feat: {"label": feat.replace("_", " ").title(), "type": "numeric", "unit": "", "min": 0.0, "max": 30.0}
            for feat in FEATURE_SCHEMAS["breast_cancer"]["numeric"]
        }
    },
}


def _build_numeric_transformer() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def _build_categorical_transformer() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
    ])


def build_preprocessor(task: str) -> ColumnTransformer:
    """
    Create a ColumnTransformer for *task* using the feature schema.

    Note: Tree-based models also receive scaled features here for consistency;
    scaling does not hurt trees and allows the same preprocessor to serve all models.
    """
    schema = FEATURE_SCHEMAS.get(task)
    if schema is None:
        raise ValueError(f"No feature schema defined for task '{task}'.")

    transformers = []
    if schema["numeric"]:
        transformers.append(("numeric", _build_numeric_transformer(), schema["numeric"]))
    if schema["categorical"]:
        transformers.append(("categorical", _build_categorical_transformer(), schema["categorical"]))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_xy(
    df: pd.DataFrame,
    task: str,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into feature matrix X and target series y.
    Encodes the target as integers when it contains strings.
    """
    schema = FEATURE_SCHEMAS[task]
    feature_cols = schema["numeric"] + schema["categorical"]
    # Keep only the columns that actually exist (handles variant column names)
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Detect non-numeric targets (object, string, Arrow-backed string, category, etc.)
    _is_non_numeric = (
        pd.api.types.is_string_dtype(y)
        or pd.api.types.is_object_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
        or hasattr(y.dtype, "pyarrow_dtype")  # ArrowDtype (pandas 3+)
    )
    if not _is_non_numeric:
        # Final check: try converting; if it raises, treat as categorical
        try:
            y.astype(int)
        except (ValueError, TypeError):
            _is_non_numeric = True

    if _is_non_numeric:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        y = pd.Series(y_encoded, index=y.index, name=target_col, dtype=int)
        logger.info("[%s] Target encoded: %s", task, dict(zip(le.classes_, le.transform(le.classes_))))
        return X, y

    return X, y.astype(int)


def get_feature_columns(task: str) -> List[str]:
    """Return the ordered list of feature columns for *task*."""
    schema = FEATURE_SCHEMAS[task]
    return schema["numeric"] + schema["categorical"]
