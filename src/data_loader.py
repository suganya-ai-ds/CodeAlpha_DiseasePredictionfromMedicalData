"""
MEDSCOPE AI — Data Loader
Loads each disease dataset from its verified source (built-in sklearn or local CSV).
Returns a plain DataFrame with a documented target column.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.datasets import load_breast_cancer

# Conditional imports so the module loads even when requests is absent
try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

from config.settings import DATA_RAW_DIR, DATASET_INFO

logger = logging.getLogger(__name__)


# ── Heart Disease ─────────────────────────────────────────────────────────────

HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target",
]

HEART_DISEASE_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


def _load_heart_disease_local() -> Optional[pd.DataFrame]:
    """Try to load from the local raw/heart_disease.csv file."""
    local = DATA_RAW_DIR / "heart_disease.csv"
    if local.exists():
        df = pd.read_csv(local)
        logger.info("Heart disease: loaded from local file (%d rows).", len(df))
        return df
    return None


def _load_heart_disease_download() -> Optional[pd.DataFrame]:
    """Download the Cleveland Heart Disease CSV from UCI if possible."""
    if not _HAS_REQUESTS:
        logger.warning("Heart disease: 'requests' not available; cannot download.")
        return None
    try:
        resp = requests.get(HEART_DISEASE_UCI_URL, timeout=15)
        resp.raise_for_status()
        local = DATA_RAW_DIR / "heart_disease.csv"
        local.write_text(resp.text)
        df = pd.read_csv(local, header=None, names=HEART_COLUMNS, na_values="?")
        # Binarise target: 0 = no disease; 1-4 = disease present
        df["target"] = (df["target"] > 0).astype(int)
        df.to_csv(local, index=False)
        logger.info("Heart disease: downloaded and saved (%d rows).", len(df))
        return df
    except Exception as exc:
        logger.warning("Heart disease: download failed — %s", exc)
        return None


def load_heart_disease() -> pd.DataFrame:
    """
    Return the Cleveland Heart Disease DataFrame.
    Source: UCI / sklearn-datasets / local CSV.
    Target column: 'target' (0=no disease, 1=disease present).
    """
    df = _load_heart_disease_local()
    if df is None:
        df = _load_heart_disease_download()
    if df is None:
        raise FileNotFoundError(
            "Heart disease dataset not found. "
            f"Please download from {HEART_DISEASE_UCI_URL} and save to "
            f"{DATA_RAW_DIR / 'heart_disease.csv'}, "
            "or ensure network access for automatic download."
        )
    # Ensure target is binarised even if loaded from existing local file
    if "target" in df.columns and df["target"].max() > 1:
        df["target"] = (df["target"] > 0).astype(int)
    return df


# ── Diabetes (Early Stage Risk) ───────────────────────────────────────────────

DIABETES_COLUMNS = [
    "age", "gender", "polyuria", "polydipsia", "sudden_weight_loss",
    "weakness", "polyphagia", "genital_thrush", "visual_blurring",
    "itching", "irritability", "delayed_healing", "partial_paresis",
    "muscle_stiffness", "alopecia", "obesity", "class",
]

DIABETES_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00529/diabetes_data_upload.csv"
)


def _load_diabetes_local() -> Optional[pd.DataFrame]:
    local = DATA_RAW_DIR / "diabetes.csv"
    if local.exists():
        df = pd.read_csv(local)
        logger.info("Diabetes: loaded from local file (%d rows).", len(df))
        return df
    return None


def _load_diabetes_download() -> Optional[pd.DataFrame]:
    if not _HAS_REQUESTS:
        return None
    try:
        resp = requests.get(DIABETES_UCI_URL, timeout=15)
        resp.raise_for_status()
        local = DATA_RAW_DIR / "diabetes.csv"
        local.write_bytes(resp.content)
        df = pd.read_csv(local)
        logger.info("Diabetes: downloaded and saved (%d rows).", len(df))
        return df
    except Exception as exc:
        logger.warning("Diabetes: download failed — %s", exc)
        return None


def load_diabetes() -> pd.DataFrame:
    """
    Return the Early Stage Diabetes Risk Prediction DataFrame.
    Source: UCI dataset #529.
    Target column: 'class' ('Positive' / 'Negative').
    """
    df = _load_diabetes_local()
    if df is None:
        df = _load_diabetes_download()
    if df is None:
        raise FileNotFoundError(
            "Diabetes dataset not found. "
            f"Please download from {DIABETES_UCI_URL} and save to "
            f"{DATA_RAW_DIR / 'diabetes.csv'}, "
            "or ensure network access for automatic download."
        )
    # Standardise column names (strip whitespace, lowercase except 'class')
    df.columns = [
        c.strip().lower().replace(" ", "_") for c in df.columns
    ]
    # Ensure target column is named 'class'
    if "class" not in df.columns:
        # Common alternative column names
        for alt in ["diabetes", "outcome", "target"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "class"})
                break
    return df


# ── Breast Cancer (WDBC via sklearn) ─────────────────────────────────────────

def load_breast_cancer_data() -> pd.DataFrame:
    """
    Return the Wisconsin Diagnostic Breast Cancer DataFrame.
    Source: sklearn.datasets.load_breast_cancer (UCI WDBC).
    Target column: 'target' (0=malignant, 1=benign).
    """
    bunch = load_breast_cancer()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    logger.info("Breast cancer: loaded from sklearn (%d rows).", len(df))
    return df


# ── Unified loader ────────────────────────────────────────────────────────────

LOADERS = {
    "heart_disease": load_heart_disease,
    "diabetes": load_diabetes,
    "breast_cancer": load_breast_cancer_data,
}


def load_dataset(task: str) -> pd.DataFrame:
    """
    Load the dataset for *task*.

    Parameters
    ----------
    task : str
        One of 'heart_disease', 'diabetes', 'breast_cancer'.

    Returns
    -------
    pd.DataFrame
    """
    if task not in LOADERS:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(LOADERS)}")
    return LOADERS[task]()
