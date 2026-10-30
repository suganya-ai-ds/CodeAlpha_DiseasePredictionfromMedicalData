"""
MEDSCOPE AI — Test Suite: Preprocessing
Tests the preprocessing pipeline build and feature schema utilities.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (
    FEATURE_SCHEMAS,
    build_preprocessor,
    get_feature_columns,
    prepare_xy,
)


def _make_heart_disease_fixture(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(30, 80, n).astype(float),
        "sex": rng.choice([0, 1], n).astype(float),
        "cp": rng.choice([0, 1, 2, 3], n).astype(float),
        "trestbps": rng.integers(90, 180, n).astype(float),
        "chol": rng.integers(150, 350, n).astype(float),
        "fbs": rng.choice([0, 1], n).astype(float),
        "restecg": rng.choice([0, 1, 2], n).astype(float),
        "thalach": rng.integers(80, 200, n).astype(float),
        "exang": rng.choice([0, 1], n).astype(float),
        "oldpeak": rng.uniform(0, 5, n),
        "slope": rng.choice([0, 1, 2], n).astype(float),
        "ca": rng.choice([0, 1, 2, 3], n).astype(float),
        "thal": rng.choice([1, 2, 3], n).astype(float),
        "target": rng.choice([0, 1], n),
    })


def _make_diabetes_fixture(n=40, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(20, 70, n).astype(float),
        "gender": rng.choice(["Male", "Female"], n),
        "polyuria": rng.choice(["Yes", "No"], n),
        "polydipsia": rng.choice(["Yes", "No"], n),
        "sudden_weight_loss": rng.choice(["Yes", "No"], n),
        "weakness": rng.choice(["Yes", "No"], n),
        "polyphagia": rng.choice(["Yes", "No"], n),
        "genital_thrush": rng.choice(["Yes", "No"], n),
        "visual_blurring": rng.choice(["Yes", "No"], n),
        "itching": rng.choice(["Yes", "No"], n),
        "irritability": rng.choice(["Yes", "No"], n),
        "delayed_healing": rng.choice(["Yes", "No"], n),
        "partial_paresis": rng.choice(["Yes", "No"], n),
        "muscle_stiffness": rng.choice(["Yes", "No"], n),
        "alopecia": rng.choice(["Yes", "No"], n),
        "obesity": rng.choice(["Yes", "No"], n),
        "class": rng.choice(["Positive", "Negative"], n),
    })


class TestFeatureSchemas:
    def test_all_tasks_have_schemas(self):
        for task in ["heart_disease", "diabetes", "breast_cancer"]:
            schema = FEATURE_SCHEMAS[task]
            assert "numeric" in schema
            assert "categorical" in schema

    def test_heart_disease_features(self):
        cols = get_feature_columns("heart_disease")
        assert "age" in cols
        assert "thalach" in cols
        assert "target" not in cols  # target must NOT be in feature list

    def test_diabetes_features(self):
        cols = get_feature_columns("diabetes")
        assert "age" in cols
        assert "polyuria" in cols
        assert "class" not in cols

    def test_breast_cancer_features(self):
        cols = get_feature_columns("breast_cancer")
        assert len(cols) == 30
        assert "mean radius" in cols


class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        from sklearn.compose import ColumnTransformer
        pp = build_preprocessor("heart_disease")
        assert isinstance(pp, ColumnTransformer)

    def test_fits_and_transforms_heart_disease(self):
        df = _make_heart_disease_fixture()
        X, y = prepare_xy(df, "heart_disease", "target")
        pp = build_preprocessor("heart_disease")
        X_tr = pp.fit_transform(X)
        assert X_tr.shape[0] == len(df)

    def test_fits_and_transforms_diabetes(self):
        df = _make_diabetes_fixture()
        X, y = prepare_xy(df, "diabetes", "class")
        pp = build_preprocessor("diabetes")
        X_tr = pp.fit_transform(X)
        assert X_tr.shape[0] == len(df)

    def test_no_nan_after_transform(self):
        df = _make_heart_disease_fixture()
        df.loc[0, "age"] = np.nan
        df.loc[1, "chol"] = np.nan
        X, y = prepare_xy(df, "heart_disease", "target")
        pp = build_preprocessor("heart_disease")
        X_tr = pp.fit_transform(X)
        assert not np.isnan(X_tr).any()

    def test_unknown_category_handled(self):
        """Test that OrdinalEncoder handles unseen categories at inference time."""
        df_train = _make_heart_disease_fixture(n=40)
        df_test = _make_heart_disease_fixture(n=10)
        # Inject a category that doesn't appear in training
        df_test.loc[0, "cp"] = 99
        X_train, _ = prepare_xy(df_train, "heart_disease", "target")
        X_test, _ = prepare_xy(df_test, "heart_disease", "target")
        pp = build_preprocessor("heart_disease")
        pp.fit(X_train)
        # Should not raise — unknown values become -1 via handle_unknown='use_encoded_value'
        X_transformed = pp.transform(X_test)
        assert X_transformed.shape[0] == len(df_test)


class TestPrepareXY:
    def test_target_not_in_features(self):
        df = _make_heart_disease_fixture()
        X, y = prepare_xy(df, "heart_disease", "target")
        assert "target" not in X.columns

    def test_y_is_integer_encoded(self):
        df = _make_diabetes_fixture()
        _, y = prepare_xy(df, "diabetes", "class")
        assert y.dtype in [int, "int32", "int64"]
        assert set(y.unique()).issubset({0, 1})

    def test_x_shape(self):
        df = _make_heart_disease_fixture(n=30)
        X, _ = prepare_xy(df, "heart_disease", "target")
        expected_cols = len(get_feature_columns("heart_disease"))
        assert X.shape == (30, expected_cols)
