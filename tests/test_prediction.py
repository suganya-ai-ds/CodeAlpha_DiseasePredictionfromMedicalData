"""
MEDSCOPE AI — Test Suite: Prediction Module
Tests input validation, schema checking, and model inference.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import PredictionError, validate_input


class TestValidateInput:
    def _heart_input(self, **overrides):
        base = {
            "age": 55.0,
            "sex": "1",
            "cp": "2",
            "trestbps": 130.0,
            "chol": 250.0,
            "fbs": "0",
            "restecg": "0",
            "thalach": 160.0,
            "exang": "0",
            "oldpeak": 1.5,
            "slope": "1",
            "ca": "0",
            "thal": "2",
        }
        base.update(overrides)
        return base

    def test_valid_heart_disease_input(self):
        inp = self._heart_input()
        X = validate_input(inp, "heart_disease")
        assert X.shape[0] == 1
        assert "age" in X.columns

    def test_missing_feature_raises(self):
        inp = self._heart_input()
        del inp["age"]
        with pytest.raises(PredictionError, match="Missing required features"):
            validate_input(inp, "heart_disease")

    def test_nan_numeric_raises(self):
        inp = self._heart_input(age=float("nan"))
        with pytest.raises(PredictionError, match="empty"):
            validate_input(inp, "heart_disease")

    def test_inf_numeric_raises(self):
        inp = self._heart_input(age=float("inf"))
        with pytest.raises(PredictionError, match="finite"):
            validate_input(inp, "heart_disease")

    def test_non_numeric_value_raises(self):
        inp = self._heart_input(trestbps="not_a_number")
        with pytest.raises(PredictionError, match="number"):
            validate_input(inp, "heart_disease")

    def test_valid_diabetes_input(self):
        inp = {
            "age": 45.0,
            "gender": "Female",
            "polyuria": "Yes",
            "polydipsia": "No",
            "sudden_weight_loss": "No",
            "weakness": "Yes",
            "polyphagia": "No",
            "genital_thrush": "No",
            "visual_blurring": "No",
            "itching": "Yes",
            "irritability": "No",
            "delayed_healing": "No",
            "partial_paresis": "No",
            "muscle_stiffness": "No",
            "alopecia": "No",
            "obesity": "No",
        }
        X = validate_input(inp, "diabetes")
        assert X.shape == (1, 16)


class TestPredictionIntegration:
    """Integration tests that require a trained model artifact."""
    def test_missing_artifact_raises_file_not_found(self, tmp_path, monkeypatch):
        import config.settings as settings
        monkeypatch.setattr(settings, "MODELS_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            from src.train import load_best_pipeline
            load_best_pipeline("heart_disease")
