"""
MEDSCOPE AI — Test Suite: Metrics
Tests the evaluation metric computation logic.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import compute_metrics, save_metrics_csv
from src.preprocessing import build_preprocessor, prepare_xy


def _make_fixture(n=60, seed=5):
    rng = np.random.default_rng(seed)
    import pandas as pd
    return pd.DataFrame({
        "age": rng.integers(30, 80, n).astype(float),
        "sex": rng.choice([0.0, 1.0], n),
        "cp": rng.choice([0.0, 1.0, 2.0, 3.0], n),
        "trestbps": rng.integers(90, 180, n).astype(float),
        "chol": rng.integers(150, 350, n).astype(float),
        "fbs": rng.choice([0.0, 1.0], n),
        "restecg": rng.choice([0.0, 1.0, 2.0], n),
        "thalach": rng.integers(80, 200, n).astype(float),
        "exang": rng.choice([0.0, 1.0], n),
        "oldpeak": rng.uniform(0, 5, n),
        "slope": rng.choice([0.0, 1.0, 2.0], n),
        "ca": rng.choice([0.0, 1.0, 2.0, 3.0], n),
        "thal": rng.choice([1.0, 2.0, 3.0], n),
        "target": rng.choice([0, 1], n),
    })


def _build_fitted_pipeline(df):
    from sklearn.model_selection import train_test_split
    X, y = prepare_xy(df, "heart_disease", "target")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    pp = build_preprocessor("heart_disease")
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    pipe = Pipeline([("preprocessor", pp), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test


class TestComputeMetrics:
    def setup_method(self):
        df = _make_fixture()
        self.pipe, self.X_test, self.y_test = _build_fitted_pipeline(df)

    def test_required_keys_present(self):
        m = compute_metrics(self.pipe, self.X_test, self.y_test,
                            task="heart_disease", model_name="lr")
        for key in ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc"]:
            assert key in m, f"Missing key: {key}"

    def test_accuracy_in_range(self):
        m = compute_metrics(self.pipe, self.X_test, self.y_test,
                            task="heart_disease", model_name="lr")
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_f1_in_range(self):
        m = compute_metrics(self.pipe, self.X_test, self.y_test,
                            task="heart_disease", model_name="lr")
        assert 0.0 <= m["f1"] <= 1.0

    def test_roc_auc_in_range(self):
        m = compute_metrics(self.pipe, self.X_test, self.y_test,
                            task="heart_disease", model_name="lr")
        if m["roc_auc"] is not None:
            assert 0.0 <= m["roc_auc"] <= 1.0

    def test_confusion_matrix_shape(self):
        m = compute_metrics(self.pipe, self.X_test, self.y_test,
                            task="heart_disease", model_name="lr")
        cm = m["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_specificity_is_none_or_valid(self):
        m = compute_metrics(self.pipe, self.X_test, self.y_test,
                            task="heart_disease", model_name="lr")
        if m["specificity"] is not None:
            assert 0.0 <= m["specificity"] <= 1.0


class TestSaveMetricsCSV:
    def test_creates_csv_file(self, tmp_path, monkeypatch):
        import config.settings as settings
        monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path)

        eval_results = {
            "lr": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.80,
                "f1": 0.81,
                "specificity": 0.87,
                "roc_auc": 0.89,
                "avg_precision": 0.84,
                "cv_score_mean": 0.88,
                "train_time_s": 0.5,
            }
        }
        path = save_metrics_csv(eval_results, "heart_disease")
        assert path.exists()
        df = pd.read_csv(path)
        assert "model" in df.columns
        assert df.iloc[0]["model"] == "lr"
