"""
MEDSCOPE AI — Test Suite: App Helpers
Tests utility functions and configuration integrity.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestConfiguration:
    def test_tasks_list_correct(self):
        from config.settings import TASKS
        assert set(TASKS) == {"heart_disease", "diabetes", "breast_cancer"}

    def test_dataset_info_keys(self):
        from config.settings import DATASET_INFO, TASKS
        for task in TASKS:
            info = DATASET_INFO[task]
            for key in ["name", "source", "url", "target_column",
                        "positive_class", "positive_label", "negative_label",
                        "description", "limitations"]:
                assert key in info, f"Missing key '{key}' for task '{task}'"

    def test_directories_created(self):
        from config.settings import (
            DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR,
            REPORTS_DIR, FIGURES_DIR,
        )
        for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
            assert d.exists(), f"Directory not created: {d}"

    def test_random_seed_is_int(self):
        from config.settings import RANDOM_SEED
        assert isinstance(RANDOM_SEED, int)

    def test_cv_folds_positive(self):
        from config.settings import CV_FOLDS
        assert CV_FOLDS >= 2

    def test_test_size_valid(self):
        from config.settings import TEST_SIZE
        assert 0.0 < TEST_SIZE < 1.0


class TestDataLoaderImports:
    def test_data_loader_importable(self):
        import src.data_loader  # noqa: F401

    def test_breast_cancer_loads(self):
        from src.data_loader import load_breast_cancer_data
        df = load_breast_cancer_data()
        assert len(df) > 0
        assert "target" in df.columns
        assert df["target"].nunique() == 2

    def test_breast_cancer_target_values(self):
        from src.data_loader import load_breast_cancer_data
        df = load_breast_cancer_data()
        assert set(df["target"].unique()).issubset({0, 1})


class TestModelArtifactHelpers:
    def test_load_best_pipeline_raises_on_missing(self, tmp_path, monkeypatch):
        import config.settings as settings
        monkeypatch.setattr(settings, "MODELS_DIR", tmp_path)
        from src.train import load_best_pipeline
        with pytest.raises(FileNotFoundError):
            load_best_pipeline("heart_disease")

    def test_load_metadata_raises_on_missing(self, tmp_path, monkeypatch):
        import config.settings as settings
        monkeypatch.setattr(settings, "MODELS_DIR", tmp_path)
        from src.train import load_metadata
        with pytest.raises(FileNotFoundError):
            load_metadata("heart_disease")
