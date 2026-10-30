"""
MEDSCOPE AI — Test Suite: Data Validation
Tests the data_validation module with small deterministic fixtures.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_validation import validate_dataset


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_binary_df(n=50, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(30, 80, n).astype(float),
        "bp":  rng.integers(60, 180, n).astype(float),
        "chol": rng.integers(150, 350, n).astype(float),
        "target": rng.integers(0, 2, n),
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestValidateDataset:
    def test_passes_clean_dataframe(self):
        df = _make_binary_df()
        rpt = validate_dataset(df, task="test_task", target_col="target")
        assert rpt.passed, rpt.summary()

    def test_fails_empty_dataframe(self):
        df = pd.DataFrame()
        rpt = validate_dataset(df, task="empty_task", target_col="target")
        assert not rpt.passed
        assert any("empty" in e.lower() for e in rpt.errors)

    def test_fails_missing_target_column(self):
        df = _make_binary_df()
        rpt = validate_dataset(df, task="test_task", target_col="nonexistent")
        assert not rpt.passed

    def test_warns_on_duplicates(self):
        df = _make_binary_df()
        df = pd.concat([df, df.iloc[:5]]).reset_index(drop=True)
        rpt = validate_dataset(df, task="test_task", target_col="target")
        assert any("duplicate" in w.lower() for w in rpt.warnings)

    def test_warns_on_missing_values(self):
        df = _make_binary_df()
        df.loc[0, "age"] = np.nan
        rpt = validate_dataset(df, task="test_task", target_col="target")
        assert any("missing" in w.lower() for w in rpt.warnings)

    def test_fails_target_with_missing_values(self):
        df = _make_binary_df()
        df.loc[0, "target"] = np.nan
        rpt = validate_dataset(df, task="test_task", target_col="target")
        assert not rpt.passed

    def test_warns_on_class_imbalance(self):
        df = _make_binary_df(n=100)
        df["target"] = 0
        df.loc[:5, "target"] = 1  # 94:6 ratio
        rpt = validate_dataset(df, task="test_task", target_col="target")
        assert any("imbalance" in w.lower() or "ratio" in w.lower() for w in rpt.warnings)

    def test_fails_single_class_target(self):
        df = _make_binary_df()
        df["target"] = 0  # only one class
        rpt = validate_dataset(df, task="test_task", target_col="target")
        assert not rpt.passed

    def test_warns_on_infinite_values(self):
        df = _make_binary_df()
        df.loc[0, "age"] = np.inf
        rpt = validate_dataset(df, task="test_task", target_col="target")
        assert any("infinite" in w.lower() or "inf" in w.lower() for w in rpt.warnings)

    def test_reports_required_features_missing(self):
        df = _make_binary_df()
        rpt = validate_dataset(
            df, task="test_task", target_col="target",
            required_features=["age", "nonexistent_col"],
        )
        assert not rpt.passed

    def test_unexpected_target_values(self):
        df = _make_binary_df()
        df.loc[0, "target"] = 99
        rpt = validate_dataset(
            df, task="test_task", target_col="target",
            expected_target_values=[0, 1],
        )
        assert not rpt.passed

    def test_info_populated(self):
        df = _make_binary_df()
        rpt = validate_dataset(df, task="test_task", target_col="target")
        assert "rows" in rpt.info
        assert rpt.info["rows"] == len(df)
        assert "target_distribution" in rpt.info
