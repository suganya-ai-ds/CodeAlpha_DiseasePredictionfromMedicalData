"""
MEDSCOPE AI — Data Validation
Runs a suite of quality checks on any loaded dataset and returns a structured report.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Structured record of all data-quality findings."""
    task: str
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False
        logger.error("[%s] ERROR — %s", self.task, msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning("[%s] WARNING — %s", self.task, msg)

    def add_info(self, key: str, value: Any) -> None:
        self.info[key] = value

    def summary(self) -> str:
        lines = [f"=== Validation Report: {self.task} ==="]
        lines.append(f"Status : {'PASSED' if self.passed else 'FAILED'}")
        for e in self.errors:
            lines.append(f"  [ERROR]   {e}")
        for w in self.warnings:
            lines.append(f"  [WARNING] {w}")
        for k, v in self.info.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def validate_dataset(
    df: pd.DataFrame,
    task: str,
    target_col: str,
    required_features: Optional[List[str]] = None,
    expected_target_values: Optional[List[Any]] = None,
) -> ValidationReport:
    """
    Run a comprehensive quality check on *df*.

    Parameters
    ----------
    df               : The loaded DataFrame.
    task             : Task name for reporting.
    target_col       : Name of the target column.
    required_features: Feature columns that must be present.
    expected_target_values: All valid values for the target.

    Returns
    -------
    ValidationReport
    """
    rpt = ValidationReport(task=task)

    # ── 1. Empty dataset ──────────────────────────────────────────────────────
    if df.empty:
        rpt.add_error("Dataset is empty.")
        return rpt  # Nothing more to check

    rpt.add_info("rows", len(df))
    rpt.add_info("columns", list(df.columns))
    rpt.add_info("shape", df.shape)

    # ── 2. Target column presence ─────────────────────────────────────────────
    if target_col not in df.columns:
        rpt.add_error(f"Target column '{target_col}' is missing.")
        return rpt

    # ── 3. Required feature columns ───────────────────────────────────────────
    if required_features:
        missing = [c for c in required_features if c not in df.columns]
        if missing:
            rpt.add_error(f"Required feature columns missing: {missing}")

    # ── 4. Duplicate rows ─────────────────────────────────────────────────────
    n_dupes = df.duplicated().sum()
    rpt.add_info("duplicate_rows", int(n_dupes))
    if n_dupes > 0:
        rpt.add_warning(f"{n_dupes} duplicate row(s) found.")

    # ── 5. Missing values ─────────────────────────────────────────────────────
    missing_counts = df.isnull().sum()
    missing_summary = missing_counts[missing_counts > 0].to_dict()
    rpt.add_info("missing_values", missing_summary)
    missing_target = int(df[target_col].isnull().sum())
    if missing_target > 0:
        rpt.add_error(f"Target column has {missing_target} missing value(s).")
    if missing_summary:
        rpt.add_warning(f"Missing values detected in columns: {missing_summary}")

    # ── 6. Target distribution ────────────────────────────────────────────────
    target_dist = df[target_col].value_counts(dropna=False).to_dict()
    rpt.add_info("target_distribution", target_dist)
    n_classes = df[target_col].nunique(dropna=True)
    if n_classes < 2:
        rpt.add_error(f"Target column has only {n_classes} unique value(s) — cannot train a classifier.")
    elif n_classes > 20:
        rpt.add_warning(f"Target column has {n_classes} unique values — confirm this is a classification task.")

    # ── 7. Unexpected target values ───────────────────────────────────────────
    if expected_target_values is not None:
        unexpected = set(df[target_col].dropna().unique()) - set(expected_target_values)
        if unexpected:
            rpt.add_error(f"Unexpected target values: {unexpected}")

    # ── 8. Infinite values (numeric columns) ──────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    inf_counts = {
        c: int(np.isinf(df[c]).sum())
        for c in num_cols
        if np.isinf(df[c]).sum() > 0
    }
    rpt.add_info("infinite_value_columns", inf_counts)
    if inf_counts:
        rpt.add_warning(f"Infinite values found in: {list(inf_counts.keys())}")

    # ── 9. Constant / zero-variance features ──────────────────────────────────
    constant_cols = [c for c in num_cols if df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        rpt.add_warning(f"Constant (zero-variance) numeric columns: {constant_cols}")
    rpt.add_info("constant_columns", constant_cols)

    # ── 10. Data-type summary ─────────────────────────────────────────────────
    dtype_summary = {c: str(df[c].dtype) for c in df.columns}
    rpt.add_info("dtypes", dtype_summary)

    # ── 11. Class imbalance warning ───────────────────────────────────────────
    if n_classes == 2:
        counts = list(target_dist.values())
        ratio = max(counts) / max(min(counts), 1)
        rpt.add_info("imbalance_ratio", round(ratio, 2))
        if ratio > 3:
            rpt.add_warning(
                f"Class imbalance detected (ratio ≈ {ratio:.1f}:1). "
                "Consider stratified splitting and appropriate metrics."
            )

    logger.info("[%s] Validation complete. Passed=%s", task, rpt.passed)
    return rpt
