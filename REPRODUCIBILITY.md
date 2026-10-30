# 🔬 MEDSCOPE AI — Reproducibility & Replication Guide

> **Project:** MEDSCOPE AI — Clinical Data Intelligence & Disease Prediction Research Platform  
> **Environment:** Python 3.13 / Windows (PowerShell) / Scikit-Learn 1.7+  
> **Target Audience:** Evaluators, Researchers, and Technical Assessors  

---

## 1. Environment Setup

### Prerequisites
- Python 3.10+ (Tested extensively on Python 3.13)
- Windows PowerShell, macOS zsh, or Linux bash

### Step 1: Clone or Navigate to the Directory
```powershell
cd "c:\Users\nandh\OneDrive\Desktop\Disease Prediction\MEDSCOPE_AI"
```

### Step 2: (Optional) Create & Activate a Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Core Dependencies
```powershell
python -m pip install -r requirements.txt
```

---

## 2. Deterministic Seeds & Split Specifications

To guarantee mathematical reproducibility across any machine, all random processes are pinned using a centralized constant in [`config/settings.py`](file:///c:/Users/nandh/OneDrive/Desktop/Disease%20Prediction/MEDSCOPE_AI/config/settings.py):

- **Random Seed:** `RANDOM_SEED = 42`
- **Train/Test Split:** `TEST_SIZE = 0.20` (80% train, 20% test)
- **Cross-Validation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Classifiers Seed:** Pinned to `random_state=42` across LogisticRegression, SVC, and RandomForest.

---

## 3. Training & Artifact Replication

To retrain all 3 disease pipelines from scratch, run:
```powershell
python train_all.py
```

Or run an individual task:
```powershell
python train_all.py --task heart_disease
python train_all.py --task diabetes
python train_all.py --task breast_cancer
```

### Generated Artifacts
Running the pipeline deterministically populates:
1. **Model Joblib Files** under `models/<task>/`
   - `best_pipeline.joblib`
   - `logistic_regression_pipeline.joblib`
   - `svm_pipeline.joblib`
   - `random_forest_pipeline.joblib`
   - `metadata.json`
2. **Metrics & Quality Reports** under `outputs/reports/<task>/`
   - `metrics_comparison.csv`
   - `data_quality.json`
   - `research_report.txt`
3. **Publication Figures** under `outputs/figures/<task>/`
   - `cm_*.png` (Confusion matrices)
   - `roc_curves.png`
   - `pr_curves.png`
   - `model_comparison.png`
   - `feature_importance_*.png`

---

## 4. Automated Verification & Testing

Execute the complete automated test suite (49 tests):
```powershell
python -m pytest tests/ -v
```

All 49 unit and integration tests verify:
- Data loader integrity & shape
- Missing-value warnings & error detection
- Data leakage prevention in preprocessing
- Preprocessor categorical encoding & numeric scaling
- Model artifact serialization & deserialization
- Prediction input schema validation & range checks
- Metric computation (Accuracy, Precision, Recall, F1, ROC-AUC, Specificity)

---

## 5. Running the Interactive Streamlit Platform

Launch the web application locally:
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`.
Navigate across all 7 platform sections:
1. **Overview & Mission**
2. **Dataset Explorer**
3. **Prediction Lab**
4. **Model Performance & Comparison**
5. **Explainability & Feature Importance**
6. **Data Quality & Audit**
7. **Documentation & Limitations**
