"""
MEDSCOPE AI — Central Configuration
All paths and constants resolve relative to the project root (MEDSCOPE_AI/).
"""
from pathlib import Path

# ── Resolve project root ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # …/MEDSCOPE_AI

# ── Directory layout ──────────────────────────────────────────────────────────
DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"
FIGURES_DIR = ROOT / "outputs" / "figures"

# ── Create directories if missing ─────────────────────────────────────────────
for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Supported disease tasks ───────────────────────────────────────────────────
TASKS = ["heart_disease", "diabetes", "breast_cancer"]

# ── Random seed (used everywhere for reproducibility) ─────────────────────────
RANDOM_SEED = 42

# ── Train / test split ────────────────────────────────────────────────────────
TEST_SIZE = 0.20

# ── Cross-validation folds ────────────────────────────────────────────────────
CV_FOLDS = 5

# ── Dataset metadata ─────────────────────────────────────────────────────────
DATASET_INFO = {
    "heart_disease": {
        "name": "UCI Heart Disease (Cleveland)",
        "source": "scikit-learn / UCI ML Repository",
        "url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "license": "CC BY 4.0",
        "target_column": "target",
        "positive_class": 1,
        "positive_label": "Heart Disease Present",
        "negative_label": "No Heart Disease",
        "description": (
            "303 patients (Cleveland subset). Target: presence of heart disease "
            "(original values 0-4 binarised: 0 = absent, 1-4 = present)."
        ),
        "limitations": (
            "Single-centre cohort (Cleveland Clinic). May not generalise to other "
            "populations. Many real-world missing values replaced in common CSV "
            "distributions — verify the copy in use."
        ),
    },
    "diabetes": {
        "name": "Early Stage Diabetes Risk Prediction (UCI)",
        "source": "UCI ML Repository",
        "url": "https://archive.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset",
        "license": "CC BY 4.0",
        "target_column": "class",
        "positive_class": "Positive",
        "positive_label": "Diabetes Risk: Positive",
        "negative_label": "Diabetes Risk: Negative",
        "description": (
            "520 survey-based records from Sylhet Diabetes Hospital, Bangladesh. "
            "16 symptom-based binary/categorical features. "
            "Target: Positive / Negative diabetes classification."
        ),
        "limitations": (
            "Survey-based data from a single hospital; not a blood-test dataset. "
            "Features are self-reported symptoms, not clinical measurements."
        ),
    },
    "breast_cancer": {
        "name": "Wisconsin Diagnostic Breast Cancer (WDBC)",
        "source": "scikit-learn built-in (UCI)",
        "url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "license": "CC BY 4.0",
        "target_column": "target",
        "positive_class": 0,   # sklearn encodes malignant=0, benign=1
        "positive_label": "Malignant",
        "negative_label": "Benign",
        "description": (
            "569 cases, 30 numeric features computed from digitised FNA images. "
            "sklearn target: 0=malignant, 1=benign."
        ),
        "limitations": (
            "Features derived from a specific imaging protocol; performance may "
            "differ with other imaging systems or populations."
        ),
    },
}

# ── Model hyperparameter search spaces ───────────────────────────────────────
# Kept deliberately narrow to ensure fast, reproducible runs.
PARAM_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__max_iter": [1000],
        "classifier__solver": ["lbfgs"],
    },
    "svm": {
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["rbf", "linear"],
        "classifier__gamma": ["scale"],
    },
    "random_forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5],
    },
    "xgboost": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 6],
        "classifier__learning_rate": [0.05, 0.1],
    },
}
