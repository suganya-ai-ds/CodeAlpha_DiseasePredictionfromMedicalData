"""
MEDSCOPE AI — Streamlit Application
Run from the MEDSCOPE_AI/ directory:
  streamlit run app.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure MEDSCOPE_AI/ is on sys.path when launched from any directory
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from config.settings import DATASET_INFO, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, TASKS
from src.preprocessing import FEATURE_SCHEMAS, FEATURE_UI_META, get_feature_columns
from src.predict import PredictionError, predict

logging.basicConfig(level=logging.WARNING)

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="MEDSCOPE AI — Clinical Data Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (Navy / Teal healthcare theme) ─────────────────────────────────
st.markdown("""
<style>
/* Imports */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root variables */
:root {
    --navy: #0d1b2a;
    --navy-mid: #1b2e45;
    --teal: #00b4d8;
    --teal-light: #48cae4;
    --white: #f8fafc;
    --text-muted: #94a3b8;
    --card-bg: #1e3050;
    --border: #2d4a6e;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--navy) !important;
    color: var(--white) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--navy-mid) !important;
    border-right: 1px solid var(--border);
}

/* Main header */
.medscope-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b3a5c 50%, #0a3d62 100%);
    border: 1px solid var(--teal);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.medscope-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0,180,216,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.medscope-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: var(--white);
    margin: 0;
    letter-spacing: -0.5px;
}
.medscope-header .subtitle {
    color: var(--teal-light);
    font-size: 1rem;
    margin-top: 0.25rem;
    font-weight: 400;
}
.medscope-badge {
    display: inline-block;
    background: rgba(0,180,216,0.15);
    border: 1px solid var(--teal);
    color: var(--teal-light);
    border-radius: 20px;
    padding: 0.15rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-right: 0.5rem;
    margin-top: 0.75rem;
}

/* KPI / metric cards */
.kpi-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    text-align: center;
}
.kpi-card .kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--teal-light);
}
.kpi-card .kpi-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.25rem;
}

/* Section card */
.section-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Alert cards */
.alert-info {
    background: rgba(0,180,216,0.1);
    border-left: 4px solid var(--teal);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.75rem 0;
    color: var(--white);
    font-size: 0.88rem;
}
.alert-warning {
    background: rgba(245,158,11,0.12);
    border-left: 4px solid var(--warning);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.75rem 0;
    color: var(--white);
    font-size: 0.88rem;
}
.alert-success {
    background: rgba(34,197,94,0.12);
    border-left: 4px solid var(--success);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.75rem 0;
    color: var(--white);
    font-size: 0.88rem;
}
.alert-danger {
    background: rgba(239,68,68,0.12);
    border-left: 4px solid var(--danger);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.75rem 0;
    color: var(--white);
    font-size: 0.88rem;
}

/* Prediction result */
.pred-result-positive {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.pred-result-negative {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05));
    border: 1px solid rgba(34,197,94,0.4);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.pred-label {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.pred-prob {
    font-size: 0.9rem;
    color: var(--text-muted);
}

/* Streamlit widget overrides */
.stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label {
    color: var(--white) !important;
    font-size: 0.85rem !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Table */
.dataframe { font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dataset_cached(task: str) -> pd.DataFrame:
    from src.data_loader import load_dataset
    return load_dataset(task)


@st.cache_resource(show_spinner=False)
def load_pipeline_cached(task: str):
    from src.train import load_best_pipeline
    return load_best_pipeline(task)


def model_is_trained(task: str) -> bool:
    return (MODELS_DIR / task / "best_pipeline.joblib").exists()


def load_metadata_safe(task: str) -> dict:
    path = MODELS_DIR / task / "metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_metrics_safe(task: str) -> pd.DataFrame | None:
    path = REPORTS_DIR / task / "metrics_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def fig_path(task: str, name: str) -> Path | None:
    p = FIGURES_DIR / task / f"{name}.png"
    return p if p.exists() else None


# ── Sidebar navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1rem 0;'>
        <div style='color:#00b4d8;font-size:1.2rem;font-weight:700;'>🏥 MEDSCOPE AI</div>
        <div style='color:#94a3b8;font-size:0.75rem;'>Clinical Data Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Dataset Explorer", "🔬 Disease Prediction", "📈 Model Performance",
         "💡 Explainability", "🔍 Data Quality", "📖 About"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        "<div style='color:#64748b;font-size:0.7rem;'>⚠️ Educational prototype only.<br>"
        "Not for clinical use.</div>",
        unsafe_allow_html=True,
    )


# ── Page: Home ───────────────────────────────────────────────────────────────

if page == "🏠 Home":
    st.markdown("""
    <div class="medscope-header">
        <h1>🏥 MEDSCOPE AI</h1>
        <div class="subtitle">Clinical Data Intelligence &amp; Disease Prediction Research Platform</div>
        <div style="margin-top:0.75rem;">
            <span class="medscope-badge">❤️ Heart Disease</span>
            <span class="medscope-badge">🩺 Diabetes</span>
            <span class="medscope-badge">🔬 Breast Cancer</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        trained = sum(model_is_trained(t) for t in TASKS)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{trained}/{len(TASKS)}</div>
            <div class="kpi-label">Models Trained</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-value">3</div>
            <div class="kpi-label">Disease Datasets</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-value">LR · SVM · RF</div>
            <div class="kpi-label">Algorithms</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("What is MEDSCOPE AI?")
        st.markdown("""
MEDSCOPE AI is an **educational machine-learning platform** that demonstrates structured
clinical data analysis and disease outcome prediction across three independent datasets.

**Supported tasks:**
- **Heart Disease** — Cleveland UCI dataset; binary classification (disease present/absent)
- **Diabetes** — Early Stage Risk Prediction (UCI #529); symptom-based survey
- **Breast Cancer** — Wisconsin Diagnostic dataset (sklearn); malignant vs benign

**Core features:**
- Independent ML pipelines for each disease (no feature cross-contamination)
- Multiple classifiers with cross-validated model selection
- Comprehensive evaluation: accuracy, F1, ROC-AUC, specificity, confusion matrix
- Feature importance and explainability visualisations
- Interactive prediction demonstration with schema-validated inputs
        """)

    with col_right:
        st.subheader("Methodology")
        st.markdown("""
| Step | Method |
|------|--------|
| Preprocessing | Scikit-learn ColumnTransformer |
| Validation | Stratified 5-fold CV |
| Model selection | Best CV ROC-AUC |
| Explainability | Gini / Coefficients / Permutation |
| Evaluation | Held-out test set (20%) |
        """)
        st.markdown("""
<div class="alert-warning">
⚠️ <strong>Responsible Use Notice</strong><br>
This is a research prototype. Predictions are for educational demonstration only.
They do not constitute a clinical diagnosis. Always consult a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)

    if not any(model_is_trained(t) for t in TASKS):
        st.markdown("""
<div class="alert-info">
ℹ️ <strong>Models not yet trained.</strong>
Run <code>python train_all.py</code> from the <code>MEDSCOPE_AI/</code> directory to train all models.
</div>
""", unsafe_allow_html=True)


# ── Page: Dataset Explorer ───────────────────────────────────────────────────

elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")
    task = st.selectbox(
        "Select disease dataset",
        TASKS,
        format_func=lambda t: DATASET_INFO[t]["name"],
    )
    info = DATASET_INFO[task]

    with st.expander("Dataset Metadata", expanded=True):
        mc1, mc2 = st.columns(2)
        mc1.markdown(f"**Name:** {info['name']}")
        mc1.markdown(f"**Source:** [{info['url']}]({info['url']})")
        mc1.markdown(f"**License:** {info.get('license','N/A')}")
        mc2.markdown(f"**Target column:** `{info['target_column']}`")
        mc2.markdown(f"**Positive class:** {info['positive_label']}")
        mc2.markdown(f"**Negative class:** {info['negative_label']}")
        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**Limitations:** _{info['limitations']}_")

    try:
        with st.spinner("Loading dataset…"):
            df = load_dataset_cached(task)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing cells", int(df.isnull().sum().sum()))
        c4.metric("Duplicates", int(df.duplicated().sum()))

        st.subheader("Sample Rows")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().round(3), use_container_width=True)

        st.subheader("Target Distribution")
        target_col = info["target_column"]
        dist = df[target_col].value_counts()
        st.bar_chart(dist)

    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error loading dataset: {e}")


# ── Page: Disease Prediction ──────────────────────────────────────────────────

elif page == "🔬 Disease Prediction":
    st.title("🔬 Disease Prediction Demo")
    st.markdown("""
<div class="alert-warning">
⚠️ <strong>Educational demonstration only.</strong>
This model has not been clinically validated. Never use model output as a substitute
for professional medical advice, diagnosis, or treatment.
</div>
""", unsafe_allow_html=True)

    task = st.selectbox(
        "Select disease task",
        TASKS,
        format_func=lambda t: DATASET_INFO[t]["name"],
    )

    if not model_is_trained(task):
        st.error(
            f"No trained model found for **{task}**. "
            f"Run `python train_all.py --task {task}` first."
        )
        st.stop()

    meta = load_metadata_safe(task)
    st.info(f"Model: **{meta.get('algorithm','unknown')}** | "
            f"CV AUC: **{meta.get('cv_roc_auc','N/A')}** | "
            f"Trained: **{meta.get('trained_at','N/A')[:10]}**")

    schema = FEATURE_SCHEMAS[task]
    ui_meta = FEATURE_UI_META.get(task, {})
    feature_cols = get_feature_columns(task)

    st.subheader("Enter Patient Features")
    st.caption("All fields are required. Values are for demonstration purposes only.")

    with st.form("prediction_form"):
        input_vals: dict = {}
        cols_per_row = 3
        feat_chunks = [feature_cols[i:i+cols_per_row] for i in range(0, len(feature_cols), cols_per_row)]

        for chunk in feat_chunks:
            row_cols = st.columns(len(chunk))
            for col_widget, feat in zip(row_cols, chunk):
                meta_feat = ui_meta.get(feat, {})
                label = meta_feat.get("label", feat)
                ftype = meta_feat.get("type", "numeric")

                with col_widget:
                    if ftype == "categorical":
                        options_map = meta_feat.get("options", {})
                        keys = list(options_map.keys())
                        chosen = st.selectbox(
                            label,
                            keys,
                            format_func=lambda k, m=options_map: m.get(k, k),
                            key=f"inp_{feat}",
                        )
                        input_vals[feat] = chosen
                    else:
                        unit = meta_feat.get("unit", "")
                        mn = meta_feat.get("min", 0.0)
                        mx = meta_feat.get("max", 1000.0)
                        default = round((mn + mx) / 2, 1)
                        val = st.number_input(
                            f"{label} ({unit})" if unit else label,
                            min_value=float(mn),
                            max_value=float(mx),
                            value=float(default),
                            step=1.0 if (mx - mn) > 5 else 0.1,
                            key=f"inp_{feat}",
                        )
                        input_vals[feat] = val

        submitted = st.form_submit_button("🔬 Run Prediction", use_container_width=True)

    if submitted:
        with st.spinner("Running model inference…"):
            try:
                pipeline = load_pipeline_cached(task)
                result = predict(task, input_vals, pipeline=pipeline)
                pred_class = result["predicted_class"]
                pred_label = result["predicted_label"]
                probability = result["probability"]

                # Determine if positive
                info = DATASET_INFO[task]
                pos_class = info["positive_class"]
                if isinstance(pos_class, str):
                    pos_class = 1
                is_positive = pred_class == pos_class

                css_class = "pred-result-positive" if is_positive else "pred-result-negative"
                icon = "⚠️" if is_positive else "✅"
                prob_text = f"Model score: {probability:.1%}" if probability is not None else ""

                st.markdown(f"""
<div class="{css_class}">
    <div style="font-size:2rem;">{icon}</div>
    <div class="pred-label">{pred_label}</div>
    <div class="pred-prob">{prob_text}</div>
</div>
""", unsafe_allow_html=True)

                st.markdown("""
<div class="alert-info">
ℹ️ <strong>Interpretation note:</strong>
The model score (if shown) reflects the classifier's output — not a calibrated clinical probability.
It depends on the dataset's class distribution, feature quality, and model calibration.
This result is <strong>for educational purposes only</strong>.
</div>
""", unsafe_allow_html=True)

            except PredictionError as e:
                st.error(f"Input validation error: {e}")
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    if st.button("↩ Reset form"):
        st.rerun()


# ── Page: Model Performance ───────────────────────────────────────────────────

elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    task = st.selectbox(
        "Select disease task",
        TASKS,
        format_func=lambda t: DATASET_INFO[t]["name"],
    )

    if not model_is_trained(task):
        st.warning(f"No model artifacts found for **{task}**. Train the models first.")
        st.stop()

    meta = load_metadata_safe(task)
    metrics_df = load_metrics_safe(task)

    if meta:
        st.subheader("Training Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Algorithm", meta.get("algorithm", "N/A"))
        c2.metric("CV ROC-AUC", f"{meta.get('cv_roc_auc', 0):.4f}")
        c3.metric("Train Samples", meta.get("train_samples", "N/A"))
        c4.metric("Test Samples", meta.get("test_samples", "N/A"))

    if metrics_df is not None:
        st.subheader("Model Comparison Table")
        display_cols = ["model", "accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "cv_score_mean"]
        display_cols = [c for c in display_cols if c in metrics_df.columns]
        st.dataframe(metrics_df[display_cols].set_index("model").round(4), use_container_width=True)

    # Figures
    st.subheader("Visualisations")
    tab1, tab2, tab3 = st.tabs(["ROC Curve", "Precision-Recall", "Model Comparison"])

    with tab1:
        p = fig_path(task, "roc_curves")
        if p:
            st.image(str(p))
        else:
            st.info("ROC curve not yet generated. Run training first.")

    with tab2:
        p = fig_path(task, "pr_curves")
        if p:
            st.image(str(p))
        else:
            st.info("Precision-Recall curve not yet generated.")

    with tab3:
        p = fig_path(task, "model_comparison")
        if p:
            st.image(str(p))
        else:
            st.info("Comparison chart not yet generated.")

    st.subheader("Confusion Matrices")
    if metrics_df is not None:
        for model_name in metrics_df["model"].tolist():
            p = fig_path(task, f"cm_{model_name}")
            if p:
                with st.expander(f"Confusion Matrix — {model_name}"):
                    st.image(str(p))


# ── Page: Explainability ──────────────────────────────────────────────────────

elif page == "💡 Explainability":
    st.title("💡 Explainability & Feature Importance")

    st.markdown("""
<div class="alert-info">
ℹ️ Feature importance indicates which features most influenced model predictions —
not medical causation. Correlation does not imply clinical relevance.
Results depend on the dataset and model.
</div>
""", unsafe_allow_html=True)

    task = st.selectbox(
        "Select disease task",
        TASKS,
        format_func=lambda t: DATASET_INFO[t]["name"],
    )

    if not model_is_trained(task):
        st.warning("No model artifacts found. Train the models first.")
        st.stop()

    meta = load_metadata_safe(task)
    best_model = meta.get("algorithm", "best_pipeline")

    st.subheader(f"Feature Importance — {best_model} ({task.replace('_',' ').title()})")
    p = fig_path(task, f"feature_importance_{best_model}")
    if p:
        st.image(str(p))
    else:
        st.info("Feature importance chart not yet generated. Run training first.")

    p_shap = fig_path(task, f"shap_summary_{best_model}")
    if p_shap:
        st.subheader("SHAP Summary Plot")
        st.image(str(p_shap))

    st.subheader("Interpretation Notes")
    clf_notes = {
        "logistic_regression": (
            "Logistic Regression coefficients represent log-odds ratios. "
            "Bars show absolute magnitudes. Positive coefficients increase predicted probability; "
            "negative coefficients decrease it. Coefficients are influenced by feature scale."
        ),
        "random_forest": (
            "Gini impurity reduction measures how much each feature decreases impurity "
            "across all trees. Higher values indicate a stronger split contribution. "
            "This is averaged across all trees in the ensemble."
        ),
        "svm": (
            "SVM models do not inherently provide simple feature importance. "
            "Permutation importance measures the drop in model accuracy when a "
            "feature's values are randomly shuffled, indicating reliance on that feature."
        ),
    }
    note = clf_notes.get(best_model, "Feature importances reflect model-specific characteristics.")
    st.markdown(f"**{best_model.replace('_',' ').title()}:** {note}")

    st.markdown("""
**General disclaimer:** Feature importance does not establish causation.
A high importance score means the feature was predictive in this dataset —
not that it is a medical cause of the disease.
    """)


# ── Page: Data Quality ────────────────────────────────────────────────────────

elif page == "🔍 Data Quality":
    st.title("🔍 Data Quality Reports")
    task = st.selectbox(
        "Select disease task",
        TASKS,
        format_func=lambda t: DATASET_INFO[t]["name"],
    )

    report_path = REPORTS_DIR / task / "data_quality.json"
    if not report_path.exists():
        st.info(
            "Data quality report not yet generated. "
            "Run `python train_all.py` to generate reports."
        )
        # Try to show live validation
        try:
            with st.spinner("Running live validation…"):
                from src.data_loader import load_dataset
                from src.data_validation import validate_dataset
                df = load_dataset(task)
                info = DATASET_INFO[task]
                rpt = validate_dataset(df, task=task, target_col=info["target_column"])

            st.subheader("Live Validation Results")
            if rpt.passed:
                st.success("✅ Dataset passed all validation checks.")
            else:
                st.error("❌ Validation issues detected.")
            for e in rpt.errors:
                st.markdown(f'<div class="alert-danger">❌ {e}</div>', unsafe_allow_html=True)
            for w in rpt.warnings:
                st.markdown(f'<div class="alert-warning">⚠️ {w}</div>', unsafe_allow_html=True)
            for k, v in rpt.info.items():
                if k not in ("dtypes", "columns"):
                    st.markdown(f"**{k}:** `{v}`")
        except Exception as ex:
            st.error(f"Could not load dataset: {ex}")
    else:
        with open(report_path) as f:
            report_data = json.load(f)

        st.subheader("Data Quality Summary")
        if report_data.get("passed"):
            st.success("✅ Dataset passed all validation checks.")
        else:
            st.error("❌ Validation issues detected.")

        for e in report_data.get("errors", []):
            st.markdown(f'<div class="alert-danger">❌ {e}</div>', unsafe_allow_html=True)
        for w in report_data.get("warnings", []):
            st.markdown(f'<div class="alert-warning">⚠️ {w}</div>', unsafe_allow_html=True)

        info_data = report_data.get("info", {})
        col1, col2, col3, col4 = st.columns(4)
        shape = info_data.get("shape", [0, 0])
        col1.metric("Rows", shape[0] if isinstance(shape, list) else "N/A")
        col2.metric("Columns", shape[1] if isinstance(shape, list) else "N/A")
        col3.metric("Duplicates", info_data.get("duplicate_rows", "N/A"))
        col4.metric("Imbalance ratio", info_data.get("imbalance_ratio", "N/A"))

        missing = info_data.get("missing_values", {})
        if missing:
            st.subheader("Missing Values")
            st.dataframe(pd.Series(missing, name="Missing Count").to_frame(), use_container_width=True)
        else:
            st.success("No missing values detected.")

        target_dist = info_data.get("target_distribution", {})
        if target_dist:
            st.subheader("Target Distribution")
            st.bar_chart(pd.Series(target_dist))

        st.caption(f"Report generated: {report_data.get('generated_at','N/A')}")


# ── Page: About ───────────────────────────────────────────────────────────────

elif page == "📖 About":
    st.title("📖 Project Documentation")

    st.subheader("MEDSCOPE AI — Overview")
    st.markdown("""
**MEDSCOPE AI** is an educational, portfolio-grade machine-learning project built to
demonstrate responsible clinical data analysis and disease prediction using structured datasets.

It was developed as an internship project showcasing ML engineering, data science,
and professional software development practices.
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["Architecture", "Datasets", "Limitations", "Tech Stack"])

    with tab1:
        st.markdown("""
### Architecture
```
MEDSCOPE_AI/
├── app.py                  # Streamlit application
├── train_all.py            # Training orchestrator
├── config/settings.py      # Central configuration
├── src/
│   ├── data_loader.py      # Dataset loading (local/network)
│   ├── data_validation.py  # Quality checks
│   ├── preprocessing.py    # Feature schemas & pipelines
│   ├── train.py            # Model training & GridSearch
│   ├── evaluate.py         # Metrics & figure generation
│   ├── predict.py          # Inference & schema validation
│   ├── explainability.py   # Feature importance
│   └── reporting.py        # Report generation
├── models/
│   ├── heart_disease/      # Per-task model artifacts
│   ├── diabetes/
│   └── breast_cancer/
├── outputs/
│   ├── figures/            # Charts (PNG)
│   └── reports/            # CSV & text reports
└── tests/                  # Automated test suite
```
        """)

    with tab2:
        for task in TASKS:
            info = DATASET_INFO[task]
            st.markdown(f"""
**{info['name']}**
- Source: [{info['url']}]({info['url']})
- License: {info.get('license','N/A')}
- Target: `{info['target_column']}` → {info['positive_label']} / {info['negative_label']}
- {info['description']}
            """)
            st.divider()

    with tab3:
        st.markdown("""
### Known Limitations

1. **Dataset scope** — Models trained on specific, potentially non-representative cohorts.
2. **Not clinically validated** — No external validation or regulatory review performed.
3. **No real-world data** — Models have not been evaluated on prospective patient data.
4. **Feature limitations** — Symptom-based features (diabetes dataset) are self-reported.
5. **Calibration** — Model scores are not calibrated probabilities; interpret with caution.
6. **Population bias** — Performance may differ for populations not represented in training data.
7. **Static model** — Model does not update or adapt to new data without retraining.

### Responsible Use
- Do not use for clinical decision-making.
- Do not share patient-identifying information with this application.
- Results are for educational and research demonstration only.
- Real-world deployment would require clinical validation, regulatory review,
  privacy safeguards, and institutional oversight.
        """)

    with tab4:
        st.markdown("""
| Component | Technology |
|-----------|-----------|
| Language | Python 3.13 |
| Web app | Streamlit 1.66 |
| ML framework | Scikit-learn 1.9 |
| Data processing | Pandas 3.1, NumPy 2.5 |
| Visualisation | Matplotlib 3.11, Plotly 7.1 |
| Model serialisation | Joblib 1.6 |
| Testing | Pytest 9.1 |
| Explainability | Permutation importance, Gini (SHAP: optional) |
        """)

    st.divider()
    st.caption(
        "MEDSCOPE AI is an educational prototype. "
        "It does not constitute medical advice, diagnosis, or clinical guidance. "
        "Always consult a qualified healthcare professional."
    )
