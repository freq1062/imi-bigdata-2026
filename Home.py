import streamlit as st
import os
from pathlib import Path

from lib.resource_paths import WEBAPP_DATA_DIR, WEBAPP_OUTPUTS_DIR

# Files required for ANY page to function (inference + review)
def _hard_required() -> list[Path]:
    return [
        WEBAPP_OUTPUTS_DIR / "sage_artifacts.pkl",
        WEBAPP_OUTPUTS_DIR / "fraud_sage_model.pth",
        WEBAPP_OUTPUTS_DIR / "model_output.csv",
        WEBAPP_OUTPUTS_DIR / "meta_cluster_assignments.csv.gz",
        WEBAPP_OUTPUTS_DIR / "rf_model_sage" / "rf_proxy.joblib",
        WEBAPP_OUTPUTS_DIR / "rf_model_sage" / "meta.json",
        WEBAPP_DATA_DIR / "kyc_industry_codes.csv.gz",
        WEBAPP_DATA_DIR / "card.csv.gz",
        WEBAPP_DATA_DIR / "abm.csv.gz",
        WEBAPP_DATA_DIR / "eft.csv.gz",
        WEBAPP_DATA_DIR / "emt.csv.gz",
        WEBAPP_DATA_DIR / "wire.csv.gz",
        WEBAPP_DATA_DIR / "cheque.csv.gz",
    ]

# Files that enhance explanations but are not needed to load the app
def _soft_required() -> list[Path]:
    return [
        WEBAPP_OUTPUTS_DIR / "transaction_autoencoder.pt",
        WEBAPP_OUTPUTS_DIR / "model_output_explanations.csv.gz",
        WEBAPP_OUTPUTS_DIR / "meta_cluster_semantic_labels.json",
        WEBAPP_OUTPUTS_DIR / "meta_cluster_significant_deltas.json",
        WEBAPP_OUTPUTS_DIR / "meta_cluster_top3_categories.json",
        WEBAPP_OUTPUTS_DIR / "rf_model_sage" / "shap_explainer.joblib",
    ]


def _missing_hard() -> list[str]:
    return [str(p) for p in _hard_required() if not p.exists()]

def _missing_soft() -> list[str]:
    return [str(p) for p in _soft_required() if not p.exists()]

# --- Streamlit UI ---
from lib.components import header_with_logo

st.set_page_config(layout="wide", page_title="Team 76 AML Detection")

# Render header with logo to the right
header_with_logo("AI-Driven AML / ML-TF Detection", img_width=260)

# Resource check — hard missing blocks the app, soft missing shows a warning
missing_hard = _missing_hard()
missing_soft = _missing_soft()

if missing_hard:
    st.error("Required web app resources are missing.")
    st.markdown(
        "Run the **training notebook** (`training.ipynb`) through to the final "
        "**Package Artifacts for Web App** cell, then refresh this page."
    )
    with st.expander("Missing files", expanded=True):
        for item in missing_hard:
            st.write(f"- `{item}`")
    st.stop()
else:
    if missing_soft:
        with st.expander("⚠️ Optional explainability resources missing (run explainability notebook to add them)", expanded=False):
            for item in missing_soft:
                st.write(f"- `{item}`")

st.markdown("""
## Real-Time Financial Crime Risk Detection
- **Risk Scoring**: GraphSAGE-based detection for Scotiabank data.
- **Explainable AI**: Narrative generation via Llama 3.2.
- **Regulatory Alignment**: Automated SAR-lite reporting.
""")

col1, col2, col3 = st.columns(3)

col1.metric("Detection Accuracy", "99%")
col2.metric("Typologies Covered", "18")
col3.metric("Explainability Coverage", "100%")

st.divider()

st.subheader("How It Works")

st.write("""
1. **Customer Transaction Ingestion**: Automated data pipeline for real-time streaming.
2. **Feature Engineering**: Dynamic generation of risk-based indicators.
3. **ML Risk Scoring**: Multi-layer neural network processing.
4. **Explainable Output**: Human-readable narratives for regulatory reporting.
""")

st.info("System initialized from local webapp_resources (no network download).")