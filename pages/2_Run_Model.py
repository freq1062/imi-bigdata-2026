"""
2_Run_Model.py — Interactive AML risk assessment for new customers.

Workflow:
  1. Analyst enters customer KYC (age, income, tenure, type, etc.)
  2. Analyst adds transactions one at a time (amount, city, merchant category…)
  3. After each change: GraphSAGE scores the customer inductively, RF proxy
     generates SHAP-driven narrative, graph updates in real time.

The graph shows:
  • Customer node (colour = risk level)
  • Merchant-category hub nodes (blue) and city hub nodes (orange)
  • Edges carry amount + transaction type labels
"""

import sys
import os
import datetime as _dt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd
import numpy as np

from lib.components import CustomerInfoCard, RuleBasedNarrative, PrecomputedNarrative, GraphVisualizer
import requests
from lib.explainability import (
    FEATURE_LABELS,
    global_verdict_explainer,
    spatial_cluster_explainer,
    temporal_reconstruction_explainer,
)
from lib.resource_paths import resolve_data_path, resolve_output_path
from lib.model_utils import (
    load_artifacts,
    load_sage_model,
    load_lgb_bundle,
    score_new_customers,
    compute_txn_aggregates,
    compute_ae_risk_from_transactions,
)

st.set_page_config(
    page_title="Run AML Model",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("<style>h1 { color: #7b68ee !important; }</style>", unsafe_allow_html=True)
from lib.components import apply_sidebar_styles
apply_sidebar_styles()

# (sidebar logo removed — header displays logo on Home page)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (cached — loaded once per server session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading GraphSAGE artifacts…")
def _load_artifacts():
    return load_artifacts(resolve_output_path("sage_artifacts.pkl"))

@st.cache_resource(show_spinner="Loading GraphSAGE model…")
def _load_model():
    arts = _load_artifacts()
    return load_sage_model(resolve_output_path("fraud_sage_model.pth"), artifacts=arts)

@st.cache_resource(show_spinner="Loading LGB scoring bundle…")
def _load_lgb_bundle():
    return load_lgb_bundle(resolve_output_path())


def _try_load_all():
    try:
        arts = _load_artifacts()
        model = _load_model()
        lgb_bndl = _load_lgb_bundle()
        return arts, model, lgb_bndl, None
    except Exception as e:
        return None, None, None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "rm_customer_id": "NEW_CUSTOMER_001",
        "rm_age": 35,
        "rm_income": 65000,
        "rm_tenure": 730,
        "rm_is_biz": False,
        "rm_sales": 0,
        "rm_emp_count": 0,
        "rm_transactions": [],
        "rm_last_result": None,
        "rm_last_rf": None,
        "rm_temporal": None,
        "rm_spatial": None,
        "rm_verdict": None,
        "rm_llm_explanation": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Common option lists
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://132.145.111.57:11434")

TRANSACTION_TYPES = ["CARD", "ATM_WITHDRAWAL", "CHEQUE", "EFT", "EMT", "WIRE"]

# Minimum number of transactions required before the model is run.
# Below this threshold the behavioural signal is too sparse to be meaningful.
MIN_TRANSACTIONS = 3


@st.cache_data
def _load_industry_codes():
    """Load KYC industry codes from CSV and return sorted 'CODE – Label' strings."""
    path = resolve_data_path("kyc_industry_codes.csv.gz")
    df = pd.read_csv(path, compression="gzip", dtype=str)
    df = df.sort_values("industry").reset_index(drop=True)
    return [f"{row['industry_code']} \u2013 {row['industry']}" for _, row in df.iterrows()]

CANADIAN_CITIES = [
    "Toronto", "Montreal", "Vancouver", "Calgary", "Edmonton",
    "Ottawa", "Winnipeg", "Quebec City", "Hamilton", "Kitchener",
    "London", "Victoria", "Halifax", "Saskatoon", "Regina",
    "Unknown",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_record():
    txns = st.session_state["rm_transactions"]
    aggs = compute_txn_aggregates(txns)
    # Include both the merchant_category (industry code) AND the txn_type.
    # Transaction types (ATM_WITHDRAWAL, EFT, CHEQUE, EMT, WIRE) exist in the
    # training cat_map, so they always produce graph edges even when an industry
    # code is unseen.  Industry MCC codes that happen to be in cat_map also link.
    cats = list(
        {t["merchant_category"] for t in txns}
        | {t["txn_type"] for t in txns if t.get("txn_type")}
    )
    cities = list({t["city"].upper() for t in txns})
    return {
        "customer_id": st.session_state["rm_customer_id"],
        "age": float(st.session_state["rm_age"]),
        "income": float(st.session_state["rm_income"]),
        "tenure": float(st.session_state["rm_tenure"]),
        "sales": float(st.session_state["rm_sales"]),
        "emp_count": float(st.session_state["rm_emp_count"]),
        "is_biz": float(int(st.session_state["rm_is_biz"])),
        **aggs,
        "merchant_categories": cats,
        "cities": cities,
    }


# Fraud-positive LGB features (high value = more suspicious)
_LGB_FRAUD_POS = [
    "cluster_consensus_score", "dgi_anomaly_score", "mlp_fraud_prob",
    "anchor_proximity_score", "customer_ae_risk_norm", "knn_suspicious_share",
    "hdb_outlier_score", "knn_gold_fraud_count", "hdb_component_fraud_rate_labeled",
]
# Fraud-positive when LOW (closer to fraud centroid = more suspicious)
_LGB_FRAUD_INV = ["dist_to_fraud_centroid", "min_dist_to_fraud_anchor", "mean_dist_to_fraud_anchor"]


def _fetch_llm_explanation(result: dict, txns: list) -> str | None:
    """Call the Ollama endpoint and return a one-paragraph investigator explanation."""
    risk_score = float(result.get("risk_score", 0.0))
    risk_tier = str(result.get("risk_tier", "LOW"))
    lgb_feats: dict = st.session_state.get("rm_lgb_features") or {}

    # Pick top 3 most fraud-indicative feature values
    cands: list[tuple[str, float, float]] = []  # (feature, val, sort_key)
    for f, v in lgb_feats.items():
        try:
            val = float(v)
        except Exception:
            continue
        if f in _LGB_FRAUD_POS:
            cands.append((f, val, val))
        elif f in _LGB_FRAUD_INV:
            cands.append((f, val, -val))  # low distance = more suspicious
    cands.sort(key=lambda x: -x[2])
    top_feats_text = "; ".join(
        f"{FEATURE_LABELS.get(f, f)} ({val:.3f})"
        for f, val, _ in cands[:3]
    ) or "no feature data available"

    gnn_score = float(result.get("gnn_score", 0.0))
    prompt = (
        "You are an AML analyst assistant. Write a concise investigator-facing AML risk explanation in 4-6 sentences. "
        "Be specific and actionable — do not repeat the input data verbatim.\n"
        f"Risk tier: {risk_tier}\n"
        f"Model fraud probability: {risk_score:.4f} (GNN graph score: {gnn_score:.4f})\n"
        f"Top suspicious feature signals: {top_feats_text}\n"
        f"Number of transactions entered: {len(txns)}\n"
        "State what specific patterns drove the flag and what the analyst should verify or escalate next."
    )
    try:
        tags_r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=6)
        tags_r.raise_for_status()
        models = [m.get("name") for m in tags_r.json().get("models", []) if m.get("name")]
        if not models:
            return None
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": models[0], "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.2, "num_predict": 220}},
            timeout=45,
        )
        resp.raise_for_status()
        return str(resp.json().get("response", "")).strip() or None
    except Exception:
        return None


def _build_kyc_dict():
    return {
        "age": st.session_state["rm_age"],
        "income": st.session_state["rm_income"],
        "tenure": st.session_state["rm_tenure"],
        "is_biz": st.session_state["rm_is_biz"],
        "sales": st.session_state["rm_sales"],
        "emp_count": st.session_state["rm_emp_count"],
    }


def _run_scoring():
    """Score current customer and update session state.

    Scoring is skipped (and results cleared) when fewer than MIN_TRANSACTIONS
    transactions have been added — the behavioural signal is otherwise too
    sparse for meaningful inference.
    """
    txn_count = len(st.session_state["rm_transactions"])
    if txn_count < MIN_TRANSACTIONS:
        st.session_state["rm_last_result"] = None
        st.session_state["rm_last_rf"] = None
        st.session_state["rm_temporal"] = None
        st.session_state["rm_spatial"] = None
        st.session_state["rm_verdict"] = None
        return

    record = _build_record()
    # Compute live AE risk from actual transactions and inject into record so
    # score_new_customers can override the training-median default.
    txns_for_ae = st.session_state["rm_transactions"]
    record["_ae_risk_norm"] = compute_ae_risk_from_transactions(txns_for_ae)
    if model is not None and artifacts is not None:
        result_df = score_new_customers(
            [record], model, artifacts, lgb_bundle=lgb_bndl
        )
        row = result_df.iloc[0]
        st.session_state["rm_last_result"] = {
            k: v for k, v in row.to_dict().items() if k != "lgb_features"
        }
        lgb_features_for_shap: dict = row.get("lgb_features") or {}
        st.session_state["rm_lgb_features"] = lgb_features_for_shap
    else:
        st.session_state["rm_lgb_features"] = {}
        st.session_state["rm_last_result"] = {
            "customer_id": record["customer_id"],
            "risk_score": 0.0,
            "gnn_score": 0.0,
            "risk_tier": "LOW",
            "predicted_label": 0,
        }
        lgb_features_for_shap = {}
    # LGB SHAP explanation — compute real SHAP values from the LGB model
    txns = st.session_state["rm_transactions"]
    lgb_feats_for_shap: dict = st.session_state.get("rm_lgb_features") or {}
    shap_drivers: list[dict] = []
    if lgb_bndl is not None and lgb_feats_for_shap and artifacts is not None:
        try:
            import shap as _shap
            import numpy as _np_shap
            import pandas as _pd_shap
            _lgb_model = lgb_bndl[0]
            _feat_names: list = artifacts.get("lgb_feature_names", [])
            if _feat_names:
                _X = _pd_shap.DataFrame(
                    [[lgb_feats_for_shap.get(f, 0.0) for f in _feat_names]],
                    columns=_feat_names,
                )
                _explainer = _shap.TreeExplainer(_lgb_model)
                _sv = _explainer.shap_values(_X)
                if isinstance(_sv, _np_shap.ndarray) and _sv.ndim == 3:
                    _sv_fraud = _sv[0, :, 1]
                elif isinstance(_sv, list):
                    _sv_fraud = _sv[1][0] if len(_sv) > 1 else _sv[0][0]
                else:
                    _sv_fraud = _sv[0] if hasattr(_sv, "__getitem__") else _sv
                _pairs = sorted(
                    zip(_feat_names, _sv_fraud.tolist()),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:5]
                shap_drivers = [
                    {
                        "feature": _f,
                        "description": FEATURE_LABELS.get(_f, _f),
                        "shap_value": round(float(_s), 6),
                        "raw_value": round(float(lgb_feats_for_shap.get(_f, 0.0)), 4),
                    }
                    for _f, _s in _pairs
                    if abs(_s) > 1e-6
                ]
        except Exception:
            pass
    st.session_state["rm_last_rf"] = {"drivers": shap_drivers} if shap_drivers else None

    temporal = temporal_reconstruction_explainer(txns)
    spatial = spatial_cluster_explainer(record, txns)
    risk_for_verdict = float(st.session_state["rm_last_result"].get("risk_score", 0.0))
    drivers = st.session_state["rm_last_rf"].get("drivers", []) if st.session_state["rm_last_rf"] else []
    verdict = global_verdict_explainer(
        risk_score=risk_for_verdict,
        drivers=drivers,
        temporal=temporal,
        spatial=spatial,
    )
    st.session_state["rm_temporal"] = temporal
    st.session_state["rm_spatial"] = spatial
    st.session_state["rm_verdict"] = verdict
    st.session_state["rm_llm_explanation"] = None  # signal that a fresh LLM fetch is needed


# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────

artifacts, model, lgb_bndl, load_err = _try_load_all()

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────

st.title("Run AML Risk Model")
st.caption(
    "Enter a customer profile and add transactions one at a time. "
    "LightGBM (on top of GNN embeddings) provides calibrated fraud probability. "
    "SHAP values explain which graph-structural features drove the score."
)

if load_err:
    st.warning(
        f"Model artifacts unavailable: `{load_err}`  \n"
        "Scores will be placeholder (0.0) until models are present."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main profile panel (no sidebar split)
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Customer & KYC Profile")
p1, p2, p3, p4 = st.columns([1.6, 1.1, 1.1, 1.0], gap="small")
with p1:
    st.session_state["rm_customer_id"] = st.text_input(
        "Customer ID",
        value=st.session_state["rm_customer_id"],
        key="main_cust_id",
    )
with p2:
    st.session_state["rm_is_biz"] = st.toggle(
        "Business Account",
        value=bool(st.session_state["rm_is_biz"]),
        key="main_is_biz",
    )
with p3:
    st.session_state["rm_age"] = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=int(st.session_state["rm_age"]),
        step=1,
        key="main_age",
    )
with p4:
    st.session_state["rm_tenure"] = st.number_input(
        "Tenure (days)",
        min_value=0,
        max_value=36_500,
        value=int(st.session_state["rm_tenure"]),
        step=30,
        key="main_tenure",
    )

p5, p6, p7 = st.columns([1.2, 1.2, 0.8], gap="small")
with p5:
    st.session_state["rm_income"] = st.number_input(
        "Annual Income ($)",
        min_value=0,
        max_value=10_000_000,
        value=int(st.session_state["rm_income"]),
        step=1000,
        key="main_income",
    )
with p6:
    if st.session_state["rm_is_biz"]:
        st.session_state["rm_sales"] = st.number_input(
            "Annual Sales ($)",
            min_value=0,
            max_value=500_000_000,
            value=int(st.session_state["rm_sales"]),
            step=10_000,
            key="main_sales",
        )
    else:
        st.session_state["rm_sales"] = 0
        st.number_input(
            "Annual Sales ($)",
            min_value=0,
            max_value=500_000_000,
            value=0,
            step=10_000,
            disabled=True,
            key="main_sales_disabled",
        )
with p7:
    if st.session_state["rm_is_biz"]:
        st.session_state["rm_emp_count"] = st.number_input(
            "Employee Count",
            min_value=0,
            max_value=10_000,
            value=int(st.session_state["rm_emp_count"]),
            step=1,
            key="main_emp",
        )
    else:
        st.session_state["rm_emp_count"] = 0
        st.number_input(
            "Employee Count",
            min_value=0,
            max_value=10_000,
            value=0,
            step=1,
            disabled=True,
            key="main_emp_disabled",
        )

# Auto re-score when KYC changes (while transactions exist)
_kyc_hash = hash((
    st.session_state["rm_age"],
    st.session_state["rm_income"],
    st.session_state["rm_tenure"],
    st.session_state["rm_is_biz"],
    st.session_state["rm_sales"],
    st.session_state["rm_emp_count"],
))
if (st.session_state.get("_rm_kyc_hash") != _kyc_hash
        and st.session_state["rm_transactions"]):
    st.session_state["_rm_kyc_hash"] = _kyc_hash
    _run_scoring()

if st.button("Reset Session", width="content", key="main_reset"):
    for k in [k for k in st.session_state if k.startswith("rm_")]:
        del st.session_state[k]
    _init_state()
    st.rerun()

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Main: two-column layout
# ─────────────────────────────────────────────────────────────────────────────

col_input, col_result = st.columns([1, 2], gap="large")

# ── Left column: transaction form ─────────────────────────────────────────────
with col_input:
    st.subheader("Add Transaction")
    industry_codes = _load_industry_codes()
    with st.form("tx_form", clear_on_submit=True):
        amount = st.number_input("Amount (CAD $)", min_value=0.01, value=100.0, step=10.0)
        txn_type = st.selectbox("Transaction Type", TRANSACTION_TYPES, index=0)
        # Transaction timestamp
        _now = _dt.datetime.now()
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            txn_date = st.date_input("Transaction Date", value=_now.date(), key="tx_date")
        with t_col2:
            txn_time_val = st.time_input("Transaction Time", value=_now.time().replace(second=0, microsecond=0), step=60, key="tx_time")
        cat_choice = st.selectbox(
            "Merchant Category (Industry Code)",
            industry_codes,
            index=0,
            help="Applies to CARD transactions. Non-card types (ATM, CHEQUE, etc.) use their own category for the model.",
        )
        city = st.selectbox("City", CANADIAN_CITIES, index=0)
        is_ecom = st.checkbox("E-Commerce Transaction")
        debit_credit = st.radio("Debit / Credit", ["debit", "credit"], horizontal=True)
        submitted = st.form_submit_button(
            "Add Transaction & Score", width="content", type="primary"
        )

    if submitted:
        is_cash = txn_type == "ATM_WITHDRAWAL"
        if txn_type == "CARD":
            model_cat = cat_choice.split(" \u2013 ")[0]
        else:
            model_cat = txn_type
        final_city = city
        txn_iso = _dt.datetime.combine(txn_date, txn_time_val).isoformat()
        st.session_state["rm_transactions"].append({
            "amount": float(amount),
            "txn_type": txn_type,
            "merchant_category": model_cat,
            "city": final_city,
            "cash_indicator": int(is_cash),
            "ecommerce_ind": int(is_ecom),
            "debit_credit": debit_credit,
            "txn_time": txn_iso,
        })
        _run_scoring()
        st.rerun()

    # Transaction log
    txns = st.session_state["rm_transactions"]
    if txns:
        st.markdown(f"**Transaction Log** ({len(txns)} total)")
        tx_display = [
            {
                "#": i + 1,
                "Time": t.get("txn_time", "")[:16].replace("T", " "),
                "Amount": f"${t['amount']:,.2f}",
                "Type": t.get("txn_type", ""),
                "Category": t["merchant_category"],
                "City": t["city"],
                "E-com": "Yes" if t["ecommerce_ind"] else "",
                "Dr/Cr": t["debit_credit"],
            }
            for i, t in enumerate(txns)
        ]
        st.dataframe(pd.DataFrame(tx_display), width="content", hide_index=True)

        col_rm, col_clr = st.columns(2)
        with col_rm:
            if st.button("↩ Remove Last", width="content"):
                st.session_state["rm_transactions"].pop()
                if st.session_state["rm_transactions"]:
                    _run_scoring()
                else:
                    st.session_state["rm_last_result"] = None
                    st.session_state["rm_last_rf"] = None
                st.rerun()
        with col_clr:
            if st.button("Clear All", width="content"):
                st.session_state["rm_transactions"].clear()
                st.session_state["rm_last_result"] = None
                st.session_state["rm_last_rf"] = None
                st.rerun()
    else:
        st.info("No transactions yet. Add one above to score this customer.")

# ── Right column: live result card ────────────────────────────────────────────
with col_result:
    st.subheader("Live Risk Assessment")
    result = st.session_state.get("rm_last_result")
    rf_result = st.session_state.get("rm_last_rf")
    temporal = st.session_state.get("rm_temporal")
    spatial = st.session_state.get("rm_spatial")
    verdict = st.session_state.get("rm_verdict")
    kyc = _build_kyc_dict()
    txns = st.session_state["rm_transactions"]
    customer_id = st.session_state["rm_customer_id"]

    if result is None:
        # Show single-node graph while waiting for enough transactions
        n_so_far = len(txns)
        remaining = max(MIN_TRANSACTIONS - n_so_far, 0)
        if remaining > 0:
            st.info(
                f"Add at least **{MIN_TRANSACTIONS} transactions** before scoring. "
                f"{remaining} more needed — short histories produce unreliable scores."
            )
        else:
            st.info("Add a transaction to trigger risk scoring. Below is the customer node at 0% risk.")
        html = GraphVisualizer.build_live_graph(customer_id, 0.0, txns)
        import streamlit.components.v1 as _stcomp_rm
        _stcomp_rm.html(html, height=420, scrolling=False)
        st.markdown(
            "<div style='text-align:center;color:#64748b;font-size:0.85rem;margin-top:-8px;'>"
            "Graph grows as you add transactions.</div>",
            unsafe_allow_html=True,
        )
    else:
        risk_score = float(result["risk_score"])
        risk_tier = str(result["risk_tier"])
        drivers = rf_result["drivers"] if rf_result and rf_result.get("drivers") else []

        card = CustomerInfoCard(narrative_engine=RuleBasedNarrative())
        card.render(
            customer_id=customer_id,
            risk_score=risk_score,
            risk_tier=risk_tier,
            drivers=drivers,
            transactions=txns,
            kyc=kyc,
            graph_mode="live",
            graph_height="420px",
        )

        if rf_result:
            gnn_score = float(result.get("gnn_score", risk_score))
            st.caption(
                f"LightGBM risk score: **{risk_score:.3%}** · "
                f"GNN score: **{gnn_score:.3%}** "
                f"(LightGBM drives final score; GNN embedding contextualises graph neighbourhood)"
            )

        st.markdown("### Unified Explainability")
        t_tab, s_tab, g_tab = st.tabs([
            "Temporal (Autoencoder)",
            "Spatial (GNN Cluster)",
            "Global (SHAP Verdict)",
        ])

        with t_tab:
            if temporal and temporal.get("available"):
                st.caption(
                    "Per-feature reconstruction error pinpoints which behavioral dimensions diverge most from baseline."
                )
                top_temporal = temporal.get("top_features", [])
                if top_temporal:
                    tdf = pd.DataFrame(top_temporal)
                    tdf = tdf.rename(columns={"feature_label": "Feature", "error": "Reconstruction Error"})
                    st.bar_chart(tdf.set_index("Feature")["Reconstruction Error"], height=260)
                    st.dataframe(
                        tdf[["Feature", "Reconstruction Error"]],
                        width="content",
                        hide_index=True,
                    )
                st.metric("Anomalous Txn Share", f"{float(temporal.get('anomaly_rate', 0.0)):.1%}")
            else:
                st.info(temporal.get("reason", "Temporal explainer unavailable.") if temporal else "Temporal explainer unavailable.")

        with s_tab:
            if spatial and spatial.get("available"):
                st.caption(
                    "Cluster profile is selected via centroid-style matching against live behavioral aggregates."
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Assigned Cluster", f"{spatial.get('cluster_id')}")
                c2.metric("Pattern", str(spatial.get("cluster_label", "Unknown")))
                c3.metric("Risk Prior", str(spatial.get("cluster_risk", "Unknown")))

                top_drivers = spatial.get("top_feature_drivers", [])
                if top_drivers:
                    ddf = pd.DataFrame(top_drivers)
                    ddf["Feature"] = ddf["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
                    ddf = ddf.rename(columns={"z_score": "Cluster Z-Score"})
                    st.dataframe(
                        ddf[["Feature", "Cluster Z-Score"]],
                        width="content",
                        hide_index=True,
                    )

                hdf = spatial.get("heatmap_df")
                if isinstance(hdf, pd.DataFrame) and not hdf.empty:
                    st.markdown("**GNN Explainer Heatmap (Top 9 Patterns)**")
                    feature_cols = [
                        c for c in hdf.columns if c not in ("cluster_id", "profile_name")
                    ]
                    pretty = hdf.copy()
                    pretty = pretty.set_index("cluster_id")
                    rename_map = {c: FEATURE_LABELS.get(c, c) for c in feature_cols}
                    pretty = pretty.rename(columns=rename_map)
                    numeric_cols = [rename_map[c] for c in feature_cols if c in rename_map]
                    styled = pretty.style
                    if numeric_cols:
                        styled = styled.background_gradient(
                            cmap="RdYlBu_r", axis=None, subset=numeric_cols
                        ).format({c: "{:.2f}" for c in numeric_cols})
                    st.dataframe(
                        styled,
                        width="content",
                    )
            else:
                st.info(spatial.get("reason", "Spatial explainer unavailable.") if spatial else "Spatial explainer unavailable.")

        with g_tab:
            if verdict:
                st.markdown(
                    f"<div style='background:#0f172a;border:1px solid #334155;border-radius:10px;padding:12px;color:#e2e8f0;'>{verdict.get('summary', '')}</div>",
                    unsafe_allow_html=True,
                )

                push = verdict.get("push", [])[:5]
                pull = verdict.get("pull", [])[:3]
                if push:
                    st.markdown("**Risk Push Contributors**")
                    st.dataframe(
                        pd.DataFrame(push)[["description", "share_pct"]].rename(
                            columns={"description": "Driver", "share_pct": "Contribution %"}
                        ),
                        width="content",
                        hide_index=True,
                    )
                if pull:
                    st.markdown("**Risk Pull Contributors**")
                    st.dataframe(
                        pd.DataFrame(pull)[["description", "share_pct"]].rename(
                            columns={"description": "Driver", "share_pct": "Contribution %"}
                        ),
                        width="content",
                        hide_index=True,
                    )