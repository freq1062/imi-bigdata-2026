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
import streamlit.components.v1 as st_components
import pandas as pd
import numpy as np

from lib.components import CustomerInfoCard, RuleBasedNarrative, GraphVisualizer
from lib.model_utils import (
    load_artifacts,
    load_sage_model,
    load_rf_model,
    score_new_customers,
    explain_with_rf,
    compute_txn_aggregates,
)

st.set_page_config(
    page_title="Run AML Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# (sidebar logo removed — header displays logo on Home page)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (cached — loaded once per server session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading GraphSAGE artifacts…")
def _load_artifacts():
    return load_artifacts("outputs/sage_artifacts.pkl")

@st.cache_resource(show_spinner="Loading GraphSAGE model…")
def _load_model(_artifacts):
    return load_sage_model("outputs/fraud_sage_model.pth", artifacts=_artifacts)

@st.cache_resource(show_spinner="Loading RF proxy…")
def _load_rf():
    return load_rf_model("outputs/rf_model_sage")


def _try_load_all():
    try:
        arts = _load_artifacts()
        model = _load_model(arts)
        rf, expl, meta = _load_rf()
        return arts, model, rf, expl, meta, None
    except Exception as e:
        return None, None, None, None, None, str(e)


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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Common option lists
# ─────────────────────────────────────────────────────────────────────────────

TRANSACTION_TYPES = ["CARD", "ATM_WITHDRAWAL", "CHEQUE", "EFT", "EMT", "WIRE"]


@st.cache_data
def _load_industry_codes():
    """Load KYC industry codes from CSV and return sorted 'CODE – Label' strings."""
    path = os.path.join(_ROOT, "data", "kyc_industry_codes.csv.gz")
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
    """Score current customer and update session state."""
    record = _build_record()
    if model is not None and artifacts is not None:
        result_df = score_new_customers([record], model, artifacts)
        st.session_state["rm_last_result"] = result_df.iloc[0].to_dict()
    else:
        st.session_state["rm_last_result"] = {
            "customer_id": record["customer_id"],
            "risk_score": 0.0,
            "risk_tier": "LOW",
            "predicted_label": 0,
        }
    # RF explanation
    txns = st.session_state["rm_transactions"]
    kyc_feats = {
        "age": record["age"],
        "income": record["income"],
        "tenure": record["tenure"],
        "tenure_days": record["tenure"],
        "sales": record["sales"],
        "annual_sales": record["sales"],
        "emp_count": record["emp_count"],
        "employee_count": record["emp_count"],
        "is_biz": record["is_biz"],
        "is_business": record["is_biz"],
        "avg_txn_amount": record["avg_txn_amount"],
        "max_txn_amount": record["max_txn_amount"],
        "std_txn_amount": record["std_txn_amount"],
        "txn_count": record["txn_count"],
        "cash_rate": record["cash_rate"],
        "cash_withdrawal_rate": record["cash_rate"],
        "ecom_rate": record["ecom_rate"],
        "ecommerce_rate": record["ecom_rate"],
        "avg_24h_velocity": record["avg_24h_velocity"],
        "avg_amount_zscore": 0.0,
        # Location & temporal diversity features (returned by compute_txn_aggregates)
        "unique_cities":     record.get("unique_cities", 0.0),
        "unique_categories": record.get("unique_categories", 0.0),
        "min_time_delta":    record.get("min_time_delta", 0.0),
        "time_span_hours":   record.get("time_span_hours", 0.0),
        "geo_velocity":      record.get("geo_velocity", 0.0),
    }
    if rf_model is not None and rf_explainer is not None:
        st.session_state["rm_last_rf"] = explain_with_rf(
            kyc_feats, rf_model, rf_explainer, rf_meta # type: ignore
        )
    else:
        st.session_state["rm_last_rf"] = None


# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────

artifacts, model, rf_model, rf_explainer, rf_meta, load_err = _try_load_all()

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────

st.title("Run AML Risk Model")
st.caption(
    "Enter a customer profile and add transactions one at a time. "
    "GraphSAGE scores risk inductively \u2014 no retraining needed. "
    "RF proxy provides human-readable SHAP explanations."
)

if load_err:
    st.warning(
        f"Model artifacts unavailable: `{load_err}`  \n"
        "Scores will be placeholder (0.0) until models are present."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Customer KYC
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Customer Profile")
    st.session_state["rm_customer_id"] = st.text_input(
        "Customer ID",
        value=st.session_state["rm_customer_id"],
        key="sid_cust_id",
    )
    st.subheader("KYC Information")
    st.session_state["rm_is_biz"] = st.toggle(
        "Business Account",
        value=bool(st.session_state["rm_is_biz"]),
        key="sid_is_biz",
    )
    st.session_state["rm_age"] = st.number_input(
        "Age (years)", min_value=18, max_value=100,
        value=int(st.session_state["rm_age"]), step=1, key="sid_age",
    )
    st.session_state["rm_income"] = st.number_input(
        "Annual Income ($)", min_value=0, max_value=10_000_000,
        value=int(st.session_state["rm_income"]), step=1000, key="sid_income",
    )
    st.session_state["rm_tenure"] = st.number_input(
        "Account Tenure (days)", min_value=0, max_value=36_500,
        value=int(st.session_state["rm_tenure"]), step=30, key="sid_tenure",
    )
    if st.session_state["rm_is_biz"]:
        st.session_state["rm_sales"] = st.number_input(
            "Annual Sales ($)", min_value=0, max_value=500_000_000,
            value=int(st.session_state["rm_sales"]), step=10_000, key="sid_sales",
        )
        st.session_state["rm_emp_count"] = st.number_input(
            "Employee Count", min_value=0, max_value=10_000,
            value=int(st.session_state["rm_emp_count"]), step=1, key="sid_emp",
        )
    else:
        st.session_state["rm_sales"] = 0
        st.session_state["rm_emp_count"] = 0

    st.divider()
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

    if st.button("Reset", width="content", key="sid_reset"):
        for k in [k for k in st.session_state if k.startswith("rm_")]:
            del st.session_state[k]
        _init_state()
        st.rerun()

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
    kyc = _build_kyc_dict()
    txns = st.session_state["rm_transactions"]
    customer_id = st.session_state["rm_customer_id"]

    if result is None:
        # Show single-node graph while no transactions added
        st.info("Add a transaction to trigger risk scoring. Below is the customer node at 0% risk.")
        html = GraphVisualizer.build_live_graph(customer_id, 0.0, [])
        st_components.html(html, height=380, scrolling=False)
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
            st.caption(
                f"GraphSAGE score: **{risk_score:.3%}**  ·  "
                f"RF proxy score: **{rf_result.get('rf_score', 0):.3%}** "
                f"(RF drives narrative; GraphSAGE is the authoritative risk signal)"
            )