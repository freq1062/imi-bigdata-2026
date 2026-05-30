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
        WEBAPP_OUTPUTS_DIR / "model_output_explanations.csv",
        WEBAPP_OUTPUTS_DIR / "meta_cluster_semantic_labels.json",
        WEBAPP_OUTPUTS_DIR / "meta_cluster_significant_deltas.json",
        WEBAPP_OUTPUTS_DIR / "meta_cluster_top3_categories.json",

    ]


def _missing_hard() -> list[str]:
    return [str(p) for p in _hard_required() if not p.exists()]

def _missing_soft() -> list[str]:
    return [str(p) for p in _soft_required() if not p.exists()]

# --- Streamlit UI ---
from lib.components import header_with_logo

st.set_page_config(layout="wide", page_title="Team 76 AML Detection")
st.markdown("""
<style>
/* Larger body text everywhere except headings */
p, li, .stMarkdown p, .stAlert p, label, .stInfo, div[data-testid="stText"] {
    font-size: 1.35rem !important;
    line-height: 1.7 !important;
}
</style>
""", unsafe_allow_html=True)
header_with_logo("Project Aegis: AI-Driven AML / ML-TF Detection")

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
### Real-Time Financial Crime Risk Detection
""")

# ── Feature cards (CSS keyframe animation, no JS needed) ──
st.markdown("""
<style>
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(40px); }
    to   { opacity: 1; transform: translateY(0); }
}
.feature-grid { display: flex; gap: 20px; margin: 8px 0 24px 0; }
.feature-card {
    flex: 1;
    background-color: #1e1e2e;
    border: 1px solid #3a3a5c;
    border-radius: 14px;
    padding: 36px 28px;
    min-height: 200px;
    opacity: 0;
    animation: fadeUp 0.7s ease forwards;
}
.feature-card:nth-child(1) { animation-delay: 0.1s; }
.feature-card:nth-child(2) { animation-delay: 0.3s; }
.feature-card:nth-child(3) { animation-delay: 0.5s; }
.feature-card:hover { background-color: #2a2a4a; border-color: #7b68ee; }
.feature-icon { margin-bottom: 18px; }
.feature-icon svg { width: 48px; height: 48px; }
.feature-title { font-size: 1.15rem; font-weight: 700; color: #c9b8ff; margin-bottom: 10px; }
.feature-desc  { font-size: 1.05rem; color: #a0a0c0; line-height: 1.7; }
</style>
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">
            <!-- Network graph nodes: Risk Scoring -->
            <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="24" cy="8"  r="5" stroke="#7b68ee" stroke-width="2.5"/>
                <circle cx="8"  cy="38" r="5" stroke="#7b68ee" stroke-width="2.5"/>
                <circle cx="40" cy="38" r="5" stroke="#7b68ee" stroke-width="2.5"/>
                <circle cx="24" cy="26" r="4" fill="#7b68ee" fill-opacity="0.25" stroke="#7b68ee" stroke-width="2"/>
                <line x1="24" y1="13" x2="24" y2="22" stroke="#7b68ee" stroke-width="2" stroke-linecap="round"/>
                <line x1="21" y1="29" x2="11"  y2="35" stroke="#7b68ee" stroke-width="2" stroke-linecap="round"/>
                <line x1="27" y1="29" x2="37"  y2="35" stroke="#7b68ee" stroke-width="2" stroke-linecap="round"/>
            </svg>
        </div>
        <div class="feature-title">Risk Scoring</div>
        <div class="feature-desc">GraphSAGE-based graph neural network detection trained on Scotiabank transaction data to surface suspicious activity with high precision.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">
            <!-- Document with text lines + highlight: Explainable AI -->
            <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="8" y="4" width="28" height="36" rx="3" stroke="#7b68ee" stroke-width="2.5"/>
                <rect x="8" y="32" width="28" height="8" rx="2" fill="#7b68ee" fill-opacity="0.18"/>
                <line x1="14" y1="13" x2="30" y2="13" stroke="#7b68ee" stroke-width="2" stroke-linecap="round"/>
                <line x1="14" y1="19" x2="30" y2="19" stroke="#7b68ee" stroke-width="2" stroke-linecap="round"/>
                <line x1="14" y1="25" x2="23" y2="25" stroke="#7b68ee" stroke-width="2" stroke-linecap="round"/>
                <circle cx="38" cy="38" r="7" fill="#1e1e2e" stroke="#7b68ee" stroke-width="2.5"/>
                <line x1="38" y1="35" x2="38" y2="38" stroke="#7b68ee" stroke-width="2" stroke-linecap="round"/>
                <circle cx="38" cy="40.5" r="1" fill="#7b68ee"/>
            </svg>
        </div>
        <div class="feature-title">Explainable AI</div>
        <div class="feature-desc">Every prediction is paired with a human-readable narrative via Llama 3.2, making model decisions traceable and audit-ready.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">
            <!-- Stacked layers / database: Knowledge Library -->
            <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <ellipse cx="24" cy="12" rx="16" ry="6" stroke="#7b68ee" stroke-width="2.5"/>
                <path d="M8 12 C8 12 8 22 8 22 C8 25.31 15.16 28 24 28 C32.84 28 40 25.31 40 22 L40 12" stroke="#7b68ee" stroke-width="2.5" fill="none"/>
                <path d="M8 22 C8 22 8 32 8 32 C8 35.31 15.16 38 24 38 C32.84 38 40 35.31 40 32 L40 22" stroke="#7b68ee" stroke-width="2.5" fill="none"/>
                <ellipse cx="24" cy="32" rx="16" ry="6" fill="#7b68ee" fill-opacity="0.15" stroke="#7b68ee" stroke-width="2"/>
            </svg>
        </div>
        <div class="feature-title">Knowledge Library</div>
        <div class="feature-desc">A traceable, curated collection of AML/TF typologies, red flags, and model feature mappings.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Stats belt (all three visible at once) ──
import streamlit.components.v1 as components
components.html("""
<style>
  body { margin: 0; background: transparent; font-family: sans-serif; }
  .stats-belt {
    display: flex;
    justify-content: space-around;
    align-items: center;
    padding: 24px 0 8px 0;
    border-top: 1px solid #3a3a5c;
  }
  .stat-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
  }
  .stat-item:not(:last-child) {
    border-right: 1px solid #3a3a5c;
  }
  .stat-value { font-size: 3.2rem; font-weight: 900; color: #7b68ee; line-height: 1; }
  .stat-label { font-size: 1.15rem; color: #c0b8e0; margin-top: 8px; letter-spacing: 0.03em; }
</style>
<div class="stats-belt">
  <div class="stat-item">
    <div class="stat-value">99%</div>
    <div class="stat-label">Detection Accuracy</div>
  </div>
  <div class="stat-item">
    <div class="stat-value">18</div>
    <div class="stat-label">Typologies Covered</div>
  </div>
  <div class="stat-item">
    <div class="stat-value">100%</div>
    <div class="stat-label">Explainability Coverage</div>
  </div>
</div>
""", height=130)

st.divider()

st.subheader("How It Works")

st.markdown("""
<style>
.how-it-works-grid {
    display: flex;
    gap: 16px;
    margin-top: 8px;
}
.hiw-card {
    flex: 1;
    background-color: #1e1e2e;
    border: 1px solid #3a3a5c;
    border-radius: 10px;
    padding: 24px 20px;
    transition: background-color 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
    cursor: default;
}
.hiw-card:hover {
    background-color: #2a2a4a;
    border-color: #7b68ee;
    transform: translateY(-3px);
}
.hiw-step {
    font-size: 1.6rem;
    font-weight: 800;
    color: #7b68ee;
    margin-bottom: 8px;
}
.hiw-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e0e0f0;
    margin-bottom: 6px;
}
.hiw-desc {
    font-size: 1.3rem;
    color: #a0a0c0;
    line-height: 1.7;
}
</style>

<div class="how-it-works-grid">
    <div class="hiw-card">
        <div class="hiw-step">01</div>
        <div class="hiw-title">Customer Transaction Ingestion</div>
        <div class="hiw-desc">Automated data pipeline for real-time streaming of financial transactions.</div>
    </div>
    <div class="hiw-card">
        <div class="hiw-step">02</div>
        <div class="hiw-title">Feature Engineering</div>
        <div class="hiw-desc">Dynamic generation of risk-based indicators from raw transaction data.</div>
    </div>
    <div class="hiw-card">
        <div class="hiw-step">03</div>
        <div class="hiw-title">ML Risk Scoring</div>
        <div class="hiw-desc">Multi-layer neural network processing to assign and rank fraud risk scores.</div>
    </div>
    <div class="hiw-card">
        <div class="hiw-step">04</div>
        <div class="hiw-title">Explainable Output</div>
        <div class="hiw-desc">Human-readable narratives generated for regulatory reporting and review.</div>
    </div>
</div>
""", unsafe_allow_html=True)

print("[INFO] System initialized from local webapp_resources (no network download).")

st.divider()

st.subheader("How to Navigate This App")

st.markdown("""
<style>
.nav-main-section {
    background-color: #1a1a2e;
    border: 1px solid #3a3a5c;
    border-radius: 12px;
    padding: 28px 28px 24px 28px;
    margin-bottom: 28px;
    transition: border-color 0.2s;
}
.nav-main-section:hover { border-color: #7b68ee; }
.nav-section-subtitle {
    font-size: 0.8rem;
    color: #7b68ee;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.nav-section-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #e0e0f0;
    margin-bottom: 8px;
}
.nav-section-desc {
    font-size: 1.25rem;
    color: #a0a0c0;
    line-height: 1.7;
    margin-bottom: 0;
}
.nav-feature-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 18px;
}
.nav-feature-panel {
    flex: 1 1 calc(33% - 12px);
    min-width: 190px;
    background-color: #22223a;
    border: 1px solid #3a3a5c;
    border-radius: 8px;
    padding: 16px 18px;
    transition: background-color 0.2s, border-color 0.2s, transform 0.15s;
    box-sizing: border-box;
}
.nav-feature-panel:hover {
    background-color: #2c2c50;
    border-color: #7b68ee;
    transform: translateY(-2px);
}
.nav-panel-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #7b68ee;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.nav-panel-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #c9b8ff;
    margin-bottom: 8px;
}
.nav-panel-body {
    font-size: 1.2rem;
    color: #a0a0c0;
    line-height: 1.65;
}
.nav-badge {
    display: inline-block;
    background-color: #2a2a4a;
    border: 1px solid #7b68ee;
    border-radius: 6px;
    padding: 2px 10px;
    color: #c9b8ff;
    font-size: 0.82rem;
    font-weight: 700;
    vertical-align: middle;
}
</style>

<!-- ── Page 1: Knowledge Library ── -->
<div class="nav-main-section">
    <div class="nav-section-title">Knowledge Library</div>
    <div class="nav-section-desc">An educational reference containing the AML typologies, risk patterns, and regulatory intelligence that underpin the detection model.</div>
    <div class="nav-feature-grid">
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Sidebar</div>
            <div class="nav-panel-title">Module Navigation</div>
            <div class="nav-panel-body">Use the sidebar radio menu to switch between the five modules: <em>Core Intelligence &amp; Risk Patterns</em>, <em>Red Flag Indicator Library</em>, <em>Risk Clusters for Modelling</em>, <em>LLM &amp; NLP Opportunities</em>, and <em>Source Index</em>.</div>
        </div>
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Main Panel</div>
            <div class="nav-panel-title">Expandable Sections</div>
            <div class="nav-panel-body">Each module is organised into expandable section headers. Click any header to reveal detailed typology breakdowns, mechanism descriptions, and detection logic.</div>
        </div>
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Usage</div>
            <div class="nav-panel-title">Read-Only Reference</div>
            <div class="nav-panel-body">No inputs are required on this page. Browse freely to build context on the fraud patterns and regulations the model is built around.</div>
        </div>
    </div>
</div>

<!-- ── Page 2: Run Model ── -->
<div class="nav-main-section">
    <div class="nav-section-title">Run Model</div>
    <div class="nav-section-desc">Interactively assess a new customer in real time. The GraphSAGE model rescores the customer after every change you make.</div>
    <div class="nav-feature-grid">
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Sidebar</div>
            <div class="nav-panel-title">Customer Profile</div>
            <div class="nav-panel-body">Enter a Customer ID, age, annual income (CAD), account tenure (months), and customer type. For business accounts, employee count and annual sales fields will appear automatically.</div>
        </div>
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Main Panel</div>
            <div class="nav-panel-title">Add Transaction</div>
            <div class="nav-panel-body">Fill in the amount, transaction type, merchant category, city, and debit/credit direction, then click <strong>Add Transaction</strong>. The risk score and graph update immediately after each addition.</div>
        </div>
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Main Panel</div>
            <div class="nav-panel-title">Transaction Graph</div>
            <div class="nav-panel-body">The graph renders the customer node (colour reflects risk level), merchant-category hub nodes (blue), and city hub nodes (orange). Edge labels show the transaction amount and type.</div>
        </div>
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Button</div>
            <div class="nav-panel-title"><span class="nav-badge">Remove Last</span></div>
            <div class="nav-panel-body">Removes the most recently added transaction and immediately rescores the customer, letting you undo a single entry without clearing everything.</div>
        </div>
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Button</div>
            <div class="nav-panel-title"><span class="nav-badge">Clear All</span></div>
            <div class="nav-panel-body">Removes all transactions for the current customer in one action, while keeping the customer profile fields intact so you can start a fresh transaction set.</div>
        </div>
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Sidebar Button</div>
            <div class="nav-panel-title"><span class="nav-badge">Reset</span></div>
            <div class="nav-panel-body">Clears the entire customer profile and all associated transactions, returning the page to its initial blank state ready for a new assessment.</div>
        </div>
    </div>
</div>

<!-- ── Page 3: Model Output ── -->
<div class="nav-main-section">
    <div class="nav-section-title">Model Output</div>
    <div class="nav-section-desc">Review pre-computed AML risk scores for the full customer database. Use this page to triage and action flagged accounts.</div>
    <div class="nav-feature-grid">
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Top of Page</div>
            <div class="nav-panel-title">Database Statistics</div>
            <div class="nav-panel-body">Opening metrics show total customers ingested, the number flagged as high-risk, and the total number of graph edges in the transaction network.</div>
        </div>
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Navigation</div>
            <div class="nav-panel-title">Tabs: Top Suspicious / Search</div>
            <div class="nav-panel-body">The <strong>Top Suspicious Customers</strong> tab lists the highest-risk profiles ranked by score. The <strong>Search</strong> tab lets you look up any customer directly by Customer ID or Transaction ID.</div>
        </div>
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Customer Card</div>
            <div class="nav-panel-title">Risk Details</div>
            <div class="nav-panel-body">Each card shows the risk score, a SHAP-driven narrative explanation, an interactive transaction graph, and a filterable transaction table.</div>
        </div>
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Transaction Table</div>
            <div class="nav-panel-title">Display and Filter Transactions</div>
            <div class="nav-panel-body">Click the Transaction drop down button to display a table with all the customers transactions. Filter by category, type, city, and date range.</div>
        </div>
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Button</div>
            <div class="nav-panel-title"><span class="nav-badge">Clear Filters</span></div>
            <div class="nav-panel-body">Resets any active category or city filter, restoring the full transaction table for the current customer card.</div>
        </div>
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Button</div>
            <div class="nav-panel-title"><span class="nav-badge">Confirm Fraud</span></div>
            <div class="nav-panel-body">Marks the customer as confirmed fraud in the investigator review log. Click again on a marked customer to remove the confirmation.</div>
        </div>
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Button</div>
            <div class="nav-panel-title"><span class="nav-badge">Clear / Not Fraud</span></div>
            <div class="nav-panel-body">Marks the customer as reviewed and cleared. Click again to remove the cleared status if your assessment changes.</div>
        </div>
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Sidebar</div>
            <div class="nav-panel-title">Review Summary</div>
            <div class="nav-panel-body">Displays a live running count of confirmed-fraud and cleared decisions made in the current session so you can track review progress at a glance.</div>
        </div>
        <div class="nav-feature-panel">
            <div class="nav-panel-label">Sidebar Buttons</div>
            <div class="nav-panel-title"><span class="nav-badge">Export</span> &amp; <span class="nav-badge">Clear All Reviews</span></div>
            <div class="nav-panel-body"><strong>Export Review Decisions</strong> downloads a CSV of all decisions made this session. <strong>Clear All Reviews</strong> resets every investigator decision, starting the review log fresh.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
print("[INFO] System initialized from local webapp_resources (no network download).")
