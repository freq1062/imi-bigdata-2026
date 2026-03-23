import streamlit as st
import os
import zipfile
import shutil
import gdown
import gzip

# --- Configuration & Resource Setup ---
MODEL_FILE_ID = '1geTbVi3oyaGYAcZjuq_LV4uI8iQnZe82'
DATA_FILE_ID = '1A7w3GqZTCVsv-A8gXj5NKV2FNc0zoduG'

ZIP_MODEL = 'model_resources.zip'
ZIP_DATA = 'data.zip'
DATA_DIR = 'data'

def csv_to_gzip(file_path):
    """Compresses a CSV file to .csv.gz and removes the original."""
    with open(file_path, 'rb') as f_in:
        with gzip.open(file_path + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(file_path)

def extract_and_cleanup(zip_path, target_dir='.'):
    """
    Unpacks and intelligently flattens single-folder wrappers while 
    preserving critical subdirectories like rf_model_sage/.
    """
    if not os.path.exists(zip_path):
        return
        
    temp_dir = f'temp_{os.path.basename(zip_path)}'
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # 1. Check if the zip has a single 'wrapper' folder
    items = os.listdir(temp_dir)
    if len(items) == 1 and os.path.isdir(os.path.join(temp_dir, items[0])):
        effective_root = os.path.join(temp_dir, items[0])
    else:
        effective_root = temp_dir

    # 2. Move everything from effective_root to target_dir
    for item in os.listdir(effective_root):
        source = os.path.join(effective_root, item)
        dest = os.path.join(target_dir, item)
        
        # Move file or entire directory (like rf_model_sage)
        if os.path.exists(dest):
            if os.path.isdir(dest):
                shutil.rmtree(dest)
            else:
                os.remove(dest)
        shutil.move(source, dest)
        print(f"Moved to root: {item}")

    # 3. Final cleanup
    shutil.rmtree(temp_dir)
    os.remove(zip_path)
    st.success("✅ Resources successfully moved to parent directory!")

def setup_all_resources():
    """Downloads and prepares both models and data."""
    # 1. Handle Model Resources
    if not os.path.exists("setup_complete.txt"):
        st.write("Fetching Model Weights...")
        gdown.download(id=MODEL_FILE_ID, output=ZIP_MODEL, quiet=False)
        extract_and_cleanup(ZIP_MODEL)
        with open("setup_complete.txt", "w") as f: f.write("Models ready.")

    # 2. Handle Data Resources
    files_to_check = [f"{DATA_DIR}/labels.csv.gz", f"{DATA_DIR}/card.csv.gz"]
    if not all(os.path.exists(f) for f in files_to_check):
        st.write("Fetching Scotiabank/BankSim Dataset...")
        gdown.download(id=DATA_FILE_ID, output=ZIP_DATA, quiet=False)
        with zipfile.ZipFile(ZIP_DATA, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        st.write("Compressing data for optimal AML library usage...")
        for filename in os.listdir(DATA_DIR):
            if filename.endswith('.csv'):
                csv_to_gzip(os.path.join(DATA_DIR, filename))
        os.remove(ZIP_DATA)

# --- Streamlit UI ---
from lib.components import header_with_logo

st.set_page_config(layout="wide", page_title="Team 76 AML Detection")
st.markdown("""
<style>
/* Larger body text everywhere except headings */
p, li, .stMarkdown p, .stAlert p, label, .stInfo, div[data-testid="stText"] {
    font-size: 1.1rem !important;
    line-height: 1.7 !important;
}
</style>
""", unsafe_allow_html=True)
header_with_logo("Project Aegis: AI-Driven AML / ML-TF Detection")

# Trigger Initialization
if not os.path.exists("setup_complete.txt") or not os.path.isdir(DATA_DIR):
    with st.status("Initializing System Resources...", expanded=True) as status:
        setup_all_resources()
        status.update(label="✅ All Resources Ready!", state="complete")

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
    font-size: 1rem;
    font-weight: 700;
    color: #e0e0f0;
    margin-bottom: 6px;
}
.hiw-desc {
    font-size: 1.05rem;
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

st.info("System initialized and model resources loaded from secure storage.")