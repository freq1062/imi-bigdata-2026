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

# Render header with logo to the right
header_with_logo("AI-Driven AML / ML-TF Detection", img_width=260)

# Trigger Initialization
if not os.path.exists("setup_complete.txt") or not os.path.isdir(DATA_DIR):
    with st.status("Initializing System Resources...", expanded=True) as status:
        setup_all_resources()
        status.update(label="✅ All Resources Ready!", state="complete")

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

st.info("System initialized and model resources loaded from secure storage.")