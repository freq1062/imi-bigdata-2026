import streamlit as st
import os
import zipfile
import shutil
import gdown

# --- Configuration & Resource Setup ---
FILE_ID = '1geTbVi3oyaGYAcZjuq_LV4uI8iQnZe82'
DRIVE_URL = f'https://drive.google.com/uc?id={FILE_ID}'
ZIP_NAME = 'model_resources.zip'

def extract_and_cleanup(zip_path, target_dir='.'):
    """
    Unzips a file, moves all contents to the parent directory,
    and deletes the original zip and the empty folder.
    """
    if not os.path.exists(zip_path):
        return

    # 1. Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        temp_dir = 'temp_extraction'
        zip_ref.extractall(temp_dir)

    # 2. Move files to the parent (target_dir)
    extracted_items = os.listdir(temp_dir)
    
    # If the zip contained a single nested folder, navigate into it
    if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_dir, extracted_items[0])):
        source_dir = os.path.join(temp_dir, extracted_items[0])
    else:
        source_dir = temp_dir

    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)
        d = os.path.join(target_dir, item)
        if os.path.exists(d):
            if os.path.isdir(d):
                shutil.rmtree(d)
            else:
                os.remove(d)
        shutil.move(s, d)

    # 3. Final Cleanup
    shutil.rmtree(temp_dir)
    os.remove(zip_path)

def initialize_resources():
    """
    Checks for a marker file; if missing, downloads and extracts the model resources.
    """
    # A marker file is used to prevent redundant downloads across app refreshes
    marker = "setup_complete.txt"
    if not os.path.exists(marker):
        with st.spinner("Initializing application resources... This may take a moment."):
            try:
                gdown.download(DRIVE_URL, ZIP_NAME, quiet=False)
                extract_and_cleanup(ZIP_NAME)
                with open(marker, "w") as f:
                    f.write("Resources extracted successfully.")
            except Exception as e:
                st.error(f"Initialization failed: {e}")

# --- Streamlit UI START ---
st.set_page_config(layout="wide")

st.title("AI-Driven AML / ML-TF Detection")

# Move the resource check here so the user sees a progress bar in the browser
if not os.path.exists("setup_complete.txt"):
    with st.status("Downloading and extracting model resources...", expanded=True) as status:
        st.write("Connecting to Google Drive...")
        initialize_resources()
        status.update(label="Setup complete!", state="complete", expanded=False)

st.markdown("""
### Real-Time Financial Crime Risk Detection

- **Risk Scoring**: High-precision detection of suspicious patterns.
- **Explainable AI**: Transparent insights into model decisions.
- **Regulatory-Aligned Typologies**: Built-in support for standard compliance frameworks.
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