import streamlit as st

st.set_page_config(layout="wide")

st.title("AI-Driven AML / ML-TF Detection")

st.markdown("""
### Real-Time Financial Crime Risk Detection

- Risk Scoring
- Explainable AI
- Regulatory-Aligned Typologies
""")

col1, col2, col3 = st.columns(3)

col1.metric("Detection Accuracy", "94%")
col2.metric("Typologies Covered", "18")
col3.metric("Explainability Coverage", "100%")

st.divider()

st.subheader("How It Works")

st.write("""
1. Customer transaction ingestion  
2. Feature engineering  
3. ML risk scoring  
4. Explainable output  
""")
