import streamlit as st

st.title("AML Knowledge Library")

typology = st.selectbox(
    "Select Typology",
    ["Structuring", "Smurfing", "Trade-Based ML", "Cross-Border Risk"]
)

if typology == "Structuring":
    st.header("Structuring")
    
    st.markdown("### Definition")
    st.write("Breaking large transactions into smaller ones to avoid reporting thresholds.")
    
    st.markdown("### Red Flags")
    st.write("""
    - Multiple deposits under reporting threshold
    - Sudden spike in transaction volume
    - Rapid same-day deposits
    """)
    
    st.markdown("### Model Detection Features")
    st.write("""
    - Transaction Clustering Index
    - Reporting Threshold Proximity
    - Velocity Change Score
    """)