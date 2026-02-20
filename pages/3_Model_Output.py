import streamlit as st
import shap

st.title("Model Output & Explainability")

if "risk_score" not in st.session_state:
    st.warning("Please run model first.")
else:
    risk = st.session_state["risk_score"] * 100
    
    st.metric("Risk Score", f"{risk:.2f}%")
    
    if risk > 70:
        st.error("High Risk")
    elif risk > 40:
        st.warning("Medium Risk")
    else:
        st.success("Low Risk")


explainer = pickle.load(open("model/shap_explainer.pkl", "rb"))

shap_values = explainer(features)

st.subheader("Feature Contribution")

shap.plots.bar(shap_values[0])
st.pyplot()

if risk > 70:
    st.markdown("""
    ### Behavioral Summary
    
    The customer demonstrates structured transaction behavior consistent with
    known AML typologies. Clustering below reporting thresholds and velocity spikes
    significantly increase risk classification.
    """)

st.set_page_config(
    page_title="AML Risk Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.big-font {
    font-size:25px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)