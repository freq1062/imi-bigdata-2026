import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

st.title("Run AML Risk Model")

df = pd.read_csv("data/demo_customers.csv")

customer_id = st.selectbox("Select Customer", df["customer_id"].unique())

customer_data = df[df["customer_id"] == customer_id]

fig = px.histogram(customer_data, x="transaction_amount")
st.plotly_chart(fig)

fig2 = px.line(customer_data, x="date", y="transaction_amount")
st.plotly_chart(fig2)

if st.button("Analyze for Financial Crime Risk"):

    model = pickle.load(open("model/trained_model.pkl", "rb"))
    
    features = customer_data[["feature1","feature2","feature3"]]
    prediction = model.predict_proba(features)[0][1]
    
    st.session_state["risk_score"] = prediction
    st.success("Analysis Complete")