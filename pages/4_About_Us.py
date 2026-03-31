import streamlit as st
import os
from lib.components import apply_sidebar_styles

ASSETS = os.path.join(os.path.dirname(__file__), "Assets")

st.set_page_config(layout="wide")
st.markdown("<style>h1, h3 { color: #7b68ee !important; }</style>", unsafe_allow_html=True)
apply_sidebar_styles()
st.title("About Us")

st.markdown("""
Our team, Project Aegis, is the collaboration between two driven and passionate undergraduate Computer Science 
students from the University of Toronto. We are dedicated to leveraging our technical skills and creativity to develop 
            innovative solutions and further our AI & Big Data abilities. Our project focuses on building an 
            AI-driven system for detecting Money Laundering and Terrorist Financing activities, utilizing 
            advanced machine learning techniques, such as GAT/GraphSAGE, random forest, and more to deliver a high performance and
            explainable model. 
""")

st.divider()

# ---- TEAM MEMBER 1 ----
col1, col2 = st.columns([1, 3])

with col1:
    st.image(os.path.join(ASSETS, "AlexMin.jpeg"), width=200)

with col2:
    st.subheader("Alex Min")
    st.markdown("**Lead ML Engineer**")
    st.write("""
    Alex Min is a second year student at the University of Toronto studying Computer Science and Business. He is super 
    passionate about machine learning and hope to go into research about it, in particular models that can learn continuously after training. 
    
    He was inspired to make Project Aegis because he believes that catching financial fraud is an area that Machine Learning can really 
    contribute to the benefit of society. Many people have also tackled this problem in the past, and there are so many different 
    approaches one could take, which is fascinating. He worked on the feature engineering, model development, and explainability parts 
    of the project.

    Alex likes playing Minecraft and Tetris, enjoys making songs on the piano, and reading sci-fi stories.
    """)

st.divider()

# ---- TEAM MEMBER 2 ----
col1, col2 = st.columns([1, 3])

with col1:
    st.image(os.path.join(ASSETS, "MarcusAnastacio.jpeg"), width=200)

with col2:
    st.subheader("Marcus Anastacio")
    st.markdown("**Front End Developer**")
    st.write("""
    Marcus Anastacio is a second year Computer Science specialist at the University of Toronto, minoring in
    Business and Game Studies. He grew up in Calgary, Alberta, but both his parents are Brazillian. Marcus has always
    been passionate about technology and its potential to solve real-world problems. Particularly, he is fascinated
    by AI and Machine Learning, and it's capability to improve the world around us. 
    
    
    He is excited to be a part of project Aegis and to contribute his skills and creativity to the development of Aegis' AI-driven AML/ML-TF detection 
    system. He worked on creating the Streamlit Web App, as well as created the AML Knowledge Library for the project. He also assisted with brainstorming,
    debugging and problem solving for the Machine Learning model development and feature engineering.
             
    In his free time, Marcus enjoys playing and developing video games, playing drums in his rock/metal band, reading fantasy and Sci-Fi, 
    and playing complicated board games and Magic the Gathering with his friends. 
    """)

st.divider()
