# Project Aegis: Graph-SAGE Model for AML Detection
Team 76 | IMI Big Data & AI Competition 2026

Project Aegis is an end-to-end Anti-Money Laundering (AML) solution. While traditional systems look at individual transactions in isolation, Aegis uses Graph Neural Networks to analyze the social and geographical "pockets" where financial crime thrives.

[Link to project](https://imi-bigdata-2026-team76.streamlit.app/)
<img width="1441" height="594" alt="image" src="https://github.com/user-attachments/assets/38022968-5d74-42e1-8591-7c2fa42525ae" />

## How It's Made

**Tech used:** Python, PyTorch Geometric, Streamlit, Llama 3.2 (via Ollama), Scikit-Learn.

**Model:** Inductive GraphSAGE

We implemented a GraphSAGE (SAmple and aggreGatE) architecture. Unlike standard Graph Attention Networks (GATs) that are transductive (can only score nodes they've seen before), GraphSAGE learns an aggregation function.

    This allows our model to score "unseen" customers or new merchants in real-time without retraining the entire graph.

    We represented Customers, Locations, and Merchant Categories as nodes, with the over 6 million transactions connecting them as edges.

With labeled fraud at only ~1% in the original dataset, we utilized two advanced techniques:

    Integrated the BankSim agent-based simulation dataset to boost our initial fraud signal to ~6%.

    Iterative Pseudo-Labeling: The model was trained on high-confidence "initially labeled" cases, then allowed to label the rest of the 6-million-transaction dataset as it gained confidence, effectively turning our unlabeled data into a massive training resource.

**Explainability:** The "Shadow" Random Forest

To solve the "Black Box" problem of Graph Networks, we built a dual-layer explanation system:

    Proxy Modeling: A Random Forest model was trained to mimic the GraphSAGE's decision-making process using processed customer features.

    SHAP Analysis: We used Shapley values to extract exactly which features (e.g., "Transactions in last 24h" or "Cash %") triggered the flag.

    LLM Narratives: These SHAP values were fed into Llama 3.2:3b to generate human-readable SAR (Suspicious Activity Report) narratives and connected to possible organization types identified in our AML Knowledge Library.

    Example: " Customer exhibits behavior consistent with Project Guardian (Synthetic Opioids) or Project Protect (Human Trafficking). Further investigation and analysis are required to confirm these findings and determine the specific risk profile."

## Performance & Results

We evaluated the model using two primary metrics to account for the heavy class imbalance:
**ROC-AUC** - 0.996	Measures the trade-off between True Positives and False Positives.
**PR-AUC** - 0.986	Specifically measures success in catching criminals (Recall) vs. accuracy of flags (Precision).

## Optimizations

Memory Efficiency: The models were trained on lab machines that have a 9.5GB disk quota. To address this, we implemented Gzip compression for all CSV assets and a custom file-flattening script to manage the 213MB model artifacts efficiently.

Inductive Inference: The Streamlit app runs inference on the CPU using pre-computed embeddings, making it fast even on low-spec hardware (i5/8GB RAM). 

## Lessons Learned

In AML, individual behaviour is easy to hide, while relationships are not. 
Creating a robust startup script is really hard, getting things to work between different machines and versions of Python took a lot of time and effort, especially with the torch-sparse library.
Research is incredibly important, 90% of the total competition time we had was spent looking at different approaches to the problem, watching YouTube videos, and gradually piecing an idea together.
