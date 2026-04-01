# Project Aegis: Graph-SAGE Model for AML Detection
Team 76 | IMI Big Data & AI Competition 2026

Project Aegis is an end-to-end Anti-Money Laundering (AML) solution. While traditional systems look at individual transactions in isolation, Aegis uses Graph Neural Networks to analyze the social and geographical "pockets" where financial crime thrives.

[Link to project](https://imi-bigdata-2026-team76.streamlit.app/)
<img width="1441" height="594" alt="image" src="https://github.com/user-attachments/assets/38022968-5d74-42e1-8591-7c2fa42525ae" />

## How It's Made

**Tech used:** Python, PyTorch Geometric, Streamlit, Llama 3.2 (via Ollama), Scikit-Learn, LightGBM.

**Model:** Multi-Stage Graph-Based Pipeline

Aegis uses a five-stage pipeline where each component feeds into the next, progressively refining the fraud signal from raw transactions to a calibrated risk score.

**Stage 1 — Transaction Autoencoder**

Before the graph is built, a symmetric MLP autoencoder is trained unsupervised on all transactions to reconstruct 15 behavioral features (velocity bursts, geo-impossibility, structuring signals, etc.). After a supervised fine-tuning pass on labeled fraud, the per-transaction reconstruction error becomes a risk signal. High reconstruction error means the transaction looks nothing like normal behavior. The mean error per customer is then carried forward as a node feature in the graph.

**Stage 2 — Heterogeneous GraphSAGE with DGI Pre-Training**

We implemented a GraphSAGE (SAmple and aggreGatE) architecture on a heterogeneous graph with three node types: Customers, Merchant Categories, and Cities. Unlike transductive methods (e.g., standard GATs), GraphSAGE learns an aggregation function, allowing it to score new, previously unseen customers at inference time without retraining.

To address the severe label scarcity (~1% fraud), the encoder is first pre-trained using Deep Graph Infomax (DGI) — an unsupervised objective that trains the model to distinguish real node embeddings from those on corrupted graphs. This forces the encoder to capture meaningful behavioral and structural patterns before it ever sees a fraud label. The model is then fine-tuned on a small set of high-confidence labeled cases using Focal Loss to handle class imbalance.

Each customer node carries 18 features derived from KYC data, transaction aggregates, graph structural metrics, and the autoencoder risk score from Stage 1.

**Stage 3 — GMM Clustering**

A Gaussian Mixture Model (GMM) is fit on the 64-dimensional customer embeddings produced by the GraphSAGE encoder. This clusters customers into behavioral archetypes in the embedding space. For each customer, the GMM provides a cluster assignment and a membership confidence score. Customers with low likelihood under all mixture components — those that don't fit any learned behavioral pattern — receive a high anomaly score. These cluster-level statistics become features for the final ranker.

**Stage 4 — LightGBM Ranker**

A LightGBM classifier is trained on ~30 embedding-derived features: PCA projections of the 64-d GNN embedding, GMM cluster statistics, k-NN proximity to known fraud/legit centroids, the DGI anomaly score, the autoencoder risk signal, and anchor proximity scores. The final `review_priority_score` is a weighted ensemble combining the LightGBM probability, cluster consensus, DGI anomaly, autoencoder risk, and anchor proximity.

**Explainability:** GNNExplainer + SHAP

To solve the "Black Box" problem of Graph Networks, we built a two-layer explanation system:

    GNNExplainer: We use PyG's GNNExplainer to generate per-customer edge relevance masks, identifying which specific Merchant Category and City connections were most influential in the GraphSAGE encoder's scoring of a flagged customer.

    SHAP Analysis: A SHAP TreeExplainer is run on the LightGBM model to produce per-feature attributions, pinpointing exactly which behavioral signals (e.g., "DGI anomaly score", "Autoencoder risk", "Cluster fraud rate") drove the final risk score.

    LLM Narratives: These SHAP values were fed into Gemma 2b to generate human-readable SAR (Suspicious Activity Report) narratives and connected to possible organization types identified in our AML Knowledge Library.

    Example: " Customer exhibits behavior consistent with Project Guardian (Synthetic Opioids) or Project Protect (Human Trafficking). Further investigation and analysis are required to confirm these findings and determine the specific risk profile."

## How to run

1. Ensure you have Python 3.10 - 3.12. Run the setup script at the top of `train.ipynb` and confirm that all packages are installed correctly. If not, you may need to clear the existing versions to make torch, numpy, torch-scatter, etc. compatible with each other by running the following commands:

`pip uninstall torch torch-scatter torch-sparse numpy scipy -y`
`pip cache purge`
Then run the installation cell again.
2. Simply run the rest of the cells! A fresh model will be trained and the Streamlit web app will open at `localhost:8502`(For some reason it says localhost:8501.) You can also run `streamlit run Home.py` from the root directory of the project. 

            If you run the Streamlit app before train.ipynb, the model and necessary assets will be downloaded automatically. But that's not nearly as cool. 

3. Within the web app, navigate using the side panel on the left.
   
       The "Knowledge Library" tab is an interactive database of AML/TF research we have conducted with sources.
       The "Run Model" tab allows you to add transactions to a customer and see what risk score the model would output, with explanations in real time.
       The "Model Output" tab shows the top K highest risk customers from the dataset with full graph visualizations, explanations, and options to view past transactions for human analysts to review. Decisions can be exported to csv.

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
Having a solid understanding of the problem was critical to be able to find an effective solution for it.
