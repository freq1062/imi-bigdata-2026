# Model Improvement Iterations — Findings

All iterations attempted to improve fraud detection over the unsupervised DGI + GMM baseline.  
Gold labels available: **10 fraud, 990 legit (train+val)** | **2 fraud, 181 legit (test)**  
Baseline: DGI MLP (64-dim embeddings) — PR-AUC = 0.0149, ROC-AUC = 0.660

---

## Iteration 2 — GraphSMOTE (class balancing in embedding space)

### Strategy
Apply SMOTE directly in the 67-dimensional DGI embedding space (64 DGI dims + 3 auxiliary
features: `gmm_max_prob`, `customer_ae_risk_norm`, `component_confidence`) to address the
severe class imbalance (only 8 gold-labeled fraud customers in train+val).

Generated **801 synthetic fraud samples** via `k_neighbors=7` interpolation, producing a
balanced 1,618-sample training set (809 fraud : 809 legit). A fresh `EmbeddingMLP` was
retrained from scratch on the balanced data.

### Results

| Metric | Baseline | GraphSMOTE |
|---|---|---|
| PR-AUC | 0.0149 | ~0.011 |
| ROC-AUC | 0.660 | ~0.50 |
| F1 | 0.049 | 0.022 |
| Confusion | TN=104, FP=77, FN=0, TP=2 | TN=0, FP=181, FN=0, TP=2 |

### Root Cause
The MLP early-stopped at epoch 0 with `val_f1=0` throughout. SMOTE generates synthetic
samples by interpolating between existing fraud points — but if those 8 fraud embeddings are
scattered throughout the legitimate manifold (confirmed later by t-SNE), every synthetic
sample is ambiguous. The problem is not class imbalance; it is the **lack of discriminative
signal in the DGI embeddings** themselves.

The DGI encoder is trained purely unsupervised (row-shuffle corruption). Fraud customers
that _structurally_ resemble legitimate customers in the graph receive nearly identical 64-dim
representations. SMOTE cannot fix this.

---

## Iteration 3 — Behavioral Feature-Augmented MLP with Pseudo-Label Expansion

### Strategy
Augment the 64-dim DGI embeddings with 12 behavioural features and 4 static KYC features
(80-dim total), use focal loss (γ=2) to up-weight hard examples, and expand the training set
with pseudo-labels from the 3,447 customers in LLM-confirmed fraud GMM components (clusters
73, 23, 48, 99, 153, 190, 106, 156). Gold-labeled samples received 5× sample weight.

**New feature matrix (81 dimensions):**
- DGI embeddings × 64
- Behavioural: `mean_amount`, `std_amount`, `txn_count`, velocity × 3, `ecommerce_rate`,
  `cash_rate`, `mean_geo_velocity_kmph`, z-scores × 2, `category_entropy` × 13
- Customer static: `age`, `income`, `tenure`, `is_biz` × 4

### Results

| Metric | Baseline | Iter 3 |
|---|---|---|
| PR-AUC | 0.0149 | 0.009 |
| ROC-AUC | 0.660 | 0.52 |
| F1 | 0.049 | ~0.015 |

Training curve was flat throughout — the model converged to predict everyone as negative.

### Root Cause
The 64 DGI dimensions drown out the 12 behavioural dimensions with noise.  
Pseudo-labels are noisy (components are only 10–50% fraudulent), and the gradient signal
from 1–2 gold-fraud validation examples is too weak to steer a 81→64→32→1
network toward meaningful separation.

---

## Iteration 4 — LDA Embedding Projection + Anomaly Ensemble

### Strategy
Abandon neural training and use methods that work with very few labeled samples:

| Component | Method | Label usage |
|---|---|---|
| LDA projection | Linear Discriminant Analysis on DGI 64-dim | 8 fraud + 809 legit (gold only) |
| Behavioural Mahalanobis | Distance from gold-legit centroid in 12-dim space | Legit only |
| DGI anomaly score | `1 - realness_probability` | Unsupervised |
| AE reconstruction error | Transaction autoencoder risk | Unsupervised |
| GMM component score | `gmm_max_prob × component_fraud_rate` | Semi-supervised |

LDA was the key insight — it finds the single linear direction that maximises the
between-class to within-class variance ratio, giving the theoretically optimal linear
separator with 8 fraud vs 809 legit gold samples.

### Results

| Metric | Baseline | LDA Ensemble |
|---|---|---|
| PR-AUC | 0.0149 | 0.011 |
| ROC-AUC | 0.660 | 0.55 |

Some fraud customers scored high on the LDA axis (LDA>0.4), but many appeared at LDA≈0.0,
indistinguishable from legitimate customers.

### Root Cause
LDA finds the optimal linear separator *given the current embeddings*. The separator
is mathematically correct but the embeddings themselves do not encode fraud.  
Many fraud customers land at LDA=0 because their DGI embeddings are in the bulk of
the legitimate distribution — no linear projection can separate them.

---

## t-SNE Confirmation — Fraud/Legit Separability in Embedding Space

A t-SNE visualization of both the 64-dim DGI embeddings and the 80-dim
DGI+behavioural embeddings was produced. All gold-labeled fraud customers (red stars)
were overlaid on the 2D projection.

**Finding:** Fraud customers are distributed uniformly throughout the legitimate
manifold in both spaces. There is no compact fraud cluster, no neighbourhood that
is predominantly fraud, and no direction that separates the two classes reliably.

This definitively confirmed the root cause: **the DGI row-shuffle corruption produces
embeddings that encode graph topology, not fraud behaviour.**

---

## Iteration 5 — One-Class Anomaly Detection on Behavioural Features Only

### Strategy
Train a One-Class SVM and Isolation Forest **exclusively on the 809 gold-legitimate
customers** in 12-dimensional behavioural feature space (excluding all DGI dimensions),
testing the hypothesis that the strong raw behavioural signals (mean amount 2×, std 3.7×)
would produce separation if not contaminated by DGI noise.

### Results

| Metric | Baseline | IsoForest | OC-SVM | Ensemble |
|---|---|---|---|---|
| PR-AUC | 0.0149 | 0.008 | 0.009 | 0.011 |
| ROC-AUC | 0.660 | 0.367 | 0.395 | 0.439 |
| Fraud/legit score ratio | — | 1.13× | 1.59× | — |

**Behavioural separation (gold-labeled customers):**

| Feature | Legit mean | Fraud mean | Ratio |
|---|---|---|---|
| `mean_amount` | 582 CAD | 1190 CAD | 2.04× |
| `std_amount` | 1209 CAD | 4165 CAD | 3.45× |
| `cash_rate` | 0.040 | 0.019 | 0.47× |

### Conclusion
The OC-SVM produced 1.59× mean separation, but the score *distributions* heavily
overlap — about half of fraud customers score in the low-anomaly legit range.
Fraud is too heterogeneous for a single behavioural profile.

---

## Iteration 6 — CNN Transaction Score Aggregation per Customer

### Strategy
Aggregate per-transaction CNN fraud probabilities (from the pre-trained transaction CNN)
into 6 customer-level features: `cnn_max`, `cnn_mean`, `cnn_p95`, `cnn_frac_high`,
`cnn_frac_extreme`, `cnn_count_high`. Combine with OC-SVM score and DGI baseline in a
regularised Logistic Regression.

> **Implementation note:** The CNN `transaction_id` column contains **integer row indices**
> into `master_transaction_pool`, not the actual alphanumeric transaction IDs.

### Results

| Metric | Baseline | CNN-max | LR-comb |
|---|---|---|---|
| PR-AUC | 0.0149 | 0.005 | 0.012 |
| ROC-AUC | 0.660 | 0.445 | 0.564 |

**Key finding:** Only **6,020 of 61,410 customers** (10%) had CNN-scored transactions.
For the other 90%, all CNN features were zero. The most discriminative CNN feature was
`cnn_count_high` (number of transactions with CNN prob > 0.7), with a 1.78× fraud/legit
ratio — but sparse coverage limits its utility.

---

## Summary Table

| Iteration | Method | PR-AUC | ROC-AUC | Delta PR-AUC |
|---|---|---|---|---|
| Baseline | DGI MLP (64-dim, row-shuffle) | **0.0149** | **0.660** | — |
| Iter 2 | GraphSMOTE | ~0.011 | ~0.50 | −0.004 |
| Iter 3 | Behavioural MLP (80-dim, focal loss, pseudo-labels) | 0.009 | 0.52 | −0.006 |
| Iter 4 | LDA + Anomaly Ensemble | 0.011 | 0.55 | −0.004 |
| Iter 5 | OC-SVM + Isolation Forest (behavioural only) | 0.009–0.011 | 0.37–0.44 | −0.004 |
| Iter 6 | CNN aggregation + LR | 0.012 | 0.56 | −0.002 |

**All iterations underperformed the baseline on the test gold set.**

---

## Root Cause

The **DGI row-shuffle corruption** (randomly permuting feature rows across nodes) means
the DGI encoder only needs to detect whether features "belong" to a node's neighbourhood,
not whether the features themselves are anomalous. Fraud customers with normal graph
topology (same merchants, same cities as legitimate customers) receive legitimate-looking
embeddings regardless of their transaction amounts or velocities.

## Fix Applied

The corruption function was replaced with **temporal history-swap corruption**:

- For each customer node in a mini-batch, randomly select another customer's identity
  (batch permutation) **and** randomly select one of 4 pre-computed temporal snapshots
  of that customer's feature history (at 25%, 50%, 75%, and 100% of full history).
- This creates "chimera" customers: the graph neighbourhood of customer A with the
  behavioural profile of customer B at a random point in B's transaction history.
- The DGI discriminator must now learn to detect when a customer's features are
  inconsistent with their transaction history *and* neighbourhood, not just whether
  features belong to the right node slot.
- The temporal dimension additionally exposes longitudinal behavioural changes —
  including the shift from pre-fraud to post-fraud transaction patterns.
