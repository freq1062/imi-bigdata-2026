# Fraud Detection Architecture & Pipeline

## Overview

This system detects fraudulent customers from banking transaction data using a multi-stage semi-supervised pipeline. The key challenge is extreme label scarcity — the dataset has only **10 confirmed fraud customers** and **990 confirmed legitimate customers** out of 61,410 total, leaving ~60,410 unlabeled. The pipeline is designed to operate under this scarcity by combining unsupervised graph representation learning, clustering-based pseudo-labeling, and an ensemble of supervised rankers.

---

## Data Sources

| Source | Files | Description |
|---|---|---|
| BankSim (synthetic) | `banksim/` | Synthetic transaction dataset used for pre-training and diversity |
| Scotiabank (real) | `data/labels.csv.gz`, `data/kyc_individual.csv.gz`, `data/kyc_smallbusiness.csv.gz`, `data/card.csv.gz` | Real customer KYC + transaction data with sparse gold labels |

Both sources are merged into a **master transaction pool** with a unified schema: `[transaction_id, customer_id, amount_cad, transaction_datetime, merchant_category, ecommerce_ind, cash_indicator, city, is_fraud, ...]`.

---

## Pipeline Stages

### Stage 1 — Feature Engineering

All transactions are unified into the master pool. Per-transaction behavioral features are computed, all with strict point-in-time safety:

- **`velocity_24h`** — count of prior transactions in trailing 24 hours (current transaction excluded to prevent time-travel leakage)
- **`time_delta`** — minutes since customer's previous transaction
- **`amount_behavior_z`** — z-score of amount relative to customer history
- **`amount_30d_ratio`**, **`category_amount_z`**, **`new_category_for_customer`**, **`unique_categories_last10`**
- **`distance_from_last_txn`**, **`geo_velocity_kmph`** — geo-behavioral signals
- **`is_cold_start`**, **`ecommerce_ind`**, **`cash_indicator`**

Customer-level KYC features: `age`, `income`, `tenure` (days since onboarding), `sales`, `emp_count`, `is_biz`.

**Temporal split strategy**: customers are split 80/10/10 (train/val/test) by their *first-seen transaction time*, not randomly. This prevents future data leaking into training-set features.

---

### Stage 2 — Transaction Autoencoder (Anomaly Signal)

**Architecture:** `TransactionAutoencoder` — a 3-layer symmetric encoder/decoder.

```
Input (15 features) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(16) [bottleneck]
                   ← Linear(32) ← ReLU ← Linear(64) ← ReLU ← Linear(input_dim)
```

**Training schedule:**
1. **Pre-train** on *all* transactions (both BankSim + Scotiabank) for 10 epochs to learn general normal patterns
2. **Fine-tune** on labeled customer transactions only for 30 more epochs with fraud/legit oversampling

The reconstruction error per transaction is used as an **anomaly risk score**. This is aggregated per customer (`avg_txn_ae_error`) and attached to nodes and edges in the graph.

Output saved to: `outputs/transaction_autoencoder.pt`

---

### Stage 3 — Heterogeneous Customer Graph (FraudSAGE / DGI)

#### Graph Construction

A heterogeneous tripartite graph is built from the master transaction pool:

- **Node types**: `customer` (61,410 nodes, 16 features), `category` (117 nodes, one-hot identity), `city` (141 nodes, one-hot identity)
- **Edge types**: `customer → category` (via `purchases_at`), `customer → city` (via `transacts_in`), plus symmetric reverse edges
- **Edge attributes**: `[recency_weight, ae_risk_score]` — recency weight uses exponential decay $w = e^{-\lambda \Delta t}$ with $\lambda = 0.01$ per hour

Customer node features (16 total):
- KYC: `age`, `income`, `tenure`, `sales` (business revenue), `entity_type`
- Transaction aggregates: `avg_txn_amount`, `max_txn_amount`, `std_txn_amount`, `txn_count`, `cash_rate`, `ecom_rate`, `avg_24h_velocity`
- Anomaly signal: `avg_txn_ae_error` (from autoencoder), `dgi_anomaly_score`

Total edges: ~23.6M before `ToUndirected` transform.

#### Model: FraudSAGE

3-layer heterogeneous GraphSAGE with max-pooling aggregation (`hidden_channels=64`), defined in `lib/fraud_sage.py`:

**Layer 1**: `Category/City → Customer` — aggregates hub signals into initial customer embedding  
**Layer 2**: `Customer → Category/City` — pushes customer context back into hub nodes  
**Layer 3**: `Category/City → Customer` — final aggregation, residual from Layer 1 output

Each conv uses `MaxPoolConcatSAGEConv`: neighbor features pass through a 2-layer MLP, then max-pooled, then concatenated with the target node's self-projection: `concat([self_proj(x_dst), max_pool(neigh_mlp(x_src))]) → Linear(2d → d)`.

Additional components: `BatchNorm1d` after Layers 1 and 3, `identity_feature_dropout` (p=0.30 on KYC features during training), `ELU` activations, residual skip connections.

**Classifier head**: `Linear(64 → 32) → ELU → Dropout(0.2) → Linear(32 → 1)`

**Training**: `NeighborLoader` batches with `num_neighbors=[15, 10]` and `batch_size=512`.

#### Deep Graph Infomax (DGI) Pre-training

The encoder is trained **without labels** using the Deep Graph Infomax objective:
- **Positive pass**: real node features $(X, A)$
- **Negative pass**: shuffled (corrupted) node features $(\tilde{X}, A)$
- **Discriminator**: bilinear separation of local embeddings $h_i$ from global summary $s = \sigma(\text{mean}(H))$
- **Loss**: BCE distinguishing real vs. corrupted neighborhoods

Output: 64-dimensional customer embeddings saved to `outputs/dgi_customer_embeddings.npy`.

The model weights are saved to `outputs/fraud_sage_model.pth`. Artifacts (node features, edge indices, customer mapping) are saved to `outputs/sage_artifacts.pkl`.

---

### Stage 4 — Clustering & Pseudo-Label Generation

After the DGI encoder is frozen, customer embeddings are used to discover high-risk clusters.

**Suspicious slice selection**: customers whose DGI anomaly score exceeds the 98th percentile *or* whose AE risk exceeds the 95th percentile *or* who are known train-split fraud anchors.

**Clustering (MiniBatchKMeans)**: fitted on the suspicious slice only. Cluster count $K$ is derived as `len(suspicious_slice) // 32`, clamped to `[12, 96]`.

**HDBSCAN alternative**: also fitted on the suspicious slice for a second clustering view. A **consensus score** combines both: `0.6 × KMeans fraud lift + 0.4 × HDBSCAN fraud lift`.

**Pseudo-label propagation**: customers in clusters whose train-split fraud rate exceeds a multiplier threshold (3×) receive a `second_pass_positive` flag. A reliability gate (`cluster_reliability_gate`) combines KMeans lift ≥ 3, HDBSCAN lift ≥ 3, consensus score > 90th percentile, and suspicious-slice membership.

---

### Stage 5 — Embedding MLP

A 2-layer MLP is trained on frozen DGI embeddings:

```
Input (64 + 3 aux features) → Linear(64) → ReLU → Dropout(0.15)
                             → Linear(32) → ReLU → Dropout(0.15)
                             → Linear(1)  → Sigmoid
```

The 3 auxiliary features appended to embeddings are: `gmm_max_prob`, `customer_ae_risk_norm`, `component_confidence`.

Trained with pseudo-labels (`final_train_label`) and per-sample weights. Early stopping with patience=20 on validation PR-AUC + F1 composite. Output: `mlp_fraud_prob` per customer. Saved to `outputs/dgi_embedding_mlp.pt`.

---

### Stage 6 — LightGBM Ranker

A rich feature table is constructed combining:
- Embedding PCA projections (8 components)
- Cluster statistics (KMeans + HDBSCAN lift, sizes, coverage)
- kNN neighborhood signals (mean/max distance, suspicious share, gold fraud count)
- Centroid distances (fraud centroid, legit centroid, margin)
- All anomaly signals (DGI, AE, GMM)

**LightGBM** (`n_estimators=600`, `learning_rate=0.02`, `num_leaves=31`, `subsample=0.85`, `scale_pos_weight` from label ratio) is trained on gold labels + high-confidence pseudo-positives with sample weighting: `5.0 × gold fraud, 2.5 × pseudo-positive, 2.0 × hard negative, 1.0 × gold legit`.

Output: `lgb_fraud_prob` per customer.

---

### Stage 7 — PU Bagging Reranker

A positive-unlabeled bagging ensemble (12 bags) adds the formally conservative PU learning step:

1. **Review-priority score** combines all signals: `0.35 × model + 0.25 × consensus + 0.15 × DGI + 0.15 × AE + 0.10 × anchor proximity`
2. **Pseudo-positive promotion**: top-0.5% of suspicious unlabeled by review priority + consensus ≥ 95th percentile + anchor distance ≤ 25th percentile
3. Per bag: sample unlabeled set equal to 4× positives; train LightGBM on {positives + negatives + sampled unlabeled}; aggregate predictions across bags

Output: `pu_bagging_prob`. Ensemble: `scarcity_ensemble_prob = 0.55 × PU + 0.30 × model + 0.15 × priority`.

---

### Stage 8 — Semi-Supervised Reranking

A Random Forest (`n_estimators=500`, `max_depth=6`) is trained on a semi-supervised label set:
- `gold_fraud`: weight 10.0
- `gold_legit`: weight 1.0
- `pseudo_positive` (top 0.3% unlabeled): weight 0.35
- `pseudo_negative` (bottom 20% non-suspicious): weight 0.15

Trained on meta-features: all model scores, cluster signals, DGI/AE signals, kNN signals.

Output: `scarcity_semi_prob`. Final ensemble: `scarcity_semi_ensemble_prob = 0.55 × semi + 0.25 × scarcity + 0.10 × PU + 0.10 × consensus`.

---

### Stage 9 — Fraud-Anchor Expansion

The final labeling stage uses known train-split fraud anchors as reference points:

1. For each fraud anchor, find the nearest 25 suspicious unlabeled customers in embedding space
2. Score each neighbor by `anchor_support_score = 0.35 × model + 0.30 × cluster + 0.20 × proximity + 0.15 × reliability`
3. Select those whose distance ≤ 20th percentile AND support ≥ 80th percentile AND `cluster_reliability_gate ≥ 0.45`
4. Per-anchor cap of 8 neighbors; per-component cap of 8 neighbors

A second Random Forest (`n_estimators=600`, `max_depth=6`) is trained with these fraud–anchor pseudo-positives (weight 0.25).

**Final score**: `scarcity_anchor_ensemble_prob = 0.45 × anchor_RF + 0.25 × semi_ensemble + 0.15 × scarcity_ensemble + 0.15 × cluster_reliability_gate`

All scores are saved to `outputs/rank_df_with_anchor_expansion.csv.gz`.

---

### Stage 10 — Explainability (GNNExplainer)

Implemented in `explainability_gnnexplainer.ipynb`.

**GNNExplainer on original model**: per-customer subgraphs are extracted with `NeighborLoader(num_neighbors=[20, 15, 10])` before running GNNExplainer (epochs=20–25). Running on the full 23.6M-edge graph is infeasible (~45 min per customer); NeighborLoader reduces this to ~2.3 seconds for 5 customers on CUDA.

Edge mask retention rates on the original graph (5 random customers):
- Mean edge keep > 50%: ~9%
- Mean edge keep > 90%: ~0%

This confirms the masks are selective (not diffuse), retaining only a small subset of neighboring edges as most explanatory.

**Cohort analysis (20 customers)**: 10 confirmed-fraud + 10 confirmed-legit-but-high-scored ('false positives') are analyzed. For each:
- Transaction behavior is summarized (`txn_count`, `mean_amount`, `night_ratio`, `cash_ratio`, tipping point detection)
- A 3-sentence investigation narrative is generated via Ollama (`gemma2:2b`) or a deterministic template fallback
- GNNExplainer retention metrics are computed per customer

**Cluster interpretation**: GMM component profiles (fraud rate, size, AE risk, embedding variance) are sent to Ollama/gemma2:2b for short narrative cluster labels.

Outputs: `outputs/cohort_20_gnn_mask_heatmap.png`, `outputs/cohort_20_explanations.csv`, `outputs/cohort_20_gnn_mask_metrics.csv`.

---

## Score Lineage

```
Transaction AE reconstruction error
    ↓
Customer AE risk signal ─────────────────────────────────────────────────────┐
                                                                              │
DGI (GNN self-supervised) embeddings                                         │
    ↓                                                                         │
Suspicious-slice clustering (KMeans + HDBSCAN)                               │
    ↓                                                                         │
Pseudo-labels (component fraud rate propagation)                             │
    ↓                                                                         │
MLP on frozen embeddings (mlp_fraud_prob)                                    │
    ↓                                                                         │
LightGBM ranker (lgb_fraud_prob) ◄───── rich feature table (all signals) ◄──┘
    ↓
PU bagging reranker (pu_bagging_prob)
    ↓
Semi-supervised Random Forest (scarcity_semi_prob)
    ↓
Fraud-anchor expansion Random Forest (scarcity_anchor_prob)
    ↓
FINAL: scarcity_anchor_ensemble_prob
```

---

## Key Artifacts

| File | Description |
|---|---|
| `outputs/fraud_sage_model.pth` | FraudSAGE model weights |
| `outputs/sage_artifacts.pkl` | Graph tensors: `x_cust`, `x_cat`, `x_city`, `edge_cust_cat`, `edge_cust_city`, `cust_map` |
| `outputs/dgi_customer_embeddings.npy` | 64D customer embeddings (61410 × 64) |
| `outputs/dgi_embedding_mlp.pt` | MLP weights + metadata |
| `outputs/dgi_gmm.joblib` | Fitted GMM model |
| `outputs/rf_proxy.joblib` | (Legacy) upstream RF proxy |
| `outputs/transaction_autoencoder.pt` | AE weights + scaler |
| `outputs/rank_df_with_anchor_expansion.csv.gz` | Full per-customer scoring table (final pipeline output) |
| `outputs/model_output.csv` | The final scored customer table |

---

## Label Distribution

| Category | Count |
|---|---|
| Confirmed fraud (`true_label=1`) | 10 |
| Confirmed legit (`true_label=0`) | 990 |
| Unlabeled (`true_label=-1`) | 60,410 |
| **Total customers** | **61,410** |

The `lgb_label` column marks 18 positives (10 gold + 8 high-confidence pseudo-positives). Only 18 positive training examples are available for supervised stages.

---

## Limitations and Improvement Suggestions

The following limitations were identified from code analysis and from examining the cohort explanation outputs (`outputs/cohort_20_explanations.csv`).

### 1. Label Scarcity Creates a Weak Supervision Ceiling

**Problem**: With only 10 confirmed fraud customers, every downstream supervised model is trained on near-zero positive samples. The LightGBM ranker, PU bagging, and Random Forests are all fundamentally limited by this. Bootstrap confidence intervals on Average Precision are essentially meaningless at this scale. Precision@K and Recall@K at K=5, 10, 25 are all computed over a test set with fewer than 5 fraud customers — making the metrics extremely high-variance.

**Evidence from explanations**: Several explanations acknowledge low model scores for confirmed-fraud customers (e.g., `scarcity_anchor_ensemble_prob: 0.1913` for one true-fraud customer marked `SYNID0100957188`), suggesting the model is not confidently ranking all true positives highly.

**Suggestion**: Invest in an active label acquisition loop. The existing `outputs/active_review_queue_v2.csv` is a good starting point — prioritize reviewing the top 50 customers in that queue to quickly expand from 10 to 30-50 confirmed fraud labels. Each new confirmed label has an outsized effect on model quality given the current scale.

### 2. Cluster Labels Are Semantically Poor and Often Contradictory

**Problem**: LLM-generated cluster labels (via `gemma2:2b`) are demonstrably unreliable. The cohort explanations frequently show customers labeled with their cluster's narrative even when it contradicts the fraud signal. Examples from `cohort_20_explanations.csv`:
- A confirmed-fraud customer (`SYNID0107334515`, `true_label=1.0`) is in a cluster labeled **"Normal Users"**
- Another confirmed fraud (`SYNID0200755995`) is in cluster **"Regular Users"**
- A confirmed fraud (`SYNID0200496670`) is in cluster **"Genuine Users"**

The `gemma2:2b` model is too small (2B parameters) and its 2-3 word cluster labels add noise. The `mismatch_rate` penalty in the v2/v3 recalibration attempts to correct this but has limited precision.

**Suggestion**: Replace per-cluster LLM labeling with deterministic statistical descriptors (e.g., "High-AE-risk, 50% fraud rate, 58 members") that don't rely on LLM quality. Reserve LLM calls for the final per-customer investigation notes (the 3-sentence explanation for individual customers), where transaction context can ground the output.

### 3. False Positive Definition Is Data-Dependent and Fragile

**Problem**: The "false positive" cohort is defined as confirmed-legit customers (`true_label=0`) with the highest fraud scores. This is a valid operational definition (the model over-scores them), but the confirmation gap between 10 fraud labels and 990 legit labels creates asymmetric confidence. The highest-scored legit customer only reaches a score of 0.617 — far below what would be a confident fraud prediction in a calibrated model.

**Suggestion**: Separate the staging of false positives by score tier: `score ≥ 0.5` vs. `0.3–0.5` vs. `< 0.3`. This would allow analysts to distinguish "model confidently wrong" from "model mildly suspicious". Currently all 10 selected false positives have scores between 0.10 and 0.62 — a wide range that conflates different failure modes.

### 4. GNNExplainer Masks Are Near-Zero at Absolute Thresholds

**Problem**: Raw edge mask values from GNNExplainer cluster tightly (≈0.22 for all edges) rather than spreading across [0, 1]. Using absolute thresholds (>0.5, >0.9) on the original graph yields near-zero retention (9% at >50% on the NeighborLoader verification). The masks only show meaningful spread when per-customer min-max normalized.

**Evidence**: The verification notebook shows edge keep >90% = 0% on the original model across all 5 tested customers.

**Suggestion**: Switch to a **gradient-based attribution method** (Integrated Gradients or AttentionExplainer) instead of GNNExplainer's iterative mask optimization, which is known to produce diffuse masks on heterogeneous graphs. Alternatively, post-hoc calibration by normalizing within each customer's subgraph should be applied consistently before computing metrics and heatmaps. The calibrated verification cell (surrogate kNN graph) showed 49% of edges keeping >50% normalized relevance, showing the masks do contain structure when properly scaled.

### 5. LLM Explanation Quality Is Inconsistent

**Problem**: The Ollama LLM endpoint is remote and flaky. Several explanations in `cohort_20_explanations.csv` either fell through to template fallbacks or produced explanations that:
- Contradict the cluster label (fraud customer in "Normal Users" explained as legitimately anomalous)
- Cite the raw probability score directly rather than using behavioral language
- Reference "the model score strongly suggests" without connecting to concrete transaction patterns

Example: One explanation for `SYNID0200755995` (true fraud, score 0.950) says "the model suggests a strong probability of fraudulent activity" but provides no concrete behavioral evidence from the 74 transactions summarized.

**Suggestion**: Add stricter prompt constraints around behavioral specificity. The tipping point detection logic (`tipping_point` field) is already calculating whether a rolling 6-transaction window shows an anomaly shift — this should be explicitly injected into the prompt. Add a validation pass that rejects explanations shorter than 150 characters or containing template phrases like "based on the model score" without supporting transaction evidence.

### 6. Edge Recency Decay Is Uniform Across Edge Types

**Problem**: Edges use a single decay function $w = e^{-0.01 \Delta t_{\text{hours}}}$ regardless of whether the edge is a `purchases_at` (merchant category) or `transacts_in` (city) relationship. Merchant category edges likely have longer behavioral relevance windows than city edges.

**Suggestion**: Allow different decay $\lambda$ per edge type. For category edges, a slower decay (e.g., $\lambda = 0.005$) would better capture stable spending habits. For city edges, a faster decay (e.g., $\lambda = 0.02$) would more aggressively weight recent geo-location.

### 7. Temporal Split Places Almost All Gold Labels in Train

**Problem**: The 80/10/10 temporal split by customer first-seen time creates an extreme imbalance: 8 of the 10 confirmed-fraud customers end up in the training split, with only 1–2 in val/test. This means validation and test evaluation metrics are computed over effectively 1–2 positive examples, making PR-AUC estimates near-useless.

**Suggestion**: Consider a stratified temporal split that ensures at least 2–3 fraud customers appear in each evaluation partition. While this slightly violates strict temporal ordering, the current arrangement produces uninterpretable evaluation curves. Alternatively, use leave-one-out cross-validation across the 10 fraud customers.

### 8. The Pseudo-Label Cascade Amplifies Early Errors

**Problem**: The pipeline has 5 pseudo-label generation stages (component propagation → `second_pass_positive` → PU promotion → semi-supervised → anchor expansion). Each stage can introduce systematic errors that compound downstream. Specifically, if Stage 4 (clustering) mislabels a genuinely legitimate customer as suspicious, that customer can flow through all five positive-label stages and accumulate weight 0.25 in the final Random Forest.

**Evidence**: The `label_source_anchor` column shows a `fraud_anchor_pseudo_positive` category, meaning some customers were elevated because they are embedding-space neighbors of the 10 confirmed fraud anchors — without any independent evidence of fraud.

**Suggestion**: Add a **label consistency check** between pipeline stages: flag any customer who receives a pseudo-positive label from more than one stage but has no model score above the 80th percentile. These are candidates for analyst review before being used as training data. Introduce a hard stop at PU promotion that limits pseudo-positives to 4× the number of gold fraud labels (≤ 40 promoted at any time).
