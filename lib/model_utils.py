"""
model_utils.py — Model loading, inference, and SHAP explanation utilities.

All heavy artifacts are loaded once via @st.cache_resource.
The score_new_customers function is inductive: no retraining needed for new nodes.
"""

import warnings
import json
import os
import pickle

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.resource_paths import resolve_output_path

warnings.filterwarnings("ignore", message=".*do not occur as destination type.*")

# ── FraudSAGE architecture (must match training exactly) ─────────────────────

try:
    from torch_geometric.nn import HeteroConv, Linear
    from lib.fraud_sage import MaxPoolConcatSAGEConv
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import NeighborLoader
    import torch_geometric.transforms as T
    TORCH_GEO_AVAILABLE = True
except ImportError:
    TORCH_GEO_AVAILABLE = False


class FraudSAGE(nn.Module):
    """
    Heterogeneous GraphSAGE — 3-layer message passing.
    Architecture must be identical to the model saved in fraud_sage_model.pth.
    """

    def __init__(self, hidden_channels: int = 64):
        super().__init__()
        if not TORCH_GEO_AVAILABLE:
            raise ImportError("torch_geometric is required for FraudSAGE.")

        self.conv1 = HeteroConv(
            {
                ("category", "rev_purchases_at", "customer"): MaxPoolConcatSAGEConv(
                    (-1, -1), hidden_channels
                ),
                ("city", "rev_transacts_in", "customer"): MaxPoolConcatSAGEConv(
                    (-1, -1), hidden_channels
                ),
            },
            aggr="sum",
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = HeteroConv(
            {
                ("customer", "purchases_at", "category"): MaxPoolConcatSAGEConv(
                    (-1, -1), hidden_channels
                ),
                ("customer", "transacts_in", "city"): MaxPoolConcatSAGEConv(
                    (-1, -1), hidden_channels
                ),
            },
            aggr="sum",
        )

        self.conv3 = HeteroConv(
            {
                ("category", "rev_purchases_at", "customer"): MaxPoolConcatSAGEConv(
                    (-1, -1), hidden_channels
                ),
                ("city", "rev_transacts_in", "customer"): MaxPoolConcatSAGEConv(
                    (-1, -1), hidden_channels
                ),
            },
            aggr="sum",
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.input_proj = Linear(-1, hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x_dict, edge_index_dict):
        res = F.elu(self.input_proj(x_dict["customer"]))

        h1 = self.conv1(x_dict, edge_index_dict)
        h1["customer"] = F.elu(self.bn1(h1["customer"]) + res)
        for k in x_dict:
            if k not in h1:
                h1[k] = x_dict[k]

        h2 = self.conv2(h1, edge_index_dict)
        h2 = {k: F.elu(v) for k, v in h2.items()}
        if "customer" not in h2:
            h2["customer"] = h1["customer"]

    def encode(self, x_dict, edge_index_dict):
        """Return 64-d customer embedding (before classifier head)."""
        res = F.elu(self.input_proj(x_dict["customer"]))

        h1 = self.conv1(x_dict, edge_index_dict)
        h1["customer"] = F.elu(self.bn1(h1["customer"]) + res)
        for k in x_dict:
            if k not in h1:
                h1[k] = x_dict[k]

        h2 = self.conv2(h1, edge_index_dict)
        h2 = {k: F.elu(v) for k, v in h2.items()}
        if "customer" not in h2:
            h2["customer"] = h1["customer"]

        h3 = self.conv3(h2, edge_index_dict)
        h3["customer"] = F.elu(self.bn2(h3["customer"]) + h1["customer"])
        return h3["customer"]

    def forward(self, x_dict, edge_index_dict):
        return self.classifier(self.encode(x_dict, edge_index_dict))


class EmbeddingMLP(nn.Module):
    """Reproduce dgi_embedding_mlp.pt architecture for webapp inference."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Artifact loading with Streamlit caching ───────────────────────────────────

def load_sage_model(model_path: str | None = None, artifacts: dict = None):
    """Load FraudSAGE weights into model. Returns model on eval mode."""
    if not TORCH_GEO_AVAILABLE:
        return None
    if model_path is None:
        model_path = resolve_output_path("fraud_sage_model.pth")
    hidden = 64
    if artifacts and "hidden_channels" in artifacts:
        hidden = artifacts["hidden_channels"]
    model = FraudSAGE(hidden_channels=hidden)
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # Initialise lazy modules with a dummy forward pass before loading weights
    # We need minimal tensors that match the feature sizes
    if artifacts:
        x_cust_sample = artifacts["x_cust"][:2]
        x_cat_sample = artifacts["x_cat"][:2]
        x_city_sample = artifacts["x_city"][:2]
        dummy_data = HeteroData()
        dummy_data["customer"].x = x_cust_sample
        dummy_data["category"].x = x_cat_sample
        dummy_data["city"].x = x_city_sample
        # Minimal edges: customer 0 → category 0, customer 0 → city 0
        dummy_data["customer", "purchases_at", "category"].edge_index = torch.tensor(
            [[0], [0]], dtype=torch.long
        )
        dummy_data["customer", "transacts_in", "city"].edge_index = torch.tensor(
            [[0], [0]], dtype=torch.long
        )
        dummy_data = T.ToUndirected()(dummy_data)
        with torch.no_grad():
            model(dummy_data.x_dict, dummy_data.edge_index_dict)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_artifacts(artifacts_path: str | None = None) -> dict:
    """Load inference artifacts (scaler, maps, embeddings, edge tensors)."""
    if artifacts_path is None:
        artifacts_path = resolve_output_path("sage_artifacts.pkl")
    with open(artifacts_path, "rb") as f:
        return pickle.load(f)


def load_rf_model(rf_dir: str | None = None):
    """Load the ranker model (LGB) and SHAP explainer from rf_model_sage/."""
    if rf_dir is None:
        rf_dir = resolve_output_path()

    candidates = [
        rf_dir,
        os.path.join(rf_dir, "rf_model_sage"),
    ]
    resolved_dir = None
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "rf_proxy.joblib")):
            resolved_dir = cand
            break
    if resolved_dir is None:
        raise FileNotFoundError(
            f"Could not find RF artifacts in any of: {candidates}"
        )

    rf = joblib.load(os.path.join(resolved_dir, "rf_proxy.joblib"))
    explainer = joblib.load(os.path.join(resolved_dir, "shap_explainer.joblib"))
    with open(os.path.join(resolved_dir, "meta.json")) as f:
        meta = json.load(f)
    return rf, explainer, meta


def load_lgb_bundle(outputs_dir: str | None = None):
    """
    Load the full scoring bundle needed for inductive LGB inference:
      LGBMClassifier, GaussianMixture, EmbeddingMLP.

    Returns (lgb_model, gmm_model, mlp_model).
    mlp_model may be None if dgi_embedding_mlp.pt is absent.
    """
    if outputs_dir is None:
        outputs_dir = resolve_output_path()

    lgb_model = joblib.load(os.path.join(outputs_dir, "lgbm_fraud_ranker.joblib"))
    gmm_model = joblib.load(os.path.join(outputs_dir, "dgi_gmm.joblib"))

    mlp_path = os.path.join(outputs_dir, "dgi_embedding_mlp.pt")
    mlp_model = None
    if os.path.isfile(mlp_path):
        ckpt = torch.load(mlp_path, map_location="cpu", weights_only=False)
        mlp = EmbeddingMLP(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt.get("hidden_dim", 64),
            dropout=ckpt.get("dropout", 0.15),
        )
        mlp.load_state_dict(ckpt["model_state_dict"])
        mlp.eval()
        mlp_model = mlp

    return lgb_model, gmm_model, mlp_model


def _compute_lgb_features_from_embedding(
    embedding: np.ndarray,
    artifacts: dict,
    gmm,
    mlp_model,
) -> dict:
    """
    Compute all LGB_FEATURES from a single 64-d GNN embedding.

    Uses precomputed artifacts (PCA, anchors, centroids, component stats)
    saved inside sage_artifacts.pkl by the build_inference_artifacts script.
    """
    emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
    medians: dict = artifacts.get("lgb_training_medians", {})

    # ── PCA ──────────────────────────────────────────────────────────────────
    pca = artifacts.get("pca_model")
    pca_features = pca.transform(emb)[0] if pca is not None else np.zeros(8, dtype=np.float32)

    # ── GMM ──────────────────────────────────────────────────────────────────
    gmm_probs = gmm.predict_proba(emb)[0]
    gmm_max_prob = float(gmm_probs.max())
    component = int(gmm_probs.argmax())
    component_confidence = gmm_max_prob

    # ── Anchor distances ─────────────────────────────────────────────────────
    fraud_anchors = artifacts.get("fraud_anchor_embeddings")
    legit_anchors = artifacts.get("legit_anchor_embeddings")

    if fraud_anchors is not None and len(fraud_anchors) > 0:
        d_fraud = np.linalg.norm(fraud_anchors - emb, axis=1)
        min_dist_fraud = float(d_fraud.min())
        mean_dist_fraud = float(d_fraud.mean())
        anchor_proximity = float(1.0 / (1.0 + min_dist_fraud))
        # anchors within 3× nearest-fraud distance are "gold fraud neighbors"
        knn_fraud_count = int(np.sum(d_fraud <= min_dist_fraud * 3.0))
    else:
        min_dist_fraud = float(medians.get("min_dist_to_fraud_anchor", 5.0))
        mean_dist_fraud = float(medians.get("mean_dist_to_fraud_anchor", 5.0))
        anchor_proximity = float(medians.get("anchor_proximity_score", 0.1))
        knn_fraud_count = 0

    if legit_anchors is not None and len(legit_anchors) > 0:
        d_legit = np.linalg.norm(legit_anchors - emb, axis=1)
        min_dist_legit = float(d_legit.min())
        knn_mean_dist = float(np.sort(d_legit)[:25].mean())
        # suspicious share: fraction of 25 nearest anchors that are fraud
        total_near = 25
        knn_susp_share = float(knn_fraud_count / max(total_near, 1))
    else:
        min_dist_legit = float(medians.get("min_dist_to_legit_anchor", 5.0))
        knn_mean_dist = float(medians.get("knn_mean_distance", 5.0))
        knn_susp_share = float(medians.get("knn_suspicious_share", 0.05))

    # ── Centroids ─────────────────────────────────────────────────────────────
    fraud_c = np.array(artifacts.get("fraud_centroid", np.zeros(64)), dtype=np.float32)
    legit_c = np.array(artifacts.get("legit_centroid", np.zeros(64)), dtype=np.float32)
    dist_fraud_c = float(np.linalg.norm(emb - fraud_c))
    dist_legit_c = float(np.linalg.norm(emb - legit_c))
    centroid_margin = dist_legit_c - dist_fraud_c

    # ── DGI anomaly score (GMM log-likelihood normalized) ────────────────────
    lp = float(gmm.score(emb))
    stats = artifacts.get("dgi_log_prob_stats", {})
    lp_max = float(stats.get("p95", 70.0))
    lp_min = float(stats.get("p5", -50.0))
    dgi_anomaly = float(np.clip((lp_max - lp) / (lp_max - lp_min + 1e-8), 0.0, 1.0))

    # ── MLP fraud probability ─────────────────────────────────────────────────
    ae_risk = float(medians.get("customer_ae_risk_norm", 0.5))
    if mlp_model is not None:
        aux = np.array([[gmm_max_prob, ae_risk, component_confidence]], dtype=np.float32)
        mlp_in = np.concatenate([emb.astype(np.float32), aux], axis=1)
        with torch.no_grad():
            mlp_fraud_prob = float(
                torch.sigmoid(mlp_model(torch.tensor(mlp_in))).item()
            )
    else:
        mlp_fraud_prob = float(medians.get("mlp_fraud_prob", 0.1))

    # ── Per-component cluster stats ───────────────────────────────────────────
    comp_stats: dict = artifacts.get("lgb_component_stats", {})
    cs = comp_stats.get(component, comp_stats.get(str(component), {}))

    def _cs(key, fallback=0.0):
        return float(cs.get(key, medians.get(key, fallback)))

    return {
        **{f"emb_pca_{i + 1}": float(pca_features[i]) for i in range(len(pca_features))},
        "mlp_fraud_prob": mlp_fraud_prob,
        "dgi_anomaly_score": dgi_anomaly,
        "customer_ae_risk_norm": ae_risk,
        "gmm_max_prob": gmm_max_prob,
        "component_confidence": component_confidence,
        "min_dist_to_fraud_anchor": min_dist_fraud,
        "mean_dist_to_fraud_anchor": mean_dist_fraud,
        "min_dist_to_legit_anchor": min_dist_legit,
        "anchor_proximity_score": anchor_proximity,
        "knn_gold_fraud_count": float(knn_fraud_count),
        "knn_mean_distance": knn_mean_dist,
        "knn_suspicious_share": knn_susp_share,
        "dist_to_fraud_centroid": dist_fraud_c,
        "dist_to_legit_centroid": dist_legit_c,
        "centroid_margin": centroid_margin,
        "cluster_consensus_score": _cs("cluster_consensus_score", 0.05),
        "hdb_outlier_score": _cs("hdb_outlier_score", 0.0),
        "km_component_size": _cs("km_component_size", 300.0),
        "km_component_train_fraud_rate": _cs("km_component_train_fraud_rate", 0.005),
        "km_component_mean_dgi": _cs("km_component_mean_dgi", 0.5),
        "hdb_component_size": _cs("hdb_component_size", 300.0),
        "hdb_component_fraud_rate_labeled": _cs("hdb_component_fraud_rate_labeled", 0.005),
        "hdb_component_fraud_lift_labeled": _cs("hdb_component_fraud_lift_labeled", 1.0),
        "eft_amount_match_count": 0.0,
        "abm_dc_colocated": 0.0,
    }


# ── Transaction AE risk helper ───────────────────────────────────────────────────

def compute_ae_risk_from_transactions(transactions: list, ae_checkpoint_path: str | None = None) -> float:
    """
    Compute a behavioural risk signal from a live transaction list for the
    ``customer_ae_risk_norm`` LGB feature.

    For new customers (cold-start), the transaction autoencoder is dominated
    by the cold-start flag and gives the same high score regardless of velocity.
    Instead we compute a targeted signal from three fraud-specific dimensions:

    1. **Velocity burst** — max 1-hour transaction count / 20 (normalised).
    2. **Structuring** — amount uniformity (low std/mean for ≥5 txns).
    3. **Geographic impossibility** — max kmph geo velocity / 1 000.

    Returns a value in [0, 1].  ~0.5 matches the training population median
    (which is what LGB expects for an "average" customer).
    """
    if not transactions:
        return 0.5  # neutral — matches training median

    try:
        from lib.explainability import build_transaction_ae_features
        feats = build_transaction_ae_features(transactions)
        if feats.empty:
            return 0.5

        # ── 1. Velocity burst ─────────────────────────────────────────────────
        max_v1h = float(feats["velocity_1h"].max())
        velocity_score = min(max_v1h / 20.0, 1.0)

        # ── 2. Structuring (uniform amount + burst) ──────────────────────────
        amounts = np.array([t["amount"] for t in transactions], dtype=float)
        if len(amounts) >= 5:
            std_ratio = float(np.std(amounts) / (np.mean(amounts) + 1e-8))
            # Suspicious when std_ratio is very low (all same amount)
            uniformity = max(0.0, 1.0 - min(std_ratio * 5.0, 1.0))
            # Only charge the structuring signal when velocity is also elevated
            structuring_score = uniformity * velocity_score
        else:
            structuring_score = 0.0

        # ── 3. Geographic impossibility ───────────────────────────────────────
        max_geo = float(feats["geo_velocity_kmph"].max())
        geo_score = min(max_geo / 1000.0, 1.0)

        # ── Combine: velocity is the primary driver ───────────────────────────
        raw = float(0.50 * velocity_score + 0.35 * structuring_score + 0.15 * geo_score)
        # Shift into the [0,1] range while anchoring the zero-activity baseline at 0.5
        return float(np.clip(0.5 + raw, 0.0, 1.0))
    except Exception:
        return 0.5


# ── Inductive inference ───────────────────────────────────────────────────────

def score_new_customers(
    new_customer_records: list,
    model,
    artifacts: dict,
    device=None,
    lgb_bundle=None,
) -> pd.DataFrame:
    """
    Score brand-new customers inductively — no retraining required.

    Parameters
    ----------
    new_customer_records : list[dict]
        Each dict must supply keys from CUSTOMER_FEATURE_COLS plus optionally:
          * customer_id          — identifier (default: "NEW_{i}")
          * merchant_categories  — list of category strings
          * cities               — list of city name strings

    lgb_bundle : optional tuple (lgb_model, gmm_model, mlp_model)
        When provided, the full DGI → PCA → GMM → MLP → LGB pipeline is run
        and the LGB fraud probability is returned as risk_score (recommended).
        Without it, the raw FraudSAGE sigmoid is used (less calibrated).

    Returns
    -------
    pd.DataFrame  columns: customer_id, risk_score, gnn_score, risk_tier,
                            predicted_label, lgb_features
    """
    if not TORCH_GEO_AVAILABLE or model is None:
        results = []
        for i, rec in enumerate(new_customer_records):
            results.append(
                {
                    "customer_id": rec.get("customer_id", f"NEW_{i}"),
                    "risk_score": 0.0,
                    "gnn_score": 0.0,
                    "risk_tier": "LOW",
                    "predicted_label": 0,
                    "lgb_features": {},
                }
            )
        return pd.DataFrame(results)

    if device is None:
        device = next(model.parameters()).device

    scaler = artifacts["cust_scaler"]
    feat_cols = artifacts["customer_feature_cols"]
    cat_map_a = artifacts["cat_map"]
    city_map_a = artifacts["city_map"]
    x_cust_base = artifacts["x_cust"]
    x_cat_base = artifacts["x_cat"]
    x_city_base = artifacts["x_city"]
    ei_cc = artifacts["edge_cust_cat"]
    ei_cct = artifacts["edge_cust_city"]
    num_neighbors = artifacts.get("num_neighbors", [15, 10])
    N_existing = x_cust_base.shape[0]

    # Step 1: feature vectors
    new_feats = [[float(rec.get(c, 0.0)) for c in feat_cols] for rec in new_customer_records]
    new_feats_raw = np.array(new_feats, dtype=np.float32)
    new_feats_scaled = scaler.transform(new_feats_raw)
    new_feats_scaled = np.nan_to_num(new_feats_scaled, nan=0.0, posinf=3.0, neginf=-3.0)
    x_new = torch.tensor(new_feats_scaled, dtype=torch.float)

    # Step 2: augmented customer node tensor
    x_cust_aug = torch.cat([x_cust_base, x_new], dim=0)
    new_node_ids = list(range(N_existing, N_existing + len(new_customer_records)))

    # Step 3: build new edges to category / city
    new_cat_src, new_cat_dst = [], []
    new_city_src, new_city_dst = [], []

    for local_i, rec in enumerate(new_customer_records):
        nid = new_node_ids[local_i]
        for cat_name in rec.get("merchant_categories", []):
            cat_idx_val = cat_map_a.get(cat_name)
            if cat_idx_val is not None:
                new_cat_src.append(nid)
                new_cat_dst.append(cat_idx_val)
        for city_name in rec.get("cities", []):
            city_key = city_name.upper() if city_name else "UNKNOWN"
            city_idx_val = city_map_a.get(city_key)
            if city_idx_val is not None:
                new_city_src.append(nid)
                new_city_dst.append(city_idx_val)

    if new_cat_src:
        extra_cc = torch.tensor([new_cat_src, new_cat_dst], dtype=torch.long)
        ei_cc_aug = torch.cat([ei_cc, extra_cc], dim=1)
    else:
        ei_cc_aug = ei_cc

    if new_city_src:
        extra_city = torch.tensor([new_city_src, new_city_dst], dtype=torch.long)
        ei_cct_aug = torch.cat([ei_cct, extra_city], dim=1)
    else:
        ei_cct_aug = ei_cct

    # Step 4: temporary HeteroData graph
    tmp = HeteroData()
    tmp["customer"].x = x_cust_aug
    tmp["category"].x = x_cat_base
    tmp["city"].x = x_city_base
    tmp["customer", "purchases_at", "category"].edge_index = ei_cc_aug
    tmp["customer", "transacts_in", "city"].edge_index = ei_cct_aug
    tmp = T.ToUndirected()(tmp)

    # Step 5: NeighborLoader on new nodes
    seed_mask = torch.zeros(x_cust_aug.shape[0], dtype=torch.bool)
    for nid in new_node_ids:
        seed_mask[nid] = True

    loader = NeighborLoader(
        tmp,
        num_neighbors=num_neighbors,
        batch_size=len(new_node_ids),
        input_nodes=("customer", seed_mask),
        shuffle=False,
        num_workers=0,
    )

    # Step 6: forward — get BOTH embedding and logit
    temperature = artifacts.get("temperature_demo", 1.0)

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader)).to(device)
        embeddings = model.encode(batch.x_dict, batch.edge_index_dict)
        new_embs = embeddings[: len(new_node_ids)].cpu().numpy()  # (n_new, 64)
        logits = model.classifier(embeddings)
        logits_scaled = logits.squeeze()[: len(new_node_ids)] / temperature
        gnn_probs = torch.sigmoid(logits_scaled).cpu().numpy()

    gnn_probs = np.atleast_1d(gnn_probs)

    # Step 7: optional full LGB pipeline for calibrated risk score
    results = []
    for i, rec in enumerate(new_customer_records):
        gnn_score = float(gnn_probs[i])
        lgb_features: dict = {}

        if lgb_bundle is not None:
            lgb_model, gmm, mlp_model = lgb_bundle
            lgb_features = _compute_lgb_features_from_embedding(
                new_embs[i], artifacts, gmm, mlp_model
            )
            # Override AE risk with live-computed value when transactions are available
            ae_risk_live = float(rec.get("_ae_risk_norm", -1.0))
            if ae_risk_live >= 0.0:
                lgb_features["customer_ae_risk_norm"] = ae_risk_live
            # Transaction-specific features that ARE in LGB_FEATURES
            lgb_features["eft_amount_match_count"] = float(rec.get("eft_amount_match_count", 0.0))
            lgb_features["abm_dc_colocated"] = float(rec.get("abm_dc_colocated", 0.0))
            feat_names: list = artifacts.get("lgb_feature_names", [])
            import pandas as _pd
            X = _pd.DataFrame(
                [[lgb_features.get(f, 0.0) for f in feat_names]],
                columns=feat_names,
            )
            score = float(lgb_model.predict_proba(X)[0, 1])

            # ── Velocity/structuring boost ───────────────────────────────────
            # cluster_consensus_score is the dominant LGB feature (importance
            # 280 k) but is always 0 for new customers who land in a neutral
            # cluster.  The ae_risk_norm captures live transaction behaviour
            # (velocity bursts, structuring) that LGB cannot see via the graph
            # embedding alone.  Blend it in as a multiplicative pull toward 1.
            ae_risk_live_val = lgb_features.get("customer_ae_risk_norm", 0.5)
            ae_signal = max(0.0, (ae_risk_live_val - 0.5) * 3.0)
            if ae_signal > 0.0:
                score = min(1.0, score + ae_signal * (1.0 - score))
        else:
            score = gnn_score

        tier = "HIGH" if score >= 0.70 else ("MEDIUM" if score >= 0.40 else "LOW")
        results.append(
            {
                "customer_id": rec.get("customer_id", f"NEW_{i}"),
                "risk_score": round(score, 6),
                "gnn_score": round(gnn_score, 6),
                "risk_tier": tier,
                "predicted_label": int(score >= 0.5),
                "lgb_features": lgb_features,
            }
        )
    return pd.DataFrame(results)


# ── RF Proxy + SHAP inference ──────────────────────────────────────────────────

FEATURE_DESCRIPTIONS = {
    # Transaction-level features
    "age": "customer age",
    "income": "annual income",
    "tenure": "account tenure (days)",
    "tenure_days": "account tenure (days)",
    "sales": "annual sales (businesses)",
    "annual_sales": "annual sales (businesses)",
    "emp_count": "employee headcount",
    "employee_count": "employee headcount",
    "is_biz": "business account type",
    "is_business": "business account type",
    "avg_txn_amount": "average transaction amount",
    "max_txn_amount": "maximum single transaction",
    "std_txn_amount": "transaction amount variability",
    "txn_count": "total transaction volume",
    "cash_rate": "cash withdrawal frequency",
    "cash_withdrawal_rate": "cash withdrawal frequency",
    "ecom_rate": "e-commerce transaction rate",
    "ecommerce_rate": "e-commerce transaction rate",
    "avg_24h_velocity": "24-hour transaction velocity",
    "avg_amount_zscore": "transaction amount z-score",
    "unique_categories": "unique merchant categories",
    "unique_cities": "unique transaction cities",
    "min_time_delta": "min minutes between transactions",
    "time_span_hours": "transaction window (hours)",
    "geo_velocity": "geographic velocity (cities/hr) — impossible multi-city pattern",
    # LGB model features (graph + cluster signals)
    "emb_pca_1": "graph embedding PCA component 1",
    "emb_pca_2": "graph embedding PCA component 2",
    "emb_pca_3": "graph embedding PCA component 3",
    "emb_pca_4": "graph embedding PCA component 4",
    "emb_pca_5": "graph embedding PCA component 5",
    "emb_pca_6": "graph embedding PCA component 6",
    "emb_pca_7": "graph embedding PCA component 7",
    "emb_pca_8": "graph embedding PCA component 8",
    "mlp_fraud_prob": "neural network fraud probability",
    "dgi_anomaly_score": "graph anomaly score (DGI)",
    "customer_ae_risk_norm": "autoencoder reconstruction risk",
    "cluster_consensus_score": "cluster consensus fraud signal",
    "anchor_proximity_score": "proximity to confirmed fraud anchors",
    "min_dist_to_fraud_anchor": "distance to nearest fraud anchor",
    "mean_dist_to_fraud_anchor": "mean distance to fraud anchors",
    "min_dist_to_legit_anchor": "distance to nearest legitimate anchor",
    "knn_gold_fraud_count": "fraud cases among nearest graph neighbors",
    "knn_mean_distance": "mean distance of nearest neighbors",
    "knn_suspicious_share": "fraction of suspicious nearest neighbors",
    "gmm_max_prob": "GMM cluster membership probability",
    "component_confidence": "cluster assignment confidence",
    "km_component_size": "KMeans cluster size",
    "km_component_train_fraud_rate": "KMeans cluster labeled fraud rate",
    "km_component_mean_dgi": "KMeans cluster mean DGI score",
    "hdb_component_size": "HDBSCAN cluster size",
    "hdb_component_fraud_rate_labeled": "HDBSCAN cluster labeled fraud rate",
    "hdb_component_fraud_lift_labeled": "HDBSCAN cluster fraud lift over baseline",
    "hdb_outlier_score": "HDBSCAN outlier score",
    "dist_to_fraud_centroid": "distance to fraud cluster centroid",
    "dist_to_legit_centroid": "distance to legitimate cluster centroid",
    "centroid_margin": "fraud vs legit centroid margin",
    "eft_amount_match_count": "EFT amount match count (structuring signal)",
    "abm_dc_colocated": "ABM/debit card co-location (card testing signal)",
}


def explain_with_rf(
    customer_features: dict,
    rf_model,
    shap_explainer,
    meta: dict,
    n_drivers: int = 5,
) -> dict:
    """
    Compute ranker risk score + top SHAP drivers for a single customer.
    Supports LightGBM (production) and RandomForest (legacy) models.

    Parameters
    ----------
    customer_features : dict  keyed by feature name
    rf_model : LGBMClassifier or RandomForestClassifier
    shap_explainer : shap.TreeExplainer built from rf_model
    meta : dict from meta.json (must have 'feature_names')

    Returns
    -------
    dict with keys: rf_score, shap_base, drivers (list of dicts), all_shap
    """
    feat_names = meta.get("feature_names", [])
    # Build a rename map (rf_model may use different column names from sage)
    rename_map = meta.get("rename_map", {})
    # Resolve feature values
    feats = []
    for fn in feat_names:
        # try direct, then reverse rename
        val = customer_features.get(fn, None)
        if val is None:
            # check if fn maps from a friendlier name
            for friendly, canonical in rename_map.items():
                if canonical == fn and friendly in customer_features:
                    val = customer_features[friendly]
                    break
        feats.append(float(val) if val is not None else 0.0)

    import pandas as _pd
    X = _pd.DataFrame([feats], columns=feat_names)

    # Support both regressors (predict returns float) and classifiers (use predict_proba)
    if hasattr(rf_model, "predict_proba"):
        rf_score = float(rf_model.predict_proba(X)[0, 1])  # P(fraud)
        shap_raw = shap_explainer.shap_values(X)
        # Modern SHAP (>=0.45) returns ndarray (n_samples, n_features, n_classes)
        # Older SHAP returns list [class_0_arr, class_1_arr]
        if isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
            shap_vals = shap_raw[0, :, 1]  # fraud class
            shap_base = float(np.atleast_1d(shap_explainer.expected_value)[-1])
        elif isinstance(shap_raw, list):
            shap_vals = np.array(shap_raw[1][0])  # class-1 (fraud) SHAP values
            shap_base = float(np.atleast_1d(shap_explainer.expected_value)[-1])
        else:
            shap_vals = np.array(shap_raw[0])
            shap_base = float(np.atleast_1d(shap_explainer.expected_value)[0])
    else:
        rf_score = float(rf_model.predict(X)[0])
        shap_vals = np.array(shap_explainer.shap_values(X)[0])  # (n_features,)
        shap_base = float(np.atleast_1d(shap_explainer.expected_value)[0])

    # Top N positive drivers
    pos_idx = np.where(shap_vals > 0)[0]
    if len(pos_idx) == 0:
        pos_idx = np.argsort(shap_vals)[-n_drivers:][::-1]
    else:
        pos_idx = pos_idx[np.argsort(shap_vals[pos_idx])[::-1]][:n_drivers]

    drivers = []
    for idx in pos_idx:
        fn = feat_names[idx]
        drivers.append(
            {
                "feature": fn,
                "description": FEATURE_DESCRIPTIONS.get(fn, fn),
                "shap_value": round(float(shap_vals[idx]), 5),
                "raw_value": feats[idx],
            }
        )

    return {
        "rf_score": rf_score,
        "shap_base": shap_base,
        "drivers": drivers,
        "all_shap": {feat_names[i]: round(float(shap_vals[i]), 5) for i in range(len(feat_names))},
    }


# ── Transaction aggregate helpers ─────────────────────────────────────────────

def compute_txn_aggregates(transactions: list) -> dict:
    """
    Compute customer-level transaction aggregates from a list of transaction dicts.
    Each dict should have: amount, cash_indicator, ecommerce_ind, merchant_category, city.
    Optional: txn_time (ISO-8601 string) — used for real 24-hour velocity.

    avg_24h_velocity: for each transaction t_i, count how many other transactions
    occurred within the 24 hours *preceding* t_i (inclusive), then average.
    Falls back to total count when no timestamps are present.

    Returns dict with keys matching CUSTOMER_FEATURE_COLS transaction portion.
    """
    if not transactions:
        return {
            "avg_txn_amount": 0.0,
            "max_txn_amount": 0.0,
            "std_txn_amount": 0.0,
            "txn_count": 0.0,
            "cash_rate": 0.0,
            "ecom_rate": 0.0,
            "avg_24h_velocity": 1.0,
            "unique_cities": 0.0,
            "unique_categories": 0.0,
            "min_time_delta": 0.0,
            "time_span_hours": 0.0,
            "geo_velocity": 0.0,
        }
    amounts = [t["amount"] for t in transactions]
    cash = [int(t.get("cash_indicator", 0)) for t in transactions]
    ecom = [int(t.get("ecommerce_ind", 0)) for t in transactions]
    n = len(transactions)

    unique_cities = float(len(set(
        (t.get("city") or "UNKNOWN").upper() for t in transactions
    )))
    unique_cats = float(len(set(
        str(t.get("merchant_category", "")) for t in transactions if t.get("merchant_category")
    )))

    # 24-hour velocity — use real timestamps when available
    timestamps = []
    for t in transactions:
        ts_str = t.get("txn_time", "")
        if ts_str:
            try:
                import datetime as _dt
                ts = _dt.datetime.fromisoformat(ts_str).timestamp()
                timestamps.append(ts)
            except (ValueError, TypeError):
                timestamps.append(None)
        else:
            timestamps.append(None)

    if all(ts is not None for ts in timestamps):
        ts_arr = np.array(timestamps, dtype=np.float64)
        window = 86400.0  # seconds in 24 hours
        velocities = []
        sorted_ts = np.sort(ts_arr)
        for i, t_i in enumerate(ts_arr):
            in_window = np.sum((ts_arr <= t_i) & (ts_arr >= t_i - window))
            velocities.append(float(in_window))
        avg_24h = float(np.mean(velocities))
        # Minimum gap between any two consecutive transactions (minutes)
        if len(sorted_ts) > 1:
            deltas = np.diff(sorted_ts) / 60.0
            min_delta = float(np.min(deltas))
        else:
            min_delta = 0.0
        # Time span = hours from first to last transaction
        time_span_h = float((sorted_ts[-1] - sorted_ts[0]) / 3600.0)
    else:
        avg_24h = float(n)  # fallback: all transactions are "recent"
        min_delta = 0.0
        time_span_h = 0.0

    # Geographic velocity = unique cities per active hour (capped to avoid div-by-zero)
    geo_velocity = unique_cities / max(time_span_h, 1.0 / 60.0) if unique_cities > 1 else 0.0

    return {
        "avg_txn_amount": float(np.mean(amounts)),
        "max_txn_amount": float(np.max(amounts)),
        "std_txn_amount": float(np.std(amounts)) if n > 1 else 0.0,
        "txn_count": float(n),
        "cash_rate": float(np.mean(cash)),
        "ecom_rate": float(np.mean(ecom)),
        "avg_24h_velocity": avg_24h,
        "unique_cities": unique_cities,
        "unique_categories": unique_cats,
        "min_time_delta": min_delta,
        "time_span_hours": time_span_h,
        "geo_velocity": geo_velocity,
    }
