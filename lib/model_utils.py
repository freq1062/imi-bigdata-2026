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

        h3 = self.conv3(h2, edge_index_dict)
        h3["customer"] = F.elu(self.bn2(h3["customer"]) + h1["customer"])
        return self.classifier(h3["customer"])


# ── Artifact loading with Streamlit caching ───────────────────────────────────

def load_sage_model(model_path: str = "outputs/fraud_sage_model.pth", artifacts: dict = None):
    """Load FraudSAGE weights into model. Returns model on eval mode."""
    if not TORCH_GEO_AVAILABLE:
        return None
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


def load_artifacts(artifacts_path: str = "outputs/sage_artifacts.pkl") -> dict:
    """Load inference artifacts (scaler, maps, embeddings, edge tensors)."""
    with open(artifacts_path, "rb") as f:
        return pickle.load(f)


def load_rf_model(rf_dir: str = "outputs/rf_model_sage"):
    """Load the RF proxy model and SHAP explainer."""
    rf = joblib.load(os.path.join(rf_dir, "rf_proxy.joblib"))
    explainer = joblib.load(os.path.join(rf_dir, "shap_explainer.joblib"))
    with open(os.path.join(rf_dir, "meta.json")) as f:
        meta = json.load(f)
    return rf, explainer, meta


# ── Inductive inference ───────────────────────────────────────────────────────

def score_new_customers(
    new_customer_records: list,
    model,
    artifacts: dict,
    device=None,
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

    Returns
    -------
    pd.DataFrame  columns: customer_id, risk_score, risk_tier, predicted_label
    """
    if not TORCH_GEO_AVAILABLE or model is None:
        # Graceful degradation: return 0 risk for all
        results = []
        for i, rec in enumerate(new_customer_records):
            results.append(
                {
                    "customer_id": rec.get("customer_id", f"NEW_{i}"),
                    "risk_score": 0.0,
                    "risk_tier": "LOW",
                    "predicted_label": 0,
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

    # Step 6: forward pass with temperature scaling
    # Temperature scaling addresses overconfident predictions:
    #   - temperature_demo (default 2.0) provides smoother predictions for UI
    #   - Reduces extreme predictions (77% → 3%) for better UX
    #   - Prevents rapid score jumps when adding transactions
    #   - Higher T = more gradual, less sensitive to small changes
    temperature = artifacts.get('temperature_demo', 2.0)
    
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader)).to(device)
        logits = model(batch.x_dict, batch.edge_index_dict)
        # Apply temperature scaling: divide logits by temperature before sigmoid
        # probs = sigmoid(logits / T) where T > 1 → smoother, T < 1 → sharper
        logits_scaled = logits.squeeze()[: len(new_node_ids)] / temperature
        probs = torch.sigmoid(logits_scaled).cpu().numpy()

    probs = np.atleast_1d(probs)

    # Step 7: results
    results = []
    for i, rec in enumerate(new_customer_records):
        score = float(probs[i])
        tier = "HIGH" if score >= 0.70 else ("MEDIUM" if score >= 0.40 else "LOW")
        results.append(
            {
                "customer_id": rec.get("customer_id", f"NEW_{i}"),
                "risk_score": round(score, 6),
                "risk_tier": tier,
                "predicted_label": int(score >= 0.5),
            }
        )
    return pd.DataFrame(results)


# ── RF Proxy + SHAP inference ──────────────────────────────────────────────────

FEATURE_DESCRIPTIONS = {
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
}


def explain_with_rf(
    customer_features: dict,
    rf_model,
    shap_explainer,
    meta: dict,
    n_drivers: int = 5,
) -> dict:
    """
    Compute RF risk score + top SHAP drivers for a single customer.

    Parameters
    ----------
    customer_features : dict  keyed by feature name
    rf_model, shap_explainer : loaded RF objects
    meta : dict from meta.json

    Returns
    -------
    dict with keys: rf_score, shap_base, drivers (list of dicts)
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

    X = np.array([feats], dtype=np.float32)

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
