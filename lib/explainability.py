"""Utility functions for temporal, spatial, and global explainability views."""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import torch

from lib.autoencoder import load_autoencoder
from lib.resource_paths import resolve_output_path


FEATURE_LABELS = {
    "amount_behavior_z": "Amount vs baseline",
    "amount_30d_ratio": "Amount / 30d ratio",
    "category_amount_z": "Amount vs category",
    "new_category_for_customer": "New merchant category",
    "unique_categories_last10": "Category diversity (last 10)",
    "is_cold_start": "Cold start behavior",
    "time_delta": "Time since last txn",
    "velocity_1h": "1h velocity",
    "velocity_24h": "24h velocity",
    "velocity_7d": "7d velocity",
    "velocity_ratio_24h_7davg": "24h vs 7d acceleration",
    "distance_from_last_txn": "Location jump",
    "geo_velocity_kmph": "Geo velocity",
    "ecommerce_ind": "E-commerce indicator",
    "cash_indicator": "Cash indicator",
    "std_amount": "Amount variability",
    "mean_amount": "Average amount",
    "mean_velocity_1h": "1h velocity",
    "mean_velocity_24h": "24h velocity",
    "mean_velocity_7d": "7d velocity",
    "mean_geo_velocity_kmph": "Geo velocity",
    "txn_count": "Transaction count",
    "category_entropy": "Category entropy",
    "mean_amount_behavior_z": "Amount behavior z",
    "mean_category_amount_z": "Category amount z",
}


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


@lru_cache(maxsize=4)
def load_cluster_assets(outputs_dir: str = "outputs") -> dict[str, Any]:
    """Load cluster explainability metadata generated in the notebook pipeline."""
    # Always prefer the resolved webapp outputs dir when the given path lacks the files
    resolved = outputs_dir
    if not os.path.isfile(os.path.join(resolved, "meta_cluster_significant_deltas.json")):
        resolved = resolve_output_path()
    return {
        "significant_deltas": _read_json(
            os.path.join(resolved, "meta_cluster_significant_deltas.json"), {}
        ),
        "semantic_labels": _read_json(
            os.path.join(resolved, "meta_cluster_semantic_labels.json"), {}
        ),
        "top_categories": _read_json(
            os.path.join(resolved, "meta_cluster_top3_categories.json"), {}),
    }


@lru_cache(maxsize=1)
def load_temporal_autoencoder(checkpoint_path: str | None = None):
    """Load transaction autoencoder, scaler, and metadata if available."""
    if checkpoint_path is None:
        checkpoint_path = resolve_output_path("transaction_autoencoder.pt")
    if not os.path.exists(checkpoint_path):
        return None, None, {}
    try:
        model, scaler, meta = load_autoencoder(checkpoint_path, map_location="cpu")
        model.eval()
        return model, scaler, meta
    except Exception:
        return None, None, {}


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if abs(b) > 1e-12 else default


def _category_entropy(categories: list[str]) -> float:
    if not categories:
        return 0.0
    vc = pd.Series(categories).value_counts(normalize=True)
    ent = float(-(vc * np.log(vc + 1e-12)).sum())
    max_ent = math.log(max(len(vc), 1))
    return _safe_div(ent, max_ent, default=0.0)


def build_transaction_ae_features(transactions: list[dict]) -> pd.DataFrame:
    """Create per-transaction features aligned to the AE checkpoint feature names."""
    if not transactions:
        return pd.DataFrame()

    df = pd.DataFrame(transactions).copy()
    df["amount"] = pd.to_numeric(df.get("amount", 0.0), errors="coerce").fillna(0.0)
    df["txn_time"] = pd.to_datetime(df.get("txn_time", None), errors="coerce")
    if df["txn_time"].notna().all():
        df = df.sort_values("txn_time").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    amounts = df["amount"].to_numpy(dtype=float)
    mean_amt = float(np.mean(amounts)) if len(amounts) else 0.0
    std_amt = float(np.std(amounts)) if len(amounts) > 1 else 1.0

    amount_behavior_z = (amounts - mean_amt) / (std_amt + 1e-8)

    rolling_mean = pd.Series(amounts).expanding(min_periods=1).mean().shift(1).fillna(mean_amt)
    amount_30d_ratio = amounts / (rolling_mean.to_numpy(dtype=float) + 1e-6)

    categories = df.get("merchant_category", pd.Series([""] * len(df))).astype(str)
    category_amount_z = np.zeros(len(df), dtype=float)
    for cat in categories.unique():
        idx = categories == cat
        vals = amounts[idx.to_numpy()]
        c_mean = float(vals.mean()) if len(vals) else 0.0
        c_std = float(vals.std()) if len(vals) > 1 else 1.0
        category_amount_z[idx.to_numpy()] = (vals - c_mean) / (c_std + 1e-8)

    seen = set()
    new_category_for_customer = []
    for cat in categories:
        is_new = 0.0 if cat in seen else 1.0
        new_category_for_customer.append(is_new)
        seen.add(cat)

    unique_categories_last10 = []
    for i in range(len(df)):
        win = categories.iloc[max(0, i - 9) : i + 1]
        unique_categories_last10.append(float(win.nunique()))

    is_cold_start = np.array([1.0 if i < 3 else 0.0 for i in range(len(df))], dtype=float)

    time_delta = np.zeros(len(df), dtype=float)
    velocity_1h = np.zeros(len(df), dtype=float)
    velocity_24h = np.zeros(len(df), dtype=float)
    velocity_7d = np.zeros(len(df), dtype=float)

    city = df.get("city", pd.Series(["UNKNOWN"] * len(df))).astype(str).str.upper()
    distance_from_last_txn = np.zeros(len(df), dtype=float)
    geo_velocity_kmph = np.zeros(len(df), dtype=float)

    if df["txn_time"].notna().all():
        ts = df["txn_time"].astype("int64").to_numpy(dtype=float) / 1e9
        for i in range(len(df)):
            if i > 0:
                dt_h = max((ts[i] - ts[i - 1]) / 3600.0, 1e-6)
                time_delta[i] = dt_h * 60.0
                jump = 1.0 if city.iloc[i] != city.iloc[i - 1] else 0.0
                distance_from_last_txn[i] = jump
                geo_velocity_kmph[i] = jump * (100.0 / dt_h)

            t_i = ts[i]
            velocity_1h[i] = float(np.sum((ts <= t_i) & (ts >= t_i - 3600.0)))
            velocity_24h[i] = float(np.sum((ts <= t_i) & (ts >= t_i - 86400.0)))
            velocity_7d[i] = float(np.sum((ts <= t_i) & (ts >= t_i - 86400.0 * 7.0)))
    else:
        velocity_1h[:] = np.arange(1, len(df) + 1)
        velocity_24h[:] = np.arange(1, len(df) + 1)
        velocity_7d[:] = np.arange(1, len(df) + 1)

    velocity_ratio_24h_7davg = velocity_24h / (velocity_7d / 7.0 + 1e-6)

    out = pd.DataFrame(
        {
            "amount_behavior_z": amount_behavior_z,
            "amount_30d_ratio": amount_30d_ratio,
            "category_amount_z": category_amount_z,
            "new_category_for_customer": np.array(new_category_for_customer, dtype=float),
            "unique_categories_last10": np.array(unique_categories_last10, dtype=float),
            "is_cold_start": is_cold_start,
            "time_delta": time_delta,
            "velocity_1h": velocity_1h,
            "velocity_24h": velocity_24h,
            "velocity_7d": velocity_7d,
            "velocity_ratio_24h_7davg": velocity_ratio_24h_7davg,
            "distance_from_last_txn": distance_from_last_txn,
            "geo_velocity_kmph": geo_velocity_kmph,
            "ecommerce_ind": pd.to_numeric(df.get("ecommerce_ind", 0), errors="coerce").fillna(0.0),
            "cash_indicator": pd.to_numeric(df.get("cash_indicator", 0), errors="coerce").fillna(0.0),
        }
    )
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def temporal_reconstruction_explainer(
    transactions: list[dict],
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Return per-feature AE reconstruction errors for a customer transaction stream."""
    feats = build_transaction_ae_features(transactions)
    if feats.empty:
        return {"available": False, "reason": "No transactions available."}

    model, scaler, meta = load_temporal_autoencoder(checkpoint_path)
    if model is None or scaler is None:
        return {
            "available": False,
            "reason": "Autoencoder checkpoint unavailable.",
        }

    feature_names = meta.get("feature_names") or list(feats.columns)
    for col in feature_names:
        if col not in feats.columns:
            feats[col] = 0.0

    X = feats[feature_names].to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=3.0, neginf=-3.0)

    with torch.no_grad():
        xb = torch.tensor(X_scaled, dtype=torch.float32)
        recon = model(xb).cpu().numpy()

    sq_err = (X_scaled - recon) ** 2
    feature_error = pd.Series(sq_err.mean(axis=0), index=feature_names).sort_values(ascending=False)

    threshold = float(meta.get("threshold", 0.0) or 0.0)
    txn_mse = sq_err.mean(axis=1)
    anomaly_rate = float(np.mean(txn_mse > threshold)) if threshold > 0 else 0.0

    top = [
        {
            "feature": str(k),
            "feature_label": FEATURE_LABELS.get(str(k), str(k)),
            "error": float(v),
        }
        for k, v in feature_error.head(6).items()
    ]

    return {
        "available": True,
        "feature_error": {k: float(v) for k, v in feature_error.items()},
        "top_features": top,
        "anomaly_rate": anomaly_rate,
        "threshold": threshold,
    }


def _proxy_behavior_features(record: dict[str, float], transactions: list[dict]) -> dict[str, float]:
    """Map live-session aggregates to cluster-profile feature space."""
    cats = [str(t.get("merchant_category", "")) for t in transactions if t.get("merchant_category")]

    txn_count = float(record.get("txn_count", 0.0))
    avg_24h = float(record.get("avg_24h_velocity", 0.0))
    std_amt = float(record.get("std_txn_amount", 0.0))
    avg_amt = float(record.get("avg_txn_amount", 0.0))
    geo_v = float(record.get("geo_velocity", 0.0))

    one_h_vel = txn_count / max(float(record.get("time_span_hours", 0.0)), 1.0)
    sev_scale = max(avg_amt, 1.0)

    return {
        "txn_count": min(1.0, np.log1p(txn_count) / 4.0),
        "mean_velocity_24h": min(1.0, np.log1p(avg_24h) / 4.0),
        "mean_velocity_7d": min(1.0, np.log1p(avg_24h * 7.0) / 5.0),
        "mean_velocity_1h": min(1.0, np.log1p(one_h_vel) / 4.0),
        "std_amount": min(1.0, _safe_div(std_amt, sev_scale, 0.0) * 3.0),
        "mean_amount": min(1.0, avg_amt / 5000.0),
        "mean_amount_behavior_z": min(1.0, _safe_div(std_amt, sev_scale, 0.0) * 2.0),
        "mean_category_amount_z": min(1.0, _safe_div(std_amt, sev_scale, 0.0) * 1.8),
        "mean_geo_velocity_kmph": min(1.0, geo_v / 4.0),
        "category_entropy": _category_entropy(cats),
    }


def _phenotype_name(top_features: list[str]) -> str:
    fset = set(top_features)
    if {"mean_velocity_24h", "txn_count"} & fset:
        return "High-Speed Smurfer"
    if {"mean_amount", "mean_amount_behavior_z", "mean_category_amount_z"} & fset:
        return "High-Value Structuring"
    if {"mean_geo_velocity_kmph"} & fset:
        return "Geographic Outlier"
    if {"category_entropy"} & fset:
        return "Mule Account / Social Contagion"
    if {"std_amount"} & fset:
        return "Anomalous Nocturnal Activity"
    return "Network Mule"


def build_cluster_phenotype_table(top_k: int = 9, outputs_dir: str = "outputs") -> pd.DataFrame:
    """Create the 9-pattern phenotype table from cluster significant deltas."""
    assets = load_cluster_assets(outputs_dir)
    deltas: dict[str, list[dict]] = assets.get("significant_deltas", {})
    labels: dict[str, dict[str, str]] = assets.get("semantic_labels", {})

    rows = []
    for cid, feats in deltas.items():
        if not feats:
            continue
        top3 = sorted(feats, key=lambda x: abs(float(x.get("delta_z", 0.0))), reverse=True)[:3]
        top_features = [str(x.get("feature", "")) for x in top3]
        rows.append(
            {
                "cluster_id": int(cid),
                "top_feature_drivers": ", ".join(FEATURE_LABELS.get(f, f) for f in top_features),
                "likely_label": _phenotype_name(top_features),
                "max_abs_z": max(abs(float(x.get("delta_z", 0.0))) for x in top3),
                "profile_name": labels.get(cid, {}).get("name", "Unknown Pattern"),
                "risk_level": labels.get(cid, {}).get("risk", "Unknown"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "top_feature_drivers",
                "likely_label",
                "profile_name",
                "risk_level",
            ]
        )

    out = pd.DataFrame(rows).sort_values("max_abs_z", ascending=False).head(top_k)
    out = out.drop(columns=["max_abs_z"]).reset_index(drop=True)
    return out


def spatial_cluster_explainer(
    record: dict[str, float],
    transactions: list[dict],
    outputs_dir: str = "outputs",
) -> dict[str, Any]:
    """Assign nearest cluster profile and provide a cluster-feature heatmap matrix."""
    assets = load_cluster_assets(outputs_dir)
    deltas: dict[str, list[dict]] = assets.get("significant_deltas", {})
    labels_raw: dict = assets.get("semantic_labels", {})
    top_cats: dict[str, list[str]] = assets.get("top_categories", {})

    # Normalise label entries — support both {"name":…, "risk":…} dicts and
    # plain strings (older format where just the cluster name is stored).
    def _label(ckey: str) -> dict[str, str]:
        v = labels_raw.get(str(ckey), {})
        if isinstance(v, str):
            return {"name": v, "risk": "Unknown", "logic": ""}
        if isinstance(v, dict):
            return v
        return {"name": "Unknown", "risk": "Unknown", "logic": ""}

    labels = {str(k): _label(k) for k in labels_raw}

    if not deltas:
        return {"available": False, "reason": "Cluster delta profiles unavailable."}

    proxy = _proxy_behavior_features(record, transactions)

    scores: list[tuple[int, float]] = []
    for cid, feat_list in deltas.items():
        score = 0.0
        for item in feat_list:
            feat = str(item.get("feature", ""))
            z = float(item.get("delta_z", 0.0))
            x = float(proxy.get(feat, 0.5))
            if z >= 0:
                score += abs(z) * x
            else:
                score += abs(z) * (1.0 - x)
        scores.append((int(cid), score))

    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    top_cluster = ranked[0][0]

    top9 = [cid for cid, _ in ranked[:9]]
    top_features = sorted(
        {
            str(item.get("feature", ""))
            for cid in map(str, top9)
            for item in deltas.get(cid, [])
        }
    )
    top_features = top_features[:8]

    heat_rows = []
    for cid in top9:
        ckey = str(cid)
        zmap = {str(x.get("feature", "")): float(x.get("delta_z", 0.0)) for x in deltas.get(ckey, [])}
        row: dict[str, Any] = {f: zmap.get(f, 0.0) for f in top_features}
        row["cluster_id"] = cid
        row["profile_name"] = labels.get(ckey, {}).get("name", "Unknown")
        heat_rows.append(row)

    heat_df = pd.DataFrame(heat_rows)

    sig_top = sorted(
        deltas.get(str(top_cluster), []),
        key=lambda x: abs(float(x.get("delta_z", 0.0))),
        reverse=True,
    )[:3]
    sig_features = [str(x.get("feature", "")) for x in sig_top]

    return {
        "available": True,
        "cluster_id": top_cluster,
        "cluster_label": labels.get(str(top_cluster), {}).get("name", "Unknown Pattern"),
        "cluster_risk": labels.get(str(top_cluster), {}).get("risk", "Unknown"),
        "logic": labels.get(str(top_cluster), {}).get("logic", ""),
        "top_feature_drivers": [
            {
                "feature": f,
                "label": FEATURE_LABELS.get(f, f),
                "z_score": float(x.get("delta_z", 0.0)),
            }
            for f, x in zip(sig_features, sig_top)
        ],
        "top_categories": top_cats.get(str(top_cluster), []),
        "heatmap_df": heat_df,
    }


def _normalize_driver(driver: dict[str, Any]) -> dict[str, Any]:
    shap_val = float(driver.get("shap_value", 0.0))
    return {
        "description": str(driver.get("description", driver.get("feature", "unknown"))),
        "shap_value": shap_val,
        "abs_value": abs(shap_val),
    }


def global_verdict_explainer(
    risk_score: float,
    drivers: list[dict[str, Any]],
    temporal: dict[str, Any] | None = None,
    spatial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose additive push/pull narrative from SHAP-style local contributions."""
    norm = [_normalize_driver(d) for d in drivers]
    if not norm:
        return {
            "summary": f"Final risk {risk_score:.1%}. No SHAP drivers available.",
            "push": [],
            "pull": [],
        }

    push = [d for d in norm if d["shap_value"] > 0]
    pull = [d for d in norm if d["shap_value"] < 0]

    total_push = sum(d["abs_value"] for d in push) or 1e-8
    for d in push:
        d["share_pct"] = 100.0 * d["abs_value"] / total_push
    pull_sorted = sorted(pull, key=lambda x: x["abs_value"], reverse=True)
    push_sorted = sorted(push, key=lambda x: x["abs_value"], reverse=True)

    clauses = [f"Final Risk {risk_score:.0%}."]

    if spatial and spatial.get("available"):
        clauses.append(
            f"+{push_sorted[0]['share_pct']:.0f}% linked to {spatial.get('cluster_label')} (Cluster {spatial.get('cluster_id')})."
            if push_sorted
            else f"Cluster context: {spatial.get('cluster_label')} (Cluster {spatial.get('cluster_id')})."
        )

    if temporal and temporal.get("available") and temporal.get("top_features"):
        tf = temporal["top_features"][0]
        clauses.append(
            f"Temporal AE anomaly concentrated on {tf.get('feature_label', tf.get('feature'))}."
        )

    if push:
        clauses.append(
            "Risk push: "
            + ", ".join(f"{p['description']} (+{p['share_pct']:.0f}%)" for p in push_sorted[:3])
            + "."
        )
    if pull:
        total_pull = sum(x["abs_value"] for x in pull_sorted) or 1e-8
        clauses.append(
            "Risk pull: "
            + ", ".join(
                f"{p['description']} (-{100.0 * p['abs_value'] / total_pull:.0f}%)"
                for p in pull_sorted[:2]
            )
            + "."
        )

    return {
        "summary": " ".join(clauses),
        "push": push_sorted,
        "pull": pull_sorted,
    }


def load_model_output_explanations(outputs_dir: str = "outputs") -> pd.DataFrame:
    """Load model output explanations from CSV or gzip CSV."""
    resolved = outputs_dir if os.path.exists(outputs_dir) else resolve_output_path()
    candidates = [
        os.path.join(resolved, "model_output_explanations.csv"),
        os.path.join(resolved, "model_output_explanations.csv.gz"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    return pd.DataFrame(columns=["customer_id", "explanation"])
