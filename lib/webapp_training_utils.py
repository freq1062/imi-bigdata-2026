"""Utilities to prepare webapp_resources for first-time training runs."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

from lib.resource_paths import PROJECT_ROOT, WEBAPP_DATA_DIR, WEBAPP_OUTPUTS_DIR


DATA_FILE_ID = "1A7w3GqZTCVsv-A8gXj5NKV2FNc0zoduG"
REQUIRED_DATA_FILES = [
    "labels.csv.gz",
    "kyc_individual.csv.gz",
    "kyc_smallbusiness.csv.gz",
    "kyc_industry_codes.csv.gz",
    "kyc_occupation_codes.csv.gz",
    "card.csv.gz",
    "abm.csv.gz",
    "eft.csv.gz",
    "emt.csv.gz",
    "wire.csv.gz",
    "cheque.csv.gz",
    "westernunion.csv.gz",
]


def ensure_webapp_dirs() -> None:
    """Create web app data/output folders if they do not exist."""
    WEBAPP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    WEBAPP_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (WEBAPP_OUTPUTS_DIR / "rf_model_sage").mkdir(parents=True, exist_ok=True)


def ensure_scotia_data(*, force_download: bool = False, quiet: bool = False) -> list[str]:
    """Ensure required Scotia source files exist in webapp_resources/data.

    Returns a list of files that are still missing after setup.
    """
    ensure_webapp_dirs()

    missing = [f for f in REQUIRED_DATA_FILES if not (WEBAPP_DATA_DIR / f).exists()]
    if not missing and not force_download:
        if not quiet:
            print(f"All data files already present in {WEBAPP_DATA_DIR}")
        return []

    zip_path = WEBAPP_DATA_DIR / "data.zip"
    if not quiet:
        print("Downloading Scotia data bundle from Google Drive...")

    try:
        import gdown

        gdown.download(id=DATA_FILE_ID, output=str(zip_path), quiet=quiet)
    except Exception as exc:  # pragma: no cover - notebook runtime dependency path
        raise RuntimeError(
            "Could not download Scotia data bundle automatically. "
            "Please place the .csv.gz files in webapp_resources/data/."
        ) from exc

    if not quiet:
        print("Extracting data files...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            name = Path(member).name
            if not name or name.startswith("."):
                continue
            target = WEBAPP_DATA_DIR / name
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())

    zip_path.unlink(missing_ok=True)
    still_missing = [f for f in REQUIRED_DATA_FILES if not (WEBAPP_DATA_DIR / f).exists()]
    if still_missing and not quiet:
        print("Missing after extraction:")
        for item in still_missing:
            print(f"  - {item}")

    return still_missing


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def build_model_output_from_rank_df(rank_df: pd.DataFrame) -> pd.DataFrame:
    """Create model_output.csv schema used by the Streamlit Model Output page."""
    out = rank_df.copy()

    for col in ["customer_id", "split"]:
        if col not in out.columns:
            raise KeyError(f"rank_df is missing required column: {col}")

    if "predicted_label" not in out.columns:
        if "lgb_label" in out.columns:
            out["predicted_label"] = out["lgb_label"]
        elif "lgb_fraud_prob" in out.columns:
            out["predicted_label"] = (out["lgb_fraud_prob"] >= 0.5).astype(int)
        else:
            out["predicted_label"] = 0

    if "fraud_score" not in out.columns:
        for candidate in [
            "scarcity_anchor_ensemble_prob",
            "scarcity_semi_ensemble_prob",
            "scarcity_ensemble_prob",
            "lgb_fraud_prob",
            "mlp_fraud_prob",
        ]:
            if candidate in out.columns:
                out["fraud_score"] = out[candidate].astype(float)
                break
        else:
            out["fraud_score"] = 0.0

    if "true_label" not in out.columns:
        out["true_label"] = -1

    keep = [
        "customer_id",
        "split",
        "true_label",
        "predicted_label",
        "fraud_score",
        "lgb_fraud_prob",
        "mlp_fraud_prob",
        "scarcity_anchor_ensemble_prob",
        "scarcity_semi_ensemble_prob",
        "scarcity_ensemble_prob",
        "cluster_consensus_score",
        "dgi_anomaly_score",
        "customer_ae_risk_norm",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].copy()


def package_webapp_model_artifacts(
    *,
    rank_df: pd.DataFrame,
    meta_features: list[str] | None = None,
    outputs_dir: Path | None = None,
    include_shap: bool = False,
) -> dict[str, int]:
    """Copy trained model artifacts into webapp_resources/outputs.

    This intentionally excludes LLM explainability artifact generation.
    """
    ensure_webapp_dirs()

    outputs_dir = outputs_dir or (PROJECT_ROOT / "outputs")
    webapp_outputs = WEBAPP_OUTPUTS_DIR
    rf_dir = webapp_outputs / "rf_model_sage"
    rf_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0

    model_output_df = build_model_output_from_rank_df(rank_df)
    model_output_df.to_csv(webapp_outputs / "model_output.csv", index=False)
    copied += 1

    direct_files = [
        "fraud_sage_model.pth",
        "sage_artifacts.pkl",
        "transaction_autoencoder.pt",
        "lgbm_fraud_ranker.joblib",
        "dgi_model.pt",
        "dgi_gmm.joblib",
        "dgi_customer_embeddings.npy",
        "dgi_embedding_mlp.pt",
        "anchor_rf.joblib",
        "rank_df_with_anchor_expansion.csv.gz",
        "dgi_component_assignments.csv.gz",
    ]

    for name in direct_files:
        src = outputs_dir / name
        dst = webapp_outputs / name
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1

    rf_candidates = [outputs_dir / "rf_model_sage" / "rf_proxy.joblib", outputs_dir / "rf_proxy.joblib"]
    rf_src = _first_existing(rf_candidates)
    if rf_src is not None:
        shutil.copy2(rf_src, rf_dir / "rf_proxy.joblib")
        copied += 1
    else:
        missing += 1

    if include_shap:
        shap_candidates = [outputs_dir / "rf_model_sage" / "shap_explainer.joblib", outputs_dir / "shap_explainer.joblib"]
        shap_src = _first_existing(shap_candidates)
        if shap_src is not None:
            shutil.copy2(shap_src, rf_dir / "shap_explainer.joblib")
            copied += 1
        else:
            missing += 1

    meta_payload = {
        "feature_names": list(meta_features or []),
        "model_type": "RandomForestClassifier",
        "n_features": len(meta_features or []),
    }
    (rf_dir / "meta.json").write_text(json.dumps(meta_payload, indent=2))
    copied += 1

    assign_src = outputs_dir / "dgi_component_assignments.csv.gz"
    assign_dst = webapp_outputs / "meta_cluster_assignments.csv.gz"
    if assign_src.exists():
        shutil.copy2(assign_src, assign_dst)
        copied += 1
    elif "component" in rank_df.columns:
        rank_df[["customer_id", "component"]].rename(columns={"component": "meta_cluster"}).to_csv(
            assign_dst,
            index=False,
            compression="gzip",
        )
        copied += 1
    else:
        missing += 1

    return {"copied": copied, "missing": missing}
