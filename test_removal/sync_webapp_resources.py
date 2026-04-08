"""Sync runtime artifacts into webapp_resources for Streamlit deployment."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
WEBAPP_DIR = PROJECT_ROOT / "webapp_resources"
WEBAPP_OUTPUTS = WEBAPP_DIR / "outputs"
WEBAPP_DATA = WEBAPP_DIR / "data"

RUNTIME_OUTPUT_FILES = [
    "sage_artifacts.pkl",
    "fraud_sage_model.pth",
    "transaction_autoencoder.pt",
    "model_output.csv",
    "model_output_explanations.csv.gz",
    "meta_cluster_assignments.csv.gz",
    "meta_cluster_semantic_labels.json",
    "meta_cluster_significant_deltas.json",
    "meta_cluster_top3_categories.json",
]

RUNTIME_RF_FILES = [
    "rf_proxy.joblib",
    "shap_explainer.joblib",
    "meta.json",
]

RUNTIME_DATA_FILES = [
    "kyc_industry_codes.csv.gz",
    "card.csv.gz",
    "abm.csv.gz",
    "eft.csv.gz",
    "emt.csv.gz",
    "wire.csv.gz",
    "cheque.csv.gz",
]


def _copy_or_move(src: Path, dst: Path, move: bool) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        if dst.exists():
            dst.unlink()
        src.replace(dst)
    else:
        shutil.copy2(src, dst)
    return True


def sync_resources(move: bool = False) -> dict[str, int]:
    WEBAPP_OUTPUTS.mkdir(parents=True, exist_ok=True)
    (WEBAPP_OUTPUTS / "rf_model_sage").mkdir(parents=True, exist_ok=True)
    WEBAPP_DATA.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0

    for name in RUNTIME_OUTPUT_FILES:
        src = OUTPUTS_DIR / name
        dst = WEBAPP_OUTPUTS / name
        if _copy_or_move(src, dst, move=move):
            copied += 1
        else:
            missing += 1

    for name in RUNTIME_RF_FILES:
        src_nested = OUTPUTS_DIR / "rf_model_sage" / name
        src_flat = OUTPUTS_DIR / name
        src = src_nested if src_nested.exists() else src_flat
        dst = WEBAPP_OUTPUTS / "rf_model_sage" / name
        if _copy_or_move(src, dst, move=move):
            copied += 1
        else:
            missing += 1

    for name in RUNTIME_DATA_FILES:
        src = DATA_DIR / name
        dst = WEBAPP_DATA / name
        if _copy_or_move(src, dst, move=False):
            copied += 1
        else:
            missing += 1

    return {"synced": copied, "missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Streamlit runtime resources.")
    parser.add_argument(
        "--move-outputs",
        action="store_true",
        help="Move output artifacts instead of copying them.",
    )
    args = parser.parse_args()

    result = sync_resources(move=args.move_outputs)
    print(f"Synced files: {result['synced']}")
    print(f"Missing files: {result['missing']}")


if __name__ == "__main__":
    main()
