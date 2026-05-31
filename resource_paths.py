"""Centralized paths for Streamlit runtime artifacts."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = Path(
    os.environ.get("RESOURCES_DIR", PROJECT_ROOT / "resources")
)

OUTPUTS_DIR = RESOURCES_DIR / "outputs"
DATA_DIR = RESOURCES_DIR / "data"

LEGACY_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LEGACY_DATA_DIR = PROJECT_ROOT / "data"


def resolve_output_path(*parts: str) -> str:
    """Return preferred output path, falling back to legacy outputs/ if needed."""
    preferred = OUTPUTS_DIR.joinpath(*parts)
    if preferred.exists():
        return str(preferred)
    return str(LEGACY_OUTPUTS_DIR.joinpath(*parts))


def resolve_data_path(*parts: str) -> str:
    """Return preferred data path, falling back to legacy data/ if needed."""
    preferred = DATA_DIR.joinpath(*parts)
    if preferred.exists():
        return str(preferred)
    return str(LEGACY_DATA_DIR.joinpath(*parts))
