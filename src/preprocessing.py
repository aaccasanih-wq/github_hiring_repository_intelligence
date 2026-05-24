"""
Preprocessing — clean and normalize raw repository signals.

Handles: missing values, type coercion, outlier capping,
and prepares structured fields for the summarization stage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.utils import load_csv, save_csv, get_project_root


def load_raw_data() -> pd.DataFrame:
    """Load all available raw data (Round 1 + Round 2 if present)."""
    dfs = []
    for fname in ["round1_raw.csv", "round2_raw.csv"]:
        try:
            df = load_csv(fname, subdir="raw")
            df["source_round"] = "round1" if "round1" in fname else "round2"
            dfs.append(df)
        except FileNotFoundError:
            continue

    if not dfs:
        raise FileNotFoundError(
            "No raw data found. Run src/github_collector.py first."
        )

    combined = pd.concat(dfs, ignore_index=True)
    # Remove duplicates (same repo_id appearing in both rounds)
    combined = combined.drop_duplicates(subset="repo_id", keep="first")
    return combined


def clean_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and coerce all 10 signals to correct types. Fill missing values."""

    # Numeric columns — fill NaN with 0, coerce to int
    int_cols = [
        "contributors_count", "commits_6m", "releases_count",
        "cicd_workflows", "test_files", "readme_length",
        "dependency_lines", "age_days", "stars", "forks", "size_kb",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Boolean — coerce
    bool_cols = ["has_docs_folder", "has_license", "is_fork"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Text — fill missing
    text_cols = ["readme_preview", "description", "full_name", "language",
                 "owner", "repo_name", "license_name", "topics", "default_branch"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    return df


def cap_outliers(df: pd.DataFrame, column: str, cap: int) -> pd.DataFrame:
    """Cap a numeric column at a maximum value (avoid skew from huge repos)."""
    if column in df.columns:
        df[column] = df[column].clip(upper=cap)
    return df


def preprocess(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df : DataFrame or None
        If None, loads all available raw data.

    Returns
    -------
    Cleaned DataFrame saved to data/processed/cleaned_signals.csv
    """
    if df is None:
        df = load_raw_data()

    df = clean_signals(df)

    # Cap outliers to prevent huge repos from dominating distributions
    caps = {
        "contributors_count": 100,
        "commits_6m": 200,
        "releases_count": 100,
        "test_files": 500,
        "dependency_lines": 1000,
    }
    for col, cap in caps.items():
        df = cap_outliers(df, col, cap)

    save_csv(df, "cleaned_signals.csv", subdir="processed")
    return df


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Starting preprocessing...")
    result = preprocess()
    logger.info("Preprocessing complete — %d repos saved to data/processed/", len(result))
