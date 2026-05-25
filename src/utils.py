"""Helper functions shared across pipeline modules."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Track-A category encoding
LABEL_TO_ID = {
    "intern": 0,
    "junior": 1,
    "senior": 2,
    "lead": 3,
    "template": 4,
    "low-value": 5,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
NUM_CLASSES = len(LABEL_TO_ID)


def get_project_root() -> Path:
    """Return the absolute project root directory."""
    return Path(__file__).resolve().parent.parent


def load_dotenv():
    """Load .env from project root into os.environ (no-op if already loaded)."""
    env_path = get_project_root() / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_csv(filename: str, subdir: str = "raw") -> pd.DataFrame:
    """Load a CSV from data/<subdir>/<filename>."""
    path = get_project_root() / "data" / subdir / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, filename: str, subdir: str = "processed"):
    """Save DataFrame to data/<subdir>/<filename>. Creates dir if needed."""
    out_dir = get_project_root() / "data" / subdir
    ensure_dir(out_dir)
    path = out_dir / filename
    df.to_csv(path, index=False)
    return path


def create_splits(df: pd.DataFrame | None = None,
                  label_col: str = "label",
                  input_csv: str = "combined_labeled.csv",
                  train_size: float = 0.70,
                  val_size: float = 0.15,
                  test_size: float = 0.15,
                  random_state: int = 42) -> tuple:
    """
    Stratified train/val/test split.

    Parameters
    ----------
    df : DataFrame or None. If None, loads from data/labeled/<input_csv>
    label_col : column with string category labels
    input_csv : filename in data/labeled/ to load when df is None.
                Suffix determines output names (e.g. combined_labeled_v2.csv
                produces train_v2.csv / val_v2.csv / test_v2.csv).
    train_size, val_size, test_size : proportions (must sum to 1.0)
    random_state : seed for reproducibility

    Returns
    -------
    (train_df, val_df, test_df) — each with 'label_id' column added.
    Also saves CSVs to data/splits/.
    """
    if df is None:
        df = load_csv(input_csv, subdir="labeled")

    # Derive output suffix from input filename
    stem = Path(input_csv).stem  # e.g. "combined_labeled" or "combined_labeled_v2"
    suffix = stem.replace("combined_labeled", "")  # "" or "_v2"

    # Remove rows with error/unknown labels
    valid_labels = set(LABEL_TO_ID.keys())
    df = df[df[label_col].isin(valid_labels)].copy()

    # Encode labels
    df["label_id"] = df[label_col].map(LABEL_TO_ID)

    # First split: train vs temp (val+test)
    train, temp = train_test_split(
        df, train_size=train_size,
        stratify=df[label_col],
        random_state=random_state,
    )

    # Second split: val vs test from temp
    val_ratio = val_size / (val_size + test_size)
    val, test = train_test_split(
        temp, train_size=val_ratio,
        stratify=temp[label_col],
        random_state=random_state,
    )

    splits_dir = get_project_root() / "data" / "splits"
    ensure_dir(splits_dir)
    train.to_csv(splits_dir / f"train{suffix}.csv", index=False)
    val.to_csv(splits_dir / f"val{suffix}.csv", index=False)
    test.to_csv(splits_dir / f"test{suffix}.csv", index=False)

    return train, val, test


def encode_labels(labels: pd.Series | list) -> list[int]:
    """Convert string labels to integer IDs."""
    return [LABEL_TO_ID.get(str(l), -1) for l in labels]


def decode_labels(ids: list[int]) -> list[str]:
    """Convert integer IDs back to string labels."""
    return [ID_TO_LABEL.get(i, "unknown") for i in ids]
