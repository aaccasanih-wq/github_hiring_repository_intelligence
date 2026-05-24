"""Helper functions shared across pipeline modules."""

import os
from pathlib import Path

import pandas as pd


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
