"""
Summarization — convert structured signals into a text representation
usable by both the LLM (for labeling) and BERT (for fine-tuning).

The output format is deliberately consistent across all repos.
Fields excluded per methodology: stars, forks, owner name, topics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.utils import load_csv, save_csv


def build_text_repr(row: pd.Series) -> str:
    """
    Build a structured text block for one repository.

    Format matches the specification in sampling_strategy_track_a.md:
        Repository: {full_name} | Language: {language}
        README: {readme_preview_500_chars}
        CI/CD workflows: {cicd_workflow_count} | Test files: {test_file_count}
        Contributors: {contributors_count} | Releases: {releases_count} | Commits (last 6m): {commits_last_6m}
        Dependencies: {dependency_count} | Docs folder: {has_docs_folder} | License: {license_name}
        Repository age: {age_days} days

    Deliberately excluded: stars, forks, owner, topics.
    """
    template = (
        "Repository: {full_name} | Language: {language}\n"
        "README: {readme_preview}\n"
        "CI/CD workflows: {cicd_workflows} | Test files: {test_files}\n"
        "Contributors: {contributors} | Releases: {releases} | Commits (last 6m): {commits}\n"
        "Dependencies: {dependencies} | Docs folder: {docs} | License: {license}\n"
        "Repository age: {age} days"
    )
    return template.format(
        full_name=row.get("full_name", ""),
        language=row.get("language", "Unknown"),
        readme_preview=_safe_str(row.get("readme_preview", ""))[:500],
        cicd_workflows=row.get("cicd_workflows", 0),
        test_files=row.get("test_files", 0),
        contributors=row.get("contributors_count", 0),
        releases=row.get("releases_count", 0),
        commits=row.get("commits_6m", 0),
        dependencies=row.get("dependency_lines", 0),
        docs="Yes" if row.get("has_docs_folder", False) else "No",
        license=row.get("license_name", "None"),
        age=row.get("age_days", 0),
    )


def _safe_str(val) -> str:
    """Cast a value to string, handling NaN."""
    if pd.isna(val):
        return ""
    return str(val)


def generate_summaries(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Generate text representations for all repos.

    Parameters
    ----------
    df : DataFrame or None
        If None, loads cleaned_signals.csv from data/processed/.

    Returns
    -------
    DataFrame with original columns + 'text_repr' column.
    Saved to data/processed/repo_summaries.csv
    """
    if df is None:
        df = load_csv("cleaned_signals.csv", subdir="processed")

    df["text_repr"] = df.apply(build_text_repr, axis=1)
    df["text_length"] = df["text_repr"].str.len()

    save_csv(df, "repo_summaries.csv", subdir="processed")
    return df


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Generating repository text representations...")
    result = generate_summaries()
    logger.info("Summaries complete — %d repos saved to data/processed/repo_summaries.csv", len(result))
