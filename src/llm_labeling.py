"""
LLM Weak Labeling — use DeepSeek to annotate repositories
with one of 6 engineering-maturity categories.

Runs on the structured text representations from Stage 2.
Saves labels to data/labeled/ for downstream split + training.
"""

from __future__ import annotations

import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import pandas as pd

from src.utils import load_csv, save_csv, load_dotenv, get_project_root

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
BATCH_DELAY = 1.0  # seconds between API calls

CATEGORIES = [
    "intern",
    "junior",
    "senior",
    "lead",
    "template",
    "low-value",
]

# ---------------------------------------------------------------------------
# Prompt version 1 (BASELINE) — qualitative category definitions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V1 = """You are a senior engineering manager evaluating GitHub repositories for hiring purposes.
Your task: classify a repository into EXACTLY ONE of these 6 categories based on the provided signals.

Category definitions:

1. intern — Personal/student project. Single contributor, few commits, no CI/CD, no tests, minimal README, no releases. Often a course project or learning exercise. May be a fork of a tutorial.

2. junior — Early-career work. 1-3 contributors, basic structure, some commits, maybe 1-2 tests, no CI/CD or very basic, short README, no formal releases. Shows basic coding ability but lacks professional engineering practices.

3. senior — Professional-grade project. Multiple contributors (3+), active CI/CD with multiple workflows, meaningful test coverage (10+ test files), regular releases with versioning, structured README with setup/usage docs, dependency management. Demonstrates software engineering best practices.

4. lead — Architect-level project. Large contributor base (10+), sophisticated CI/CD (linting, testing, deployment, security scanning), extensive test suites (50+ test files), documented architecture (docs/ folder present), multiple releases with changelogs, multiple dependency manifests, license file, 1000+ commits in 6 months. Shows technical leadership and architectural thinking.

5. template — A boilerplate/starter repository designed to be copied, not contributed to. Has template topics or described as "template/starter/boilerplate". Minimal commits after initial setup. Often has 0-1 contributors despite stars.

6. low-value — Repository with negligible engineering content. Abandoned (no recent commits despite age), empty or trivial README, no tests, no CI/CD, no releases, very few files. Not worth detailed review in a hiring context.

Classification rules:
- Base your decision on ALL signals provided, not just one or two.
- A repo with zero CI/CD and zero tests is rarely senior or lead, regardless of stars or age.
- A very old repo (1000+ days) with zero recent commits is likely low-value, not senior.
- Templates can have high stars but lack engineering depth.

Return ONLY a JSON object with this exact format:
{"category": "<one_of_the_six>", "confidence": <0.0_to_1.0>, "reasoning": "<one_sentence>"}
"""

# ---------------------------------------------------------------------------
# Prompt version 2 (ALTERNATIVE) — operationalized category definitions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V2 = """You are a deterministic repository classifier. Your task is to classify a GitHub repository into EXACTLY ONE of these 6 categories based on observable engineering signals.

You must prioritize observable repository evidence over inferred developer seniority. Do not use intuition, social inference, or "vibe" to classify. Base your decision on the signals provided.

Category definitions:

1. intern — A small learning-oriented repository with minimal engineering maturity but evidence of genuine implementation effort.

Typical signals: 1 contributor, no releases, no CI/CD workflows, very few tests, basic README, low or moderate recent activity, small project scope.

Important: intern repositories may show genuine effort and coherent implementation. Do NOT classify every small repository as low-value.

2. junior — An early-career engineering repository showing some software engineering structure but limited operational maturity.

Typical signals: 1-3 contributors, some recent commits, limited testing, minimal or basic CI/CD, partial documentation, limited release management.

These repositories demonstrate software development ability but incomplete engineering practices.

3. senior — A clearly professional-grade repository with strong evidence of engineering best practices.

Typical signals: multiple contributors, active maintenance, meaningful test presence, CI/CD workflows enabled, releases/versioning, structured README, dependency management, evidence of ongoing development.

The repository should demonstrate sustained engineering discipline across multiple signals simultaneously.

4. lead — A highly mature repository demonstrating advanced engineering operations and large-scale maintenance characteristics.

Typical signals: many contributors, multiple CI/CD workflows, extensive testing infrastructure, documentation folder present, strong release history, sustained recent activity, clear maintenance complexity, evidence of automation and project organization.

This label should ONLY be assigned when MULTIPLE strong maturity signals coexist simultaneously. Do NOT assign lead purely because a repository appears popular or has high contributor count.

5. template — A repository intended primarily as a reusable starter, scaffold, or boilerplate.

Typical signals: template/starter/boilerplate framing, low contributor diversity, low ongoing activity, little engineering evolution after initialization, reusable project structure.

Templates are structurally mature but operationally shallow. They may have high stars despite limited maintenance depth.

6. low-value — A repository with little meaningful engineering content or maintenance value.

Typical signals: abandoned or inactive, extremely limited implementation, trivial or empty README, no CI/CD, no tests, no releases, little recent activity, minimal repository structure.

Important distinction: unlike intern repositories, low-value repositories lack evidence of meaningful engineering effort or sustained development intent.

Classification rules:
- Use MULTIPLE signals simultaneously. Never classify based on a single feature.
- High contributors alone does NOT imply lead.
- Old age alone does NOT imply low-value.
- Stars do NOT imply maturity.
- intern is NOT low-value — intern repos show genuine effort despite small scale.
- senior is NOT lead — lead requires multiple strong maturity signals coexisting.
- If you are uncertain between two adjacent categories, choose the LOWER one.

Return ONLY a JSON object with this exact format:
{"category": "<one_of_the_six>", "confidence": <0.0_to_1.0>, "reasoning": "<one_sentence>"}
"""

# Mapping from version label to actual prompt text
PROMPTS = {
    "v1": SYSTEM_PROMPT_V1,
    "v2": SYSTEM_PROMPT_V2,
}

USER_PROMPT_TEMPLATE = """Classify this repository into one of these categories: intern, junior, senior, lead, template, low-value.

{text_repr}"""

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DeepSeek API client
# ---------------------------------------------------------------------------


class DeepSeekLabeler:
    """Calls DeepSeek API to label one repo at a time."""

    def __init__(self, api_key: str, model: str = MODEL,
                 prompt_version: str = "v1"):
        if prompt_version not in PROMPTS:
            raise ValueError(
                f"Unknown prompt_version '{prompt_version}'. "
                f"Available: {list(PROMPTS.keys())}"
            )
        self.prompt_version = prompt_version
        self.system_prompt = PROMPTS[prompt_version]
        self.api_key = api_key
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def label_one(self, text_repr: str) -> dict:
        """Send one repo's text representation to DeepSeek, return parsed label."""
        user_msg = USER_PROMPT_TEMPLATE.format(text_repr=text_repr)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": 150,
        }

        time.sleep(BATCH_DELAY)

        try:
            resp = self.session.post(DEEPSEEK_URL, json=payload, timeout=60)
        except requests.RequestException as exc:
            logger.error("Request failed: %s", exc)
            return {"category": "error", "confidence": 0.0, "reasoning": str(exc)}

        if resp.status_code != 200:
            logger.error("HTTP %d: %s", resp.status_code, resp.text[:300])
            return {
                "category": "error",
                "confidence": 0.0,
                "reasoning": f"HTTP {resp.status_code}",
            }

        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        return self._parse_response(content)

    def _parse_response(self, content: str) -> dict:
        """Extract JSON from LLM response. Handles markdown code fences."""
        # Remove markdown fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract category from text directly
            logger.warning("Could not parse JSON from: %s", content[:120])
            for cat in CATEGORIES:
                if cat in content.lower():
                    return {"category": cat, "confidence": 0.5,
                            "reasoning": "extracted from free text"}
            return {"category": "unknown", "confidence": 0.0,
                    "reasoning": "parse error"}

        cat = parsed.get("category", "unknown").lower().strip()
        if cat not in CATEGORIES:
            cat = "unknown"
        return {
            "category": cat,
            "confidence": float(parsed.get("confidence", 0.5)),
            "reasoning": str(parsed.get("reasoning", ""))[:200],
        }


# ---------------------------------------------------------------------------
# Labeling pipeline
# ---------------------------------------------------------------------------


def label_repositories(df: pd.DataFrame | None = None,
                       api_key: str | None = None,
                       prompt_version: str = "v1") -> pd.DataFrame:
    """
    Label all repositories using DeepSeek.

    Parameters
    ----------
    df : DataFrame or None
        If None, loads from data/processed/repo_summaries.csv
    api_key : str or None
        DeepSeek API key. If None, reads from DEEPSEEK_API_KEY env var.
    prompt_version : str
        "v1" (baseline, qualitative) or "v2" (alternative, operationalized).

    Returns
    -------
    DataFrame with label, confidence, reasoning columns added.
    Saved to data/labeled/combined_labeled.csv (v1) or
    data/labeled/combined_labeled_v2.csv (v2).
    """
    if df is None:
        df = load_csv("repo_summaries.csv", subdir="processed")

    if api_key is None:
        import os
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment or .env")

    labeler = DeepSeekLabeler(api_key, prompt_version=prompt_version)
    labels = []
    total = len(df)

    for i, row in df.iterrows():
        text = row["text_repr"]
        logger.info("[%d/%d] Labeling %s ...", i + 1, total, row.get("full_name", "?"))
        result = labeler.label_one(text)
        labels.append(result)
        logger.info("  → %s (confidence: %.2f)", result["category"], result["confidence"])

        # Save checkpoint every 50 repos
        if (i + 1) % 50 == 0:
            _save_checkpoint(df, labels, i + 1, prompt_version)

    # Merge labels into dataframe
    df["label"] = [l["category"] for l in labels]
    df["label_confidence"] = [l["confidence"] for l in labels]
    df["label_reasoning"] = [l["reasoning"] for l in labels]

    output_name = "combined_labeled.csv" if prompt_version == "v1" \
                  else f"combined_labeled_{prompt_version}.csv"
    save_csv(df, output_name, subdir="labeled")
    _print_distribution(df)
    return df


def _save_checkpoint(df: pd.DataFrame, labels: list, count: int,
                     prompt_version: str = "v1"):
    """Save partial results in case of interruption."""
    checkpoint_df = df.iloc[:count].copy()
    checkpoint_df["label"] = [l["category"] for l in labels]
    checkpoint_df["label_confidence"] = [l["confidence"] for l in labels]
    checkpoint_df["label_reasoning"] = [l["reasoning"] for l in labels]
    out_dir = get_project_root() / "data" / "labeled"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if prompt_version == "v1" else f"_{prompt_version}"
    checkpoint_df.to_csv(out_dir / f"checkpoint_labeled{suffix}.csv", index=False)
    logger.info("Checkpoint saved at %d repos", count)


def _print_distribution(df: pd.DataFrame):
    """Print label distribution."""
    dist = df["label"].value_counts()
    logger.info("=" * 40)
    logger.info("Label distribution:\n%s", dist.to_string())
    logger.info("Total labeled: %d", len(df))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import os
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set. Make sure .env exists with your API key.")
        sys.exit(1)

    labeled_dir = get_project_root() / "data" / "labeled"
    baseline_path = labeled_dir / "combined_labeled.csv"
    alternative_path = labeled_dir / "combined_labeled_v2.csv"

    if baseline_path.exists() and not alternative_path.exists():
        version = "v2"
        logger.info("Baseline labeling found (%s) — running ALTERNATIVE approach (v2)", baseline_path)
    elif not baseline_path.exists():
        version = "v1"
        logger.info("No baseline labeling found — running BASELINE approach (v1)")
    else:
        logger.info("Both baseline (v1) and alternative (v2) labeling files already exist.")
        print("Both combined_labeled.csv and combined_labeled_v2.csv exist.")
        print("Delete combined_labeled_v2.csv to re-run, or call label_repositories() directly.")
        sys.exit(0)

    logger.info("Starting LLM labeling with DeepSeek (prompt_version=%s)...", version)
    result = label_repositories(prompt_version=version)
    output_file = "combined_labeled.csv" if version == "v1" else "combined_labeled_v2.csv"
    logger.info("Labeling complete — %d repos saved to data/labeled/%s", len(result), output_file)
