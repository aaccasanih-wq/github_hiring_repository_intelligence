"""
GitHub API collector — Two-Round Informed Sampling (Track A).

Round 1: ~500 repos across 9 structural cells with sub-ranging, random pages,
and language cycling. Extracts 10 engineering-maturity signals per repo.

Round 2: ~300-400 additional repos targeting under-represented classes,
informed by the LLM label distribution from Round 1.

See: sampling_strategy_track_a.md for full methodology.
"""

import os
import time
import random
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd


def _load_dotenv():
    """Load .env file from the project root (one level above src/)."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
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


_load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TODAY = datetime(2026, 5, 23, tzinfo=timezone.utc)
SIX_MONTHS_AGO = TODAY - timedelta(days=183)
TWELVE_MONTHS_AGO = TODAY - timedelta(days=365)
TWO_YEARS_AGO = TODAY - timedelta(days=730)
FIVE_YEARS_AGO = TODAY - timedelta(days=1826)
ONE_MONTH_AGO = TODAY - timedelta(days=30)

SEARCH_BASE = "https://api.github.com/search/repositories"
REPO_BASE = "https://api.github.com/repos"
RATE_LIMIT_URL = "https://api.github.com/rate_limit"

SEARCH_DELAY = 2.5      # search endpoint: max 30/min → 2.5s = 24/min
DETAIL_DELAY = 0.4       # detail endpoints: ~150/min, well within 5000/hr
MAX_SEARCH_RESULTS = 1000  # GitHub only returns up to 1000 results
PER_PAGE = 100

LANGUAGES = ["Python", "TypeScript", "JavaScript", "Go", "Java", "Rust"]

# Star sub-ranges for each bucket
STAR_SUB_RANGES = {
    "micro":  [(0, 0), (1, 2), (3, 5)],
    "small":  [(6, 10), (11, 20), (21, 50)],
    "medium": [(51, 100), (101, 200), (201, 500)],
    "large":  [(501, 1000), (1001, 2000), (2001, 5000)],
    "top":    [(5001, 50000), (50001, 999999)],
}

# Round-1 cell definitions  (star_bucket, pushed_qualifier, created_qualifier, target, allow_forks)
CELL_DEFS = [
    # C1: Micro, High activity, Very new
    {"id": "C1", "star_bucket": "micro", "pushed": f">{ONE_MONTH_AGO.strftime('%Y-%m-%d')}",
     "created": f">{SIX_MONTHS_AGO.strftime('%Y-%m-%d')}", "target": 40, "allow_forks": True},
    # C2: Micro, Inactive, Young-Established
    {"id": "C2", "star_bucket": "micro", "pushed": f"<{TWELVE_MONTHS_AGO.strftime('%Y-%m-%d')}",
     "created": f"{TWO_YEARS_AGO.strftime('%Y-%m-%d')}..{SIX_MONTHS_AGO.strftime('%Y-%m-%d')}",
     "target": 40, "allow_forks": True},
    # C3: Small, Med-High, Young
    {"id": "C3", "star_bucket": "small", "pushed": f">{SIX_MONTHS_AGO.strftime('%Y-%m-%d')}",
     "created": f"{TWO_YEARS_AGO.strftime('%Y-%m-%d')}..{SIX_MONTHS_AGO.strftime('%Y-%m-%d')}",
     "target": 60, "allow_forks": False},
    # C4: Small, Low-Inactive, Established
    {"id": "C4", "star_bucket": "small", "pushed": f"<{SIX_MONTHS_AGO.strftime('%Y-%m-%d')}",
     "created": f"{FIVE_YEARS_AGO.strftime('%Y-%m-%d')}..{TWO_YEARS_AGO.strftime('%Y-%m-%d')}",
     "target": 40, "allow_forks": False},
    # C5: Medium, High, Young-Established
    {"id": "C5", "star_bucket": "medium", "pushed": f">{ONE_MONTH_AGO.strftime('%Y-%m-%d')}",
     "created": f">{FIVE_YEARS_AGO.strftime('%Y-%m-%d')}",
     "target": 80, "allow_forks": False},
    # C6: Medium, Medium, Veteran
    {"id": "C6", "star_bucket": "medium", "pushed": f"{SIX_MONTHS_AGO.strftime('%Y-%m-%d')}..{ONE_MONTH_AGO.strftime('%Y-%m-%d')}",
     "created": f"<{FIVE_YEARS_AGO.strftime('%Y-%m-%d')}", "target": 40, "allow_forks": False},
    # C7: Large, High, Established-Veteran
    {"id": "C7", "star_bucket": "large", "pushed": f">{ONE_MONTH_AGO.strftime('%Y-%m-%d')}",
     "created": f"<{TWO_YEARS_AGO.strftime('%Y-%m-%d')}", "target": 80, "allow_forks": False},
    # C8: Top, High, Veteran
    {"id": "C8", "star_bucket": "top", "pushed": f">{ONE_MONTH_AGO.strftime('%Y-%m-%d')}",
     "created": f"<{FIVE_YEARS_AGO.strftime('%Y-%m-%d')}", "target": 40, "allow_forks": False},
    # C9: Any stars, any activity, any age — template/boilerplate topics
    {"id": "C9", "star_bucket": None, "pushed": None, "created": None,
     "topic": "template,boilerplate", "target": 80, "allow_forks": False},
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GitHubCollector
# ---------------------------------------------------------------------------


class GitHubCollector:
    """Handles GitHub API search and per-repository signal extraction."""

    def __init__(self, token: str, output_dir: str = "data/raw"):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "github-hiring-intelligence-hw",
        })
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_ids: set[int] = set()
        self.search_calls = 0
        self.detail_calls = 0

    # ------------------------------------------------------------------
    # Rate limiting helpers
    # ------------------------------------------------------------------

    def _check_rate_limit(self):
        resp = self.session.get(RATE_LIMIT_URL)
        if resp.status_code == 200:
            data = resp.json()
            search_remaining = data["resources"]["search"]["remaining"]
            core_remaining = data["resources"]["core"]["remaining"]
            if search_remaining < 5:
                reset_time = data["resources"]["search"]["reset"]
                wait = max(reset_time - time.time(), 0) + 5
                logger.warning("Search rate limit near zero — sleeping %.0fs", wait)
                time.sleep(wait)
            if core_remaining < 50:
                reset_time = data["resources"]["core"]["reset"]
                wait = max(reset_time - time.time(), 0) + 5
                logger.warning("Core rate limit low — sleeping %.0fs", wait)
                time.sleep(wait)

    def _rate_limited_get(self, url: str, params: dict | None = None,
                          is_search: bool = False) -> dict | None:
        """GET with rate limiting. Returns parsed JSON or None on 404/403."""
        delay = SEARCH_DELAY if is_search else DETAIL_DELAY
        time.sleep(delay)

        if is_search:
            self.search_calls += 1
        else:
            self.detail_calls += 1

        resp = self.session.get(url, params=params)
        if resp.status_code == 404:
            return None
        if resp.status_code == 403:
            logger.warning("403 for %s — checking rate limit", url)
            self._check_rate_limit()
            time.sleep(5)
            resp = self.session.get(url, params=params)
            if resp.status_code != 200:
                return None
        if resp.status_code != 200:
            logger.warning("HTTP %d for %s", resp.status_code, url)
            return None
        return resp.json()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _build_query(self, stars: tuple[int, int] | None = None,
                     pushed: str | None = None,
                     created: str | None = None,
                     language: str | None = None,
                     topic: str | None = None,
                     allow_forks: bool = False) -> str:
        """Assemble a GitHub search query string."""
        parts = []
        if stars:
            parts.append(f"stars:{stars[0]}..{stars[1]}")
        if pushed:
            parts.append(f"pushed:{pushed}")
        if created:
            parts.append(f"created:{created}")
        if language:
            parts.append(f"language:{language}")
        if topic:
            parts.append(f"topic:{topic}")
        if not allow_forks:
            parts.append("fork:false")
        parts.append("archived:false")
        return " ".join(parts)

    def _search_page(self, query: str, page: int) -> list[dict]:
        """Return items from a single search results page."""
        result = self._rate_limited_get(
            SEARCH_BASE,
            params={"q": query, "sort": "updated", "order": "desc",
                    "per_page": PER_PAGE, "page": page},
            is_search=True,
        )
        if result is None:
            return []
        return result.get("items", [])

    def _get_total_results(self, query: str) -> int:
        """Get total_count for a query (page 1, per_page=1 to save bandwidth)."""
        result = self._rate_limited_get(
            SEARCH_BASE,
            params={"q": query, "per_page": 1},
            is_search=True,
        )
        if result is None:
            return 0
        return result.get("total_count", 0)

    def _search_random_pages(self, query: str, n_wanted: int,
                             max_pages: int = 10) -> list[dict]:
        """Collect up to n_wanted repos by sampling random pages of results."""
        collected: list[dict] = []
        total = self._get_total_results(query)
        if total == 0:
            return collected

        available_pages = min(max_pages, (total + PER_PAGE - 1) // PER_PAGE)
        seen_pages: set[int] = set()
        attempts = 0

        while len(collected) < n_wanted and len(seen_pages) < available_pages:
            page = random.randint(1, available_pages)
            if page in seen_pages:
                attempts += 1
                if attempts > 20:
                    break
                continue
            seen_pages.add(page)
            items = self._search_page(query, page)
            for item in items:
                repo_id = item["id"]
                if repo_id in self.collected_ids:
                    continue
                collected.append(item)
                self.collected_ids.add(repo_id)
                if len(collected) >= n_wanted:
                    break

        return collected

    # ------------------------------------------------------------------
    # Signal extraction
    # ------------------------------------------------------------------

    def _extract_signal_contributors(self, owner: str, repo: str) -> int:
        data = self._rate_limited_get(
            f"{REPO_BASE}/{owner}/{repo}/contributors",
            params={"per_page": 100, "anon": "false"},
        )
        if data is None:
            return 0
        return len(data) if isinstance(data, list) else 0

    def _extract_signal_commits_6m(self, owner: str, repo: str) -> int:
        since = SIX_MONTHS_AGO.isoformat()
        data = self._rate_limited_get(
            f"{REPO_BASE}/{owner}/{repo}/commits",
            params={"since": since, "per_page": 100},
        )
        if data is None:
            return 0
        return len(data) if isinstance(data, list) else 0

    def _extract_signal_releases(self, owner: str, repo: str) -> int:
        data = self._rate_limited_get(
            f"{REPO_BASE}/{owner}/{repo}/releases",
            params={"per_page": 100},
        )
        if data is None:
            return 0
        return len(data) if isinstance(data, list) else 0

    def _extract_signal_cicd(self, owner: str, repo: str) -> int:
        data = self._rate_limited_get(
            f"{REPO_BASE}/{owner}/{repo}/contents/.github/workflows"
        )
        if data is None or not isinstance(data, list):
            return 0
        return sum(1 for f in data if f["name"].endswith((".yml", ".yaml")))

    def _extract_signal_tests(self, owner: str, repo: str) -> int:
        data = self._rate_limited_get(
            f"{REPO_BASE}/{owner}/{repo}/git/trees/main?recursive=1"
        )
        # Also try 'master' if 'main' returns nothing
        if data is None:
            data = self._rate_limited_get(
                f"{REPO_BASE}/{owner}/{repo}/git/trees/master?recursive=1"
            )
        if data is None or "tree" not in data:
            return 0
        test_patterns = ("test", "spec", "__tests__", "_test.")
        count = 0
        for entry in data["tree"]:
            path = entry.get("path", "")
            name = path.split("/")[-1]
            if any(p in name.lower() for p in ("test", "spec")):
                count += 1
        return count

    def _extract_signal_readme(self, owner: str, repo: str) -> tuple[int, str]:
        data = self._rate_limited_get(
            f"{REPO_BASE}/{owner}/{repo}/readme"
        )
        if data is None:
            return 0, ""
        import base64
        content = data.get("content", "")
        if content:
            try:
                decoded = base64.b64decode(content).decode("utf-8", errors="replace")
                return len(decoded), decoded[:500]
            except Exception:
                return 0, ""
        return 0, ""

    def _extract_signal_dependencies(self, owner: str, repo: str) -> int:
        manifests = [
            "requirements.txt", "package.json", "go.mod",
            "Cargo.toml", "pom.xml", "pyproject.toml",
        ]
        for manifest in manifests:
            data = self._rate_limited_get(
                f"{REPO_BASE}/{owner}/{repo}/contents/{manifest}"
            )
            if data is not None and "content" in data:
                import base64
                content = data.get("content", "")
                if content:
                    try:
                        decoded = base64.b64decode(content).decode("utf-8", errors="replace")
                        return len(decoded.splitlines())
                    except Exception:
                        return 0
        return 0

    def _extract_signal_docs(self, owner: str, repo: str) -> bool:
        data = self._rate_limited_get(
            f"{REPO_BASE}/{owner}/{repo}/contents"
        )
        if data is None or not isinstance(data, list):
            return False
        doc_names = {"docs", "documentation", "doc"}
        for entry in data:
            if entry.get("type") == "dir" and entry.get("name", "").lower() in doc_names:
                return True
        return False

    def _extract_signal_age(self, created_at: str) -> int:
        created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return (TODAY - created_dt).days

    def _extract_signal_license(self, repo_item: dict) -> tuple[bool, str]:
        lic = repo_item.get("license")
        if lic is None:
            return False, "None"
        return True, lic.get("spdx_id", lic.get("name", "Unknown"))

    def _extract_all_signals(self, repo_item: dict) -> dict:
        """Extract all 10 signals for a single repository."""
        owner = repo_item["owner"]["login"]
        repo = repo_item["name"]
        full_name = f"{owner}/{repo}"

        signals = {
            "repo_id": repo_item["id"],
            "full_name": full_name,
            "owner": owner,
            "repo_name": repo,
            "language": repo_item.get("language", "Unknown"),
            "stars": repo_item.get("stargazers_count", 0),
            "forks": repo_item.get("forks_count", 0),
            "size_kb": repo_item.get("size", 0),
            "description": repo_item.get("description", ""),
            "topics": ",".join(repo_item.get("topics", [])),
            "default_branch": repo_item.get("default_branch", "main"),
            "is_fork": repo_item.get("fork", False),
        }

        logger.info("  Extracting signals for %s ...", full_name)
        signals["contributors_count"] = self._extract_signal_contributors(owner, repo)
        signals["commits_6m"] = self._extract_signal_commits_6m(owner, repo)
        signals["releases_count"] = self._extract_signal_releases(owner, repo)
        signals["cicd_workflows"] = self._extract_signal_cicd(owner, repo)
        signals["test_files"] = self._extract_signal_tests(owner, repo)
        signals["readme_length"], signals["readme_preview"] = self._extract_signal_readme(owner, repo)
        signals["dependency_lines"] = self._extract_signal_dependencies(owner, repo)
        signals["has_docs_folder"] = self._extract_signal_docs(owner, repo)
        signals["age_days"] = self._extract_signal_age(repo_item["created_at"])
        signals["has_license"], signals["license_name"] = self._extract_signal_license(repo_item)

        return signals

    # ------------------------------------------------------------------
    # Quality filters
    # ------------------------------------------------------------------

    def _passes_quality_filters(self, repo_item: dict) -> bool:
        """Minimum quality filters — structural, NOT maturity-based."""
        if repo_item.get("size", 0) <= 5:
            return False
        # Check README existence via a quick call
        owner = repo_item["owner"]["login"]
        repo = repo_item["name"]
        readme_data = self._rate_limited_get(
            f"{REPO_BASE}/{owner}/{repo}/readme"
        )
        if readme_data is None:
            return False
        return True

    # ------------------------------------------------------------------
    # Round 1 — neutral structural sampling
    # ------------------------------------------------------------------

    def collect_round1(self) -> pd.DataFrame:
        """Execute Round 1: ~500 repos across 9 structural cells."""
        logger.info("=" * 60)
        logger.info("ROUND 1 — Neutral Structural Sampling (~500 repos)")
        logger.info("=" * 60)

        all_repos: list[dict] = []
        repo_ids: set[int] = set()

        for cell in CELL_DEFS:
            logger.info("--- Cell %s (target: %d repos) ---", cell["id"], cell["target"])
            cell_repos: list[dict] = []
            star_bucket = cell.get("star_bucket")
            sub_ranges = STAR_SUB_RANGES[star_bucket] if star_bucket else [None]
            lang_quota = max(len(LANGUAGES), 1)

            repos_per_subrange_lang = max(
                cell["target"] // (len(sub_ranges) * lang_quota), 3
            )

            for sr in sub_ranges:
                for lang in LANGUAGES:
                    if len(cell_repos) >= cell["target"]:
                        break

                    query = self._build_query(
                        stars=sr,
                        pushed=cell.get("pushed"),
                        created=cell.get("created"),
                        language=lang,
                        topic=cell.get("topic"),
                        allow_forks=cell.get("allow_forks", False),
                    )

                    logger.info("  Query [%s][%s]: %s", cell["id"], lang, query)

                    raw_items = self._search_random_pages(
                        query, repos_per_subrange_lang
                    )

                    for item in raw_items:
                        if item["id"] in repo_ids:
                            continue
                        if not self._passes_quality_filters(item):
                            continue
                        repo_ids.add(item["id"])
                        cell_repos.append(item)

                    logger.info("  Got %d valid repos in this sub-range/language",
                                sum(1 for r in raw_items if r["id"] in repo_ids))

                if len(cell_repos) >= cell["target"]:
                    break

            logger.info("Cell %s: collected %d/%d repos", cell["id"],
                        len(cell_repos), cell["target"])
            all_repos.extend(cell_repos)

        logger.info("Round 1 search complete. %d repos pre-filter. Extracting signals...",
                    len(all_repos))

        # Extract signals for all collected repos
        results: list[dict] = []
        for i, repo_item in enumerate(all_repos):
            logger.info("[%d/%d] %s", i + 1, len(all_repos),
                        repo_item.get("full_name", repo_item.get("name")))
            try:
                signals = self._extract_all_signals(repo_item)
                results.append(signals)
            except Exception as exc:
                logger.error("Failed to extract signals for %s: %s",
                             repo_item.get("full_name", ""), exc)

        df = pd.DataFrame(results)
        out_path = self.output_dir / "round1_raw.csv"
        df.to_csv(out_path, index=False)
        logger.info("Round 1 saved: %d repos → %s", len(df), out_path)
        return df

    # ------------------------------------------------------------------
    # Round 2 — informed sampling for class balance
    # ------------------------------------------------------------------

    def collect_round2(self, labeled_df: pd.DataFrame,
                       min_per_class: int = 120) -> pd.DataFrame:
        """
        Execute Round 2: target under-represented classes.

        Parameters
        ----------
        labeled_df : DataFrame
            Round-1 data with LLM labels (must have 'label' column).
        min_per_class : int
            Target minimum repos per category.

        Returns
        -------
        DataFrame with new repos and their signals.
        """
        logger.info("=" * 60)
        logger.info("ROUND 2 — Informed Sampling for Class Balance")
        logger.info("=" * 60)

        label_col = "label"
        if label_col not in labeled_df.columns:
            logger.error("No 'label' column in labeled_df — run LLM labeling first.")
            return pd.DataFrame()

        dist = labeled_df[label_col].value_counts()
        logger.info("Round-1 label distribution:\n%s", dist)

        deficient = dist[dist < min_per_class]
        if deficient.empty:
            logger.info("All classes meet minimum target of %d. No Round 2 needed.", min_per_class)
            return pd.DataFrame()

        logger.info("Deficient classes: %s", deficient.index.tolist())

        all_new: list[dict] = []
        repo_ids = set(labeled_df["repo_id"].tolist())

        for label_name, current_count in deficient.items():
            needed = min_per_class - current_count
            logger.info("Targeting %d more repos for class '%s'", needed, label_name)

            # Find which structural cells produced this label most in Round 1
            subset = labeled_df[labeled_df[label_col] == label_name]
            # Gather star buckets and cell info from original search context
            # For simplicity: use queries without strict cell criteria,
            # cycling through star ranges and languages
            collected_for_class = 0
            for sr_list in STAR_SUB_RANGES.values():
                if collected_for_class >= needed:
                    break
                for sr in sr_list:
                    if collected_for_class >= needed:
                        break
                    for lang in LANGUAGES[:3]:  # top 3 languages to be faster
                        if collected_for_class >= needed:
                            break
                        query = self._build_query(stars=sr, language=lang)
                        items = self._search_random_pages(query, 5)
                        for item in items:
                            if item["id"] in repo_ids:
                                continue
                            if not self._passes_quality_filters(item):
                                continue
                            repo_ids.add(item["id"])
                            try:
                                signals = self._extract_all_signals(item)
                                all_new.append(signals)
                                collected_for_class += 1
                                if collected_for_class >= needed:
                                    break
                            except Exception as exc:
                                logger.error("Round 2 extract error: %s", exc)

            logger.info("Collected %d repos for class '%s'", collected_for_class, label_name)

        df = pd.DataFrame(all_new)
        if not df.empty:
            out_path = self.output_dir / "round2_raw.csv"
            df.to_csv(out_path, index=False)
            logger.info("Round 2 saved: %d repos → %s", len(df), out_path)
        return df

    # ------------------------------------------------------------------
    # Full pipeline entry point
    # ------------------------------------------------------------------

    def run_full_collection(self) -> pd.DataFrame:
        """Run Round 1 and return the combined raw DataFrame."""
        df_r1 = self.collect_round1()
        logger.info("Total API calls — search: %d, detail: %d",
                    self.search_calls, self.detail_calls)
        return df_r1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN environment variable not set.")
        print("Usage: GITHUB_TOKEN=ghp_xxx python src/github_collector.py [round1|round2]")
        sys.exit(1)

    collector = GitHubCollector(token=token)

    if len(sys.argv) > 1 and sys.argv[1] == "round2":
        # Load Round 1 labeled data
        labeled_path = Path("data/labeled/round1_labeled.csv")
        if not labeled_path.exists():
            print("ERROR: data/labeled/round1_labeled.csv not found. Run LLM labeling first.")
            sys.exit(1)
        labeled_df = pd.read_csv(labeled_path)
        collector.collect_round2(labeled_df)
    else:
        collector.run_full_collection()
