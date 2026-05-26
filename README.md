# GitHub Hiring Repository Intelligence

**Weak-supervision NLP pipeline that classifies GitHub repositories into 6 engineering-maturity categories using DeepSeek LLM for labeling + DistilBERT fine-tuning.**

## What does the project do?

This project automatically classifies GitHub repositories into engineering maturity levels — from intern-level personal projects to lead-level large-scale codebases. It uses a **weak supervision** approach: a large language model (DeepSeek) generates training labels through carefully designed prompts, then a lightweight classifier (DistilBERT) is fine-tuned on the resulting weakly-labeled dataset. The pipeline ingests raw GitHub repository metadata, transforms it into structured textual representations, labels it via LLM, and trains a BERT-based classifier that can predict maturity categories on unseen repositories.

## Which track was selected?

**Track A** — GitHub Hiring Repository Intelligence. This track focuses on extracting engineering-maturity signals from public GitHub repositories to support hiring and talent-sourcing use cases.

## What repositories were analyzed?

**771 repositories** were collected from GitHub using a **two-round informed sampling strategy**:

- **Round 1** (469 repos): Broad queries across 9 structural cells combining star ranges (micro→top), activity levels, repository age, and 6 programming languages (Python, TypeScript, JavaScript, Go, Java, Rust). Sampling was purely structural — no maturity labels were used in search queries.
- **Round 2** (302 repos): Targeted follow-up queries informed by Round 1's label distribution to fill structural gaps in under-represented segments of the contributor-count and activity distributions.

**Quality filters** were structural, not maturity-based: size > 5KB, README must exist, repository must not be archived.

## Which GitHub signals were used?

**10 engineering signals** were extracted per repository, covering orthogonal dimensions of software engineering maturity:

| Signal | Dimension Captured | Extraction Method |
|--------|-------------------|-------------------|
| Stars | Community interest | GitHub API |
| Forks | Derivative usage | GitHub API |
| Contributors count | Team size / collaboration | GitHub API |
| Commits (6 months) | Development velocity | GitHub API |
| Releases count | Release discipline | GitHub API |
| CI/CD workflows | Automation maturity | File-tree inspection (`.github/workflows`) |
| Test files | Testing culture | Recursive file-tree inspection |
| README length | Documentation quality | GitHub API (base64-decoded) |
| Dependency lines | Ecosystem integration | Dependency manifest parsing (6 formats) |
| Repository age (days) | Longevity / sustainability | GitHub API (`created_at`) |

## How were repository summaries created?

Each repository was converted into a **structured textual representation** (`text_repr`) averaging **712 characters**, well within DistilBERT's 512-token limit. The format is deliberately consistent across all repos:

```
Repository: owner/name | Language: Python
README: <first 500 chars of README>
CI/CD workflows: 3 | Test files: 12
Contributors: 8 | Releases: 15 | Commits (last 6m): 47
Dependencies: 94 | Docs folder: Yes | License: MIT
Repository age: 1200 days
```

Critically, the text representation **excludes** popularity-correlated metadata such as stars, forks, owner name, and repository topics. This prevents the classifier from learning trivial shortcuts and forces it to attend to genuine engineering structure.

## How were prompts designed?

Two prompt versions were used in a **controlled methodological sensitivity analysis**:

### V1 — Baseline (qualitative)
Category definitions described maturity levels in terms of engineering character rather than specific, observable criteria. For example, "senior" was described as "demonstrating established engineering practices with consistent activity," and "lead" as "showing architectural leadership and sophisticated automation."

### V2 — Alternative (operationalized)
Category definitions were anchored in **concrete, observable engineering signals** with explicit criteria. For example, "senior" required "CI/CD configured, test files present, multiple contributors with consistent commits, documented releases, and README exceeding 500 characters." The key principle was reducing annotator degrees of freedom by defining every category in terms of verifiable repository properties.

Both prompts used the same **6 maturity categories**:

| Label | ID | Description |
|-------|----|-------------|
| intern | 0 | Entry-level; learning-oriented; minimal engineering infrastructure |
| junior | 1 | Emerging professional; basic CI/CD and testing; small team |
| senior | 2 | Established practices; consistent activity; documented releases |
| lead | 3 | Architectural leadership; large contributor base; sophisticated automation |
| template | 4 | Boilerplate/scaffolding; little original code beyond project structure |
| low-value | 5 | Minimal engineering signals across most dimensions |

**Labeling configuration:** DeepSeek API, `temperature=0` (deterministic), checkpoints every 50 repos, structured JSON output with category, confidence score, and reasoning.

## How was the dataset split?

**Stratified 70/15/15 split** with `seed=42`:

| Split | Repos | Percentage |
|-------|-------|------------|
| Train | 539 | 70% |
| Validation | 115 | 15% |
| Test | 116 | 15% |

Stratification preserved the label distribution across splits. Both V1 (`train.csv`, `val.csv`, `test.csv`) and V2 (`train_v2.csv`, `val_v2.csv`, `test_v2.csv`) splits were generated.

## Which BERT model was used?

**DistilBERT** (`distilbert-base-uncased`, 66M parameters) — a lightweight Transformer that retains ~95% of BERT-base's performance with 40% fewer parameters. It trains on a single T4 GPU in under 10 minutes.

**Training configuration:**
- Input: `text_repr` column
- Output: `label_id` (0–5)
- Batch size: 16
- Epochs: 8 (with early stopping, patience=3)
- Learning rate: 2e-5 (AdamW)
- Weight decay: 0.01
- Class-weighted cross-entropy loss (to handle 3.6:1 class imbalance)
- Max sequence length: 512 tokens
- Reproducibility: `seed=42`, all random seeds set

## What were the final metrics?

### V1 vs V2 comparative results

| Metric | Baseline (V1) | Alternative (V2) | Delta |
|--------|---------------|------------------|-------|
| Accuracy | 40.5% | **61.2%** | +20.7pp |
| Macro Precision | 38.8% | **57.9%** | +19.1pp |
| Macro Recall | 43.8% | **61.2%** | +17.4pp |
| Macro F1 | 39.3% | **58.8%** | +19.5pp |
| Weighted F1 | 37.5% | **60.9%** | +23.4pp |

### V2 per-class performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| intern | 0.625 | 0.600 | 0.612 | 25 |
| junior | 0.250 | 0.200 | 0.222 | 10 |
| senior | 0.636 | 0.568 | 0.600 | 37 |
| lead | 0.680 | 0.708 | 0.694 | 24 |
| template | 0.818 | 0.818 | 0.818 | 11 |
| low-value | 0.500 | 0.778 | 0.583 | 9 |

### Key finding

The 20.7-point accuracy improvement from changing **only the labeling prompt** demonstrates that **weak-label operationalization quality has a larger impact on downstream performance than model capacity or feature engineering.** The baseline model's 40.5% accuracy was primarily a label problem, not a model problem.

## What are the main limitations?

1. **No ground-truth labels.** All labels are LLM-generated. Without human-annotated validation data, we cannot distinguish LLM labeling errors from classifier errors.

2. **Proxy features.** The 10 signals are structural proxies for engineering maturity, not direct measurements of code quality, architectural decisions, or community health.

3. **Sampling bias.** Quality filters (size > 5KB, README required, non-archived) systematically exclude abandoned repositories without documentation, underrepresenting the low-value category.

4. **Ordinal task treated as nominal.** Maturity labels form an ordinal scale (intern < junior < senior < lead), but flat cross-entropy loss treats all misclassifications as equally costly.

5. **Single seed, single run.** Results come from one training run with `seed=42`. Without multiple restarts, metrics are point estimates with unknown variance.

6. **Junior is structurally underdetermined.** Junior occupies an interstitial space between intern and senior with no distinctive structural signature. Both approaches struggled with this category (V2 F1 = 0.222).

7. **Ethical considerations.** Repository signals can be gamed (inflated stars, templated CI/CD). Organizational context is invisible to the classifier. This system produces a noisy proxy signal, not an authoritative assessment of individual engineering competence.

## What are the possible business applications?

- **Talent sourcing.** Recruiters can identify engineers whose public work reflects a given experience level, complementing resume-based screening with behavioral evidence from repositories.
- **Dependency risk assessment.** Engineering managers evaluating open-source dependencies can assess whether a library is maintained with professional-grade practices or represents a personal side project.
- **Repository triage.** Automated categorization of boilerplate/template projects vs. professionally maintained codebases at scale.
- **Technical interview calibration.** Repository evidence as one signal among many when calibrating candidate expectations in interview pipelines.
- **Engineering team intelligence.** Understanding the maturity distribution of repositories within an organization or ecosystem.

## How to run the project?

### Prerequisites
- **macOS** with conda
- Python 3.9.23
- GitHub API token (`GITHUB_TOKEN`)
- DeepSeek API key (`DEEPSEEK_API_KEY`)

### Setup
```bash
# Clone and set up environment
git clone <repo-url>
cd github_hiring_repository_intelligence
conda create -n github_hiring_intelligence python=3.9.23
conda activate github_hiring_intelligence
pip install -r requirements.txt

# Configure secrets
cp .env.example .env
# Edit .env with your GITHUB_TOKEN and DEEPSEEK_API_KEY
```

### Pipeline stages

```bash
# Stage 1: Collect repositories from GitHub
python src/github_collector.py

# Stage 2: Preprocess and create text summaries
python src/preprocessing.py
python src/summarization.py

# Stage 3: LLM labeling with DeepSeek
python src/llm_labeling.py              # V1 (baseline)
APPROACH=v2 python src/llm_labeling.py  # V2 (alternative)

# Stage 4: Train/val/test split
python src/train.py --split-only

# Stage 5: Fine-tune DistilBERT
python src/train.py                 # V1
APPROACH=v2 python src/train.py     # V2

# Stage 6: Evaluate and generate metrics + figures
python src/evaluation.py
python src/visualization.py
```

All scripts run from the project root. Secrets in `.env` are auto-loaded by `src/utils.py:load_dotenv()`.

## How to run the Streamlit app?

```bash
conda activate github_hiring_intelligence
streamlit run app.py
```

The dashboard has **4 tabs**:

| Tab | Content |
|-----|---------|
| **Problem & Methodology** | Project overview, two-round sampling strategy, weak supervision pipeline, ethical considerations |
| **Exploratory Analysis** | Signal distributions, correlation heatmap, category breakdowns, V1/V2 label comparison, Sankey migration diagram, CI/CD adoption analysis |
| **Model Results** | V1 vs V2 confusion matrices, per-class F1 comparison, metric delta bars, key findings and interpretation |
| **Interactive Repository Exploration** | Search, filter by language/category/stars, V1/V2 label side-by-side comparison, README preview |

The app loads data from `data/labeled/`, `output/`, and `output/v2/`. All data loaders use Streamlit `@st.cache_data` for performance.
