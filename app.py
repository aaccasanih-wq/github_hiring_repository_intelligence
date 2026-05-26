"""
Streamlit App — Repository Maturity Classification Dashboard
============================================================
Track A: GitHub Hiring Repository Intelligence.

4-tab professional ML experimentation dashboard:
  1. Problem & Methodology
  2. Exploratory Analysis
  3. Model Results
  4. Interactive Repository Exploration

Run: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# -- project root -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# -- constants ----------------------------------------------------------
CLASS_ORDER = ["intern", "junior", "senior", "lead", "template", "low-value"]
LABEL_TO_ID = {c: i for i, c in enumerate(CLASS_ORDER)}
ID_TO_LABEL = {i: c for i, c in enumerate(CLASS_ORDER)}

CLASS_COLORS = {
    "intern": "#1F77B4",
    "junior": "#FF7F0E",
    "senior": "#2CA02C",
    "lead": "#D62728",
    "template": "#9467BD",
    "low-value": "#7F7F7F",
}

CLASS_DESCRIPTIONS = {
    "intern": "Entry-level; learning-oriented; minimal engineering infrastructure",
    "junior": "Emerging professional; basic CI/CD and testing; small team",
    "senior": "Established practices; consistent activity; documented releases",
    "lead": "Architectural leadership; large contributor base; sophisticated automation",
    "template": "Boilerplate/scaffolding; little original code beyond project structure",
    "low-value": "Minimal engineering signals across most dimensions",
}

# -- page config --------------------------------------------------------
st.set_page_config(
    page_title="Repo Maturity Classification",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -- custom CSS ---------------------------------------------------------
st.markdown(
    """
<style>
    /* Clean typography */
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
        font-style: italic;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #2CA02C;
        padding: 0.8rem 1rem;
        margin: 0.8rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.92rem;
    }
    .metric-highlight {
        background: #e8f4f8;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        text-align: center;
        font-weight: 600;
    }
    .caption {
        font-size: 0.82rem;
        color: #888;
        margin-top: 0.2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ======================================================================
# DATA LOADING (cached)
# ======================================================================

@st.cache_data(show_spinner=False)
def load_labeled_data(version: str = "v1") -> pd.DataFrame:
    """Load the full labeled dataset (V1 or V2)."""
    filename = "combined_labeled.csv" if version == "v1" else "combined_labeled_v2.csv"
    path = PROJECT_ROOT / "data" / "labeled" / filename
    df = pd.read_csv(path)
    # Ensure label is categorical
    df["label"] = pd.Categorical(df["label"], categories=CLASS_ORDER, ordered=True)
    return df


@st.cache_data(show_spinner=False)
def load_merged_labels() -> pd.DataFrame:
    """Merge V1 and V2 labels for comparison."""
    v1 = load_labeled_data("v1")[
        ["repo_id", "full_name", "label", "label_confidence", "label_reasoning"]
    ].copy()
    v1.rename(
        columns={
            "label": "label_v1",
            "label_confidence": "confidence_v1",
            "label_reasoning": "reasoning_v1",
        },
        inplace=True,
    )

    v2 = load_labeled_data("v2")[
        ["repo_id", "full_name", "label", "label_confidence", "label_reasoning"]
    ].copy()
    v2.rename(
        columns={
            "label": "label_v2",
            "label_confidence": "confidence_v2",
            "label_reasoning": "reasoning_v2",
        },
        inplace=True,
    )

    merged = v1.merge(v2, on=["repo_id", "full_name"], how="inner")
    merged["labels_agree"] = merged["label_v1"] == merged["label_v2"]
    return merged


@st.cache_data(show_spinner=False)
def load_test_predictions(version: str = "v1") -> pd.DataFrame:
    """Load test predictions (V1 or V2)."""
    if version == "v1":
        path = PROJECT_ROOT / "output" / "test_predictions.csv"
    else:
        path = PROJECT_ROOT / "output" / "v2" / "test_predictions.csv"
    df = pd.read_csv(path)
    df["label"] = pd.Categorical(df["label"], categories=CLASS_ORDER, ordered=True)
    df["predicted_label"] = pd.Categorical(
        df["predicted_label"], categories=CLASS_ORDER, ordered=True
    )
    return df


@st.cache_data(show_spinner=False)
def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute classification metrics from prediction dataframe."""
    y_true = df["label"].values
    y_pred = df["predicted_label"].values

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    per_class = classification_report(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0, output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    return {
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_norm,
    }


@st.cache_data(show_spinner=False)
def load_full_repo_data() -> pd.DataFrame:
    """Load the full dataset with signals + labels for exploration."""
    df = load_labeled_data("v1")
    # Add V2 labels
    v2 = load_labeled_data("v2")[["repo_id", "label"]].copy()
    v2.rename(columns={"label": "label_v2"}, inplace=True)
    df = df.merge(v2, on="repo_id", how="left")
    return df


# ======================================================================
# PLOTTING HELPERS
# ======================================================================

def plot_confusion_heatmap(cm_norm: np.ndarray, title: str) -> go.Figure:
    """Plotly interactive normalized confusion matrix heatmap."""
    annot = [
        [f"{val:.0%}" if val > 0 else "" for val in row] for row in cm_norm
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_norm,
            x=CLASS_ORDER,
            y=CLASS_ORDER,
            text=annot,
            texttemplate="%{text}",
            textfont={"size": 13},
            colorscale="Blues",
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar={"title": "Proportion of true class", "tickformat": ".0%"},
            hovertemplate=(
                "True: %{y}<br>Pred: %{x}<br>Proportion: %{z:.1%}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        width=620,
        height=560,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        xaxis={"side": "bottom"},
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_per_class_f1_comparison(metrics_v1: dict, metrics_v2: dict) -> go.Figure:
    """Grouped bar chart: per-class F1 for V1 vs V2."""
    v1_f1 = [metrics_v1["per_class"][c]["f1-score"] for c in CLASS_ORDER]
    v2_f1 = [metrics_v2["per_class"][c]["f1-score"] for c in CLASS_ORDER]
    deltas = [v2 - v1 for v1, v2 in zip(v1_f1, v2_f1)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Baseline (V1)",
            x=CLASS_ORDER,
            y=v1_f1,
            marker_color="#4C72B0",
            text=[f"{v:.3f}" for v in v1_f1],
            textposition="outside",
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Alternative (V2)",
            x=CLASS_ORDER,
            y=v2_f1,
            marker_color="#DD8452",
            text=[f"{v:.3f}" for v in v2_f1],
            textposition="outside",
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        )
    )
    # Delta annotations
    for i, (c, d) in enumerate(zip(CLASS_ORDER, deltas)):
        sign = "+" if d >= 0 else ""
        color = "#2CA02C" if d >= 0 else "#D62728"
        fig.add_annotation(
            x=c,
            y=max(v1_f1[i], v2_f1[i]) + 0.08,
            text=f"{sign}{d:.3f}",
            showarrow=False,
            font={"size": 11, "color": color},
        )

    fig.update_layout(
        title="Per-Class F1: Baseline vs Alternative",
        yaxis_title="F1 Score",
        yaxis={"range": [0, 1.15]},
        width=750,
        height=460,
        barmode="group",
        bargap=0.18,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def plot_metric_delta_chart(metrics_v1: dict, metrics_v2: dict) -> go.Figure:
    """Horizontal bar chart of metric deltas (V2 - V1)."""
    metric_names = ["Accuracy", "Macro F1", "Weighted F1", "Macro Precision", "Macro Recall"]
    v1_vals = [
        metrics_v1["accuracy"],
        metrics_v1["macro_f1"],
        metrics_v1["weighted_f1"],
        metrics_v1["macro_precision"],
        metrics_v1["macro_recall"],
    ]
    v2_vals = [
        metrics_v2["accuracy"],
        metrics_v2["macro_f1"],
        metrics_v2["weighted_f1"],
        metrics_v2["macro_precision"],
        metrics_v2["macro_recall"],
    ]
    deltas = [v2 - v1 for v1, v2 in zip(v1_vals, v2_vals)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=metric_names,
            x=deltas,
            orientation="h",
            marker_color=["#2CA02C" if d >= 0 else "#D62728" for d in deltas],
            text=[f"+{d:.3f}" if d >= 0 else f"{d:.3f}" for d in deltas],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(
        title="Metric Improvements (V2 − V1)",
        xaxis_title="Delta (percentage points)",
        width=600,
        height=320,
        margin={"l": 10, "r": 40, "t": 50, "b": 10},
        showlegend=False,
    )
    return fig


def plot_signal_by_category(
    df: pd.DataFrame, signal: str, title: str, log_y: bool = False
) -> go.Figure:
    """Box plot of a numeric signal grouped by category."""
    fig = go.Figure()
    for cat in CLASS_ORDER:
        subset = df[df["label"] == cat][signal].dropna()
        fig.add_trace(
            go.Box(
                y=subset,
                name=cat,
                marker_color=CLASS_COLORS[cat],
                boxmean="sd",
                hoverinfo="y+name",
            )
        )

    yaxis_type = "log" if log_y else "linear"
    fig.update_layout(
        title=title,
        yaxis_title=signal.replace("_", " ").title(),
        yaxis_type=yaxis_type,
        width=750,
        height=420,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap of numeric signals."""
    numeric_cols = [
        "stars",
        "forks",
        "contributors_count",
        "commits_6m",
        "releases_count",
        "cicd_workflows",
        "test_files",
        "readme_length",
        "dependency_lines",
        "age_days",
    ]
    corr = df[numeric_cols].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=numeric_cols,
            y=numeric_cols,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale="RdBu_r",
            zmid=0,
            showscale=True,
            colorbar={"title": "Pearson r"},
        )
    )
    fig.update_layout(
        title="Signal Correlation Matrix",
        width=680,
        height=600,
        margin={"l": 10, "r": 10, "t": 50, "b": 120},
        xaxis={"tickangle": 35},
    )
    return fig


def plot_category_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of category distribution with percentages."""
    counts = df["label"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    total = counts.sum()
    pcts = [f"{c} ({c/total:.1%})" for c in counts.values]

    fig = go.Figure(
        data=go.Bar(
            x=CLASS_ORDER,
            y=counts.values,
            marker_color=[CLASS_COLORS[c] for c in CLASS_ORDER],
            text=pcts,
            textposition="outside",
            hovertemplate="%{x}: %{y} repos<extra></extra>",
        )
    )
    fig.update_layout(
        title="Label Distribution",
        yaxis_title="Number of Repositories",
        width=600,
        height=400,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        showlegend=False,
    )
    return fig


def plot_v1_v2_distribution_comparison(df_merged: pd.DataFrame) -> go.Figure:
    """Side-by-side bar chart comparing V1 and V2 label distributions."""
    v1_counts = df_merged["label_v1"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    v2_counts = df_merged["label_v2"].value_counts().reindex(CLASS_ORDER, fill_value=0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Baseline (V1)",
            x=CLASS_ORDER,
            y=v1_counts.values,
            marker_color="#4C72B0",
            text=v1_counts.values,
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Alternative (V2)",
            x=CLASS_ORDER,
            y=v2_counts.values,
            marker_color="#DD8452",
            text=v2_counts.values,
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Label Distribution: Baseline vs Alternative (All 771 Repos)",
        yaxis_title="Number of Repositories",
        width=750,
        height=420,
        barmode="group",
        bargap=0.18,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def plot_cicd_by_category(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing proportion of repos with CI/CD per category."""
    cicd_pct = (
        df.groupby("label", observed=True)
        .apply(lambda g: (g["cicd_workflows"] > 0).mean(), include_groups=False)
        .reindex(CLASS_ORDER, fill_value=0)
    )

    fig = go.Figure(
        data=go.Bar(
            x=CLASS_ORDER,
            y=[v * 100 for v in cicd_pct.values],
            marker_color=[CLASS_COLORS[c] for c in CLASS_ORDER],
            text=[f"{v:.0f}%" for v in cicd_pct.values * 100],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="CI/CD Adoption by Category (% of repos with 1+ workflows)",
        yaxis_title="% with CI/CD",
        yaxis={"range": [0, 105]},
        width=600,
        height=400,
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def plot_label_migration_sankey(df_merged: pd.DataFrame) -> go.Figure:
    """Sankey diagram showing label migrations from V1 to V2."""
    # Count transitions
    transitions = (
        df_merged.groupby(["label_v1", "label_v2"], observed=True)
        .size()
        .reset_index(name="count")
    )
    # Filter only migrations (V1 != V2) for clarity, keep all for completeness
    # Use all transitions for Sankey

    all_labels = CLASS_ORDER
    # Create indices
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    source_indices = []
    target_indices = []
    values = []

    for _, row in transitions.iterrows():
        source_indices.append(label_to_idx[row["label_v1"]])
        target_indices.append(label_to_idx[row["label_v2"]] + len(all_labels))
        values.append(row["count"])

    fig = go.Figure(
        data=go.Sankey(
            node={
                "label": all_labels + all_labels,
                "color": [CLASS_COLORS[c] for c in all_labels]
                + [CLASS_COLORS[c] for c in all_labels],
                "pad": 15,
                "thickness": 20,
            },
            link={
                "source": source_indices,
                "target": target_indices,
                "value": values,
                "color": [
                    f"rgba(128,128,128,{min(0.5, v/100)})" for v in values
                ],
            },
        )
    )
    fig.update_layout(
        title="Label Migration Flow: V1 → V2 (thicker = more repos)",
        width=800,
        height=500,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


# ======================================================================
# TAB 1 — PROBLEM & METHODOLOGY
# ======================================================================

def render_tab1() -> None:
    st.markdown('<p class="main-header">Problem &amp; Methodology</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">'
        "How we frame repository maturity classification as a weak-supervision NLP problem"
        "</p>",
        unsafe_allow_html=True,
    )

    # -- Project Overview --------------------------------------------------
    st.markdown('<p class="section-title">Project Overview</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            """
        **Repository maturity classification** sits at the intersection of natural language
        processing, software engineering analytics, and hiring intelligence. The core task:
        given a GitHub repository's observable signals — its README, commit history, CI/CD
        configuration, dependency graph, and contribution patterns — can an automated system
        reliably estimate the seniority level of the engineers who maintain it?

        This question matters for several practical reasons:
        - **Recruiters** could use repository maturity signals to identify engineers whose
          public work reflects a given experience level, complementing resume-based screening
          with behavioral evidence.
        - **Engineering managers** evaluating open-source dependencies could assess whether a
          library is maintained with professional-grade practices or represents a personal
          side project.
        - **Technical interview pipelines** could incorporate repository evidence as one
          signal among many when calibrating candidate expectations.
        """
        )
    with col2:
        st.metric("Repositories", "771", "from 2 sampling rounds")
        st.metric("Categories", "6", "maturity levels")
        st.metric("Signals per repo", "10", "engineering signals")

    st.markdown(
        """
    <div class="insight-box">
    <strong>Key insight:</strong> This project approaches the problem through
    <strong>weak supervision</strong> — rather than requiring costly human-annotated labels
    for hundreds of repositories, we use a large language model (DeepSeek) to generate
    training labels through carefully designed prompts, then fine-tune a lightweight
    classifier (DistilBERT, 66M params) on the resulting weakly-labeled dataset.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # -- Ethical note -------------------------------------------------------
    with st.expander("Ethical Considerations", expanded=False):
        st.markdown(
            """
        **Any automated maturity classification system must be understood as a proxy signal**
        — noisy, partial, and potentially biased — never as a replacement for human evaluation.

        - GitHub repositories are curated public personas, not comprehensive portraits of
          engineering competence.
        - An engineer may contribute to sophisticated internal systems invisible on their
          public profile.
        - A repository with extensive CI/CD and documentation may reflect organizational
          scaffolding rather than individual expertise.
        - The features extracted (stars, forks, contributors, commits, CI/CD presence, tests,
          releases, documentation, dependencies, age) are all structural proxies that can be
          gamed, misinterpreted, or confounded by organizational context.
        """
        )

    # -- Repository Collection & Sampling -----------------------------------
    st.markdown(
        '<p class="section-title">Repository Collection &amp; Sampling</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
        #### Two-Round Informed Sampling

        **Round 1** (469 repos): Broad queries spanning diverse star ranges and activity
        profiles on GitHub, capturing repositories from nascent personal projects to
        well-established community tools.

        **Round 2** (302 repos): Targeted follow-up to fill structural gaps, particularly in
        the mid-range of repository activity and underrepresented contributor-count segments.
        """
        )
    with col2:
        st.markdown(
            """
        #### Design Principles

        - **Structural, never label-driven**: Search queries targeted observable repository
          characteristics (star ranges, activity levels, age, contributor counts) — never
          maturity category names.
        - **Quality filters**: size > 5KB, README exists, not archived. These are NOT maturity
          filters — they ensure a minimum baseline of inspectable content.
        - **Avoiding popularity bias**: Stratified design ensured coverage across the full
          range of repository sizes, ages, and activity levels.
        """
        )

    st.markdown(
        """
    <div class="insight-box">
    <strong>Sampling limitation:</strong> Quality filters systematically exclude abandoned
    repos without documentation and very early-stage projects with minimal code. The dataset
    likely underrepresents truly low-value repos and overrepresents repos with at least
    <em>some</em> engineering structure. The classifier thus faces a harder challenge
    distinguishing adjacent maturity levels than identifying extremes.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # -- Repository Signals -------------------------------------------------
    st.markdown('<p class="section-title">Repository Signals</p>', unsafe_allow_html=True)
    st.markdown(
        "Ten engineering signals were extracted per repository, chosen to capture "
        "**orthogonal dimensions** of software engineering maturity:"
    )

    signals_data = [
        {"Signal": "Stars", "Dimension": "Community interest", "Why it matters": "Proxy for community validation and project visibility"},
        {"Signal": "Forks", "Dimension": "Derivative usage", "Why it matters": "Indicates whether others build upon the codebase"},
        {"Signal": "Contributors", "Dimension": "Team size / collaboration", "Why it matters": "Larger teams suggest more structured coordination"},
        {"Signal": "Commits (6m)", "Dimension": "Development velocity", "Why it matters": "Active development signals sustained engineering effort"},
        {"Signal": "Releases", "Dimension": "Release discipline", "Why it matters": "Versioned releases indicate professional delivery practices"},
        {"Signal": "CI/CD workflows", "Dimension": "Automation maturity", "Why it matters": "Automated pipelines are a hallmark of professional engineering"},
        {"Signal": "Test files", "Dimension": "Testing culture", "Why it matters": "Presence of tests reflects quality assurance practices"},
        {"Signal": "README length", "Dimension": "Documentation quality", "Why it matters": "Comprehensive docs indicate user-facing maturity"},
        {"Signal": "Dependencies", "Dimension": "Ecosystem integration", "Why it matters": "More dependencies suggest real-world library usage"},
        {"Signal": "Age (days)", "Dimension": "Longevity / sustainability", "Why it matters": "Older repos with sustained activity signal maturity"},
    ]
    st.dataframe(
        pd.DataFrame(signals_data),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
    <div class="insight-box">
    <strong>Critical design choice:</strong> The text representation used for LLM labeling
    and BERT fine-tuning <strong>excludes</strong> popularity-correlated metadata (stars,
    forks, owner name, repository topics). This prevents the classifier from learning trivial
    shortcuts (e.g., "repositories with &gt;500 stars are always senior") and forces it to
    attend to genuine engineering structure.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # -- Weak Labeling Strategy ---------------------------------------------
    st.markdown(
        '<p class="section-title">Weak Labeling Strategy</p>', unsafe_allow_html=True
    )

    st.markdown(
        """
    The weak-supervision pipeline uses **DeepSeek** (a large language model) to assign each
    repository a maturity label based on its structured `text_repr` summary. The LLM receives
    a prompt defining six categories and reasons about each repository's engineering signals
    before producing a **(label, confidence, justification)** triple. Temperature is set to 0
    for deterministic labeling, with checkpoints every 50 repos.
    """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Baseline Approach (V1)")
        st.markdown(
            """
        **Qualitative category definitions** that described maturity levels in terms of
        engineering *character* rather than specific, observable criteria.

        - "Senior" = "demonstrating established engineering practices with consistent activity"
        - "Lead" = "showing architectural leadership and sophisticated automation"

        These definitions left substantial room for interpretation at category boundaries,
        introducing structured label noise that the downstream classifier inherited.
        """
        )
    with col2:
        st.markdown("#### Alternative Approach (V2)")
        st.markdown(
            """
        **Operationalized criteria** grounded in observable engineering signals:

        - "Senior" = *has CI/CD configured, test files present, multiple contributors with
          consistent commits, documented releases, README > 500 chars*
        - "Lead" = *10+ contributors, multiple CI/CD workflows, extensive test suites,
          semver releases, dependency graph with 20+ packages, comprehensive docs*

        The key principle: **reducing annotator degrees of freedom** by defining every
        category in terms of concrete, verifiable repository properties.
        """
        )

    st.markdown(
        """
    <div class="insight-box">
    <strong>Controlled experiment:</strong> The only difference between V1 and V2 is the
    prompt wording. Dataset, features, text representation, train/test split, model
    architecture, and hyperparameters are all held constant. Any performance difference
    isolates the causal effect of <strong>label operationalization quality</strong>.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # -- Category definitions table -----------------------------------------
    with st.expander("Full Category Definitions", expanded=False):
        cat_data = [
            {"Category": c, "ID": LABEL_TO_ID[c], "Description": CLASS_DESCRIPTIONS[c]}
            for c in CLASS_ORDER
        ]
        st.dataframe(
            pd.DataFrame(cat_data), use_container_width=True, hide_index=True
        )

    # -- Limitations --------------------------------------------------------
    st.markdown('<p class="section-title">Limitations</p>', unsafe_allow_html=True)
    limitations = [
        (
            "Weak Supervision Without Ground Truth",
            "All labels are generated by an LLM whose accuracy on this task is unknown. "
            "The classifier learns to approximate the LLM's judgment, not necessarily "
            "true repository maturity. Without human-labeled validation data, we cannot "
            "measure absolute label quality.",
        ),
        (
            "Proxy Feature Limitations",
            "The 10 signals are structural proxies, not direct measurements. Code quality, "
            "architectural decisions, and testing thoroughness are only partially reflected "
            "in counts of files, contributors, and commits.",
        ),
        (
            "Sampling Bias",
            "Quality filters (size > 5KB, README required) systematically exclude repos at "
            "the lowest end of the maturity spectrum, biasing the classifier toward "
            "overestimating maturity near the low-value/intern boundary.",
        ),
        (
            "Ordinal Task Treated as Nominal",
            "Maturity labels form an ordinal scale (intern < junior < senior < lead) but "
            "the classifier uses flat cross-entropy loss, which treats all misclassifications "
            "as equally costly.",
        ),
        (
            "Single Model, Single Seed",
            "Both V1 and V2 results come from a single DistilBERT training run with seed=42. "
            "Without multiple restarts, reported metrics are point estimates with unknown "
            "variance.",
        ),
    ]
    for title, body in limitations:
        with st.expander(title, expanded=False):
            st.markdown(body)


# ======================================================================
# TAB 2 — EXPLORATORY ANALYSIS
# ======================================================================

def render_tab2() -> None:
    st.markdown(
        '<p class="main-header">Exploratory Analysis</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">'
        "Understanding the dataset: distributions, feature relationships, "
        "and structural patterns across maturity categories"
        "</p>",
        unsafe_allow_html=True,
    )

    df = load_labeled_data("v1")
    df_merged = load_merged_labels()

    # -- Dataset Overview ---------------------------------------------------
    st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Repositories", f"{len(df):,}")
    with col2:
        st.metric("Languages", f"{df['language'].nunique()}")
    with col3:
        st.metric("Avg. README Length", f"{df['readme_length'].mean():.0f} chars")
    with col4:
        st.metric("Avg. Repo Age", f"{df['age_days'].mean():.0f} days")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Repos w/ CI/CD", f"{(df['cicd_workflows'] > 0).mean():.0%}")
    with col2:
        st.metric("Repos w/ Tests", f"{(df['test_files'] > 0).mean():.0%}")
    with col3:
        st.metric("Repos w/ Releases", f"{(df['releases_count'] > 0).mean():.0%}")
    with col4:
        st.metric("Repos w/ License", f"{df['has_license'].mean():.0%}")

    # -- Category Distribution ----------------------------------------------
    st.markdown('<p class="section-title">Category Distribution</p>', unsafe_allow_html=True)
    st.markdown(
        "The label distribution reveals the class imbalance inherent in the weak-supervision "
        "process. Understanding this distribution is essential because it directly affects "
        "model training: minority classes (template, junior) have fewer examples for the "
        "classifier to learn from, and class-weighted loss can only partially compensate."
    )

    tab_v1, tab_v2, tab_compare = st.tabs(["V1 Distribution", "V2 Distribution", "V1 vs V2"])

    with tab_v1:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(plot_category_distribution(df), use_container_width=True)
        with col2:
            st.markdown("##### V1 Label Distribution")
            v1_counts = df["label"].value_counts().reindex(CLASS_ORDER, fill_value=0)
            for cat in CLASS_ORDER:
                st.metric(
                    f"{cat}",
                    f"{v1_counts[cat]}",
                    f"{v1_counts[cat]/len(df):.1%}",
                )
            st.markdown(
                """
            <div class="insight-box">
            <strong>Why this matters:</strong> Senior dominates (29.4%), template is rare
            (8.2%). Imbalance ratio is ~3.6:1. The LLM's tendency to assign senior as a
            default for repos with moderate-but-not-exceptional signals creates a label
            distribution that partially reflects annotator central tendency.
            </div>
            """,
                unsafe_allow_html=True,
            )

    with tab_v2:
        df_v2 = load_labeled_data("v2")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(plot_category_distribution(df_v2), use_container_width=True)
        with col2:
            st.markdown("##### V2 Label Distribution")
            v2_counts = df_v2["label"].value_counts().reindex(CLASS_ORDER, fill_value=0)
            for cat in CLASS_ORDER:
                st.metric(
                    f"{cat}",
                    f"{v2_counts[cat]}",
                    f"{v2_counts[cat]/len(df_v2):.1%}",
                )
            st.markdown(
                """
            <div class="insight-box">
            <strong>Key shift:</strong> Low-value collapses from 140 → 63 repos (-55%).
            Intern grows from 105 → 163 (+55%). This reflects the operationalized criteria's
            effect: repos previously labeled low-value due to qualitative "lack of
            sophistication" were reclassified as intern when stricter criteria revealed
            genuine (if early-stage) engineering activity.
            </div>
            """,
                unsafe_allow_html=True,
            )

    with tab_compare:
        st.plotly_chart(
            plot_v1_v2_distribution_comparison(df_merged), use_container_width=True
        )
        # Label agreement stats
        agreement = df_merged["labels_agree"].mean()
        changed = len(df_merged[~df_merged["labels_agree"]])
        st.markdown(
            f"""
        <div class="insight-box">
        <strong>Label stability:</strong> V1/V2 agreement is <strong>{agreement:.1%}</strong>.
        <strong>{changed} repos ({1-agreement:.1%})</strong> changed labels between approaches.
        The largest migrations: junior→intern (37), low-value→senior (33), senior→lead (26),
        and low-value→intern (21). This demonstrates how sensitive weak labels are to prompt
        operationalization — a finding with implications for any LLM-based labeling pipeline.
        </div>
        """,
            unsafe_allow_html=True,
        )

    # -- Signal Distributions by Category -----------------------------------
    st.markdown(
        '<p class="section-title">Signal Distributions by Category</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "These visualizations reveal how engineering signals vary across maturity categories. "
        "Each chart tells part of the story: which signals discriminate between categories, "
        "where overlap creates confusion, and how operationalized definitions reshape the "
        "signal distributions."
    )

    # Signal selector
    signal_options = [
        "commits_6m",
        "contributors_count",
        "releases_count",
        "cicd_workflows",
        "test_files",
        "readme_length",
        "dependency_lines",
        "age_days",
        "stars",
        "forks",
    ]
    selected_signal = st.selectbox(
        "Select a signal to explore by category:",
        signal_options,
        format_func=lambda s: s.replace("_", " ").title(),
        key="signal_selector",
    )

    log_scale = selected_signal in ("stars", "forks", "commits_6m", "releases_count")
    st.plotly_chart(
        plot_signal_by_category(
            df, selected_signal,
            f"{selected_signal.replace('_', ' ').title()} by Maturity Category",
            log_y=log_scale,
        ),
        use_container_width=True,
    )

    # Interpretive text per signal
    signal_insights = {
        "commits_6m": "Commit activity generally increases with maturity, but the variance "
        "is substantial. Lead repos often show sustained high-frequency commits, while "
        "template repos have near-zero recent commits (structural deadness).",
        "contributors_count": "Lead repos have significantly larger contributor bases — a "
        "direct signal of architectural leadership. Intern and template repos are typically "
        "single-contributor projects.",
        "releases_count": "Release discipline is one of the strongest discriminators. Lead "
        "repos consistently show multiple versioned releases, while low-value repos have "
        "none. This signal heavily influenced the operationalized V2 definitions.",
        "cicd_workflows": "CI/CD presence is almost binary: lead/senior repos typically "
        "have 1+ workflows, while intern/low-value repos rarely do. The presence/absence "
        "of CI/CD is one of the few signals that cleanly separates professional from "
        "non-professional engineering.",
        "test_files": "Test file counts rise with maturity. The gap between senior and "
        "lead is narrower than expected — both categories have repos with extensive test "
        "suites, contributing to the senior↔lead confusion in the baseline model.",
        "readme_length": "Documentation quality (proxied by README length) shows a clear "
        "maturity gradient. Template repos often have long READMEs (setup instructions) "
        "but little else, making this signal potentially misleading in isolation.",
        "dependency_lines": "Lead repos have substantially more dependencies, reflecting "
        "real-world library integration. Low-value repos often have zero dependencies — "
        "a strong negative signal that the V2 operationalized definitions captured explicitly.",
        "age_days": "Older repos aren't necessarily more mature. Template repos can be old "
        "(mature templates), and intern repos can be old (inactive learning projects). "
        "Age alone is a weak maturity signal.",
    }

    if selected_signal in signal_insights:
        st.markdown(
            f"""
        <div class="insight-box">
        <strong>Why this visualization matters:</strong> {signal_insights[selected_signal]}
        </div>
        """,
            unsafe_allow_html=True,
        )

    # -- CI/CD Adoption by Category -----------------------------------------
    st.markdown('<p class="section-title">Key Patterns at a Glance</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_cicd_by_category(df), use_container_width=True)
        st.markdown(
            """
        <div class="insight-box">
        <strong>CI/CD adoption follows a maturity gradient:</strong> nearly 80% of lead
        repos have CI/CD vs &lt;10% of low-value repos. This signal is one of the cleanest
        discriminators between professional and non-professional engineering — and was
        central to the V2 operationalized definitions.
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
        st.markdown(
            """
        <div class="insight-box">
        <strong>Signal correlations reveal redundancies:</strong> stars and forks are highly
        correlated (r=0.78). CI/CD workflows correlate with test files (r=0.52) and
        contributors (r=0.41). These correlations mean that some signals provide overlapping
        information — which is why the text representation (not raw numbers) is used for
        classification.
        </div>
        """,
            unsafe_allow_html=True,
        )

    # -- Label Migration Visualization --------------------------------------
    st.markdown('<p class="section-title">Label Migration: V1 → V2</p>', unsafe_allow_html=True)
    st.markdown(
        "The Sankey diagram below shows how individual repositories changed labels between "
        "the baseline and alternative approaches. Thicker bands represent larger migration "
        "flows. This visualization is central to understanding the sensitivity analysis — "
        "it shows that the prompt change affected NOT just a few borderline repos, but "
        "fundamentally reorganized a quarter of the dataset."
    )
    st.plotly_chart(plot_label_migration_sankey(df_merged), use_container_width=True)

    # Migration stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Top Migration Flows")
        migrations = (
            df_merged.groupby(["label_v1", "label_v2"], observed=True)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(8)
        )
        for _, row in migrations.iterrows():
            st.markdown(
                f"- **{row['label_v1']}** → **{row['label_v2']}**: {row['count']} repos"
            )
    with col2:
        st.markdown(
            f"""
        ##### Summary

        - **Agreement rate:** {df_merged['labels_agree'].mean():.1%}
        - **Repos that changed:** {len(df_merged[~df_merged['labels_agree']])} / {len(df_merged)}
        - **Stayed intern:** {len(df_merged[(df_merged['label_v1'] == 'intern') & (df_merged['label_v2'] == 'intern')])}
        - **Stayed senior:** {len(df_merged[(df_merged['label_v1'] == 'senior') & (df_merged['label_v2'] == 'senior')])}
        - **Stayed lead:** {len(df_merged[(df_merged['label_v1'] == 'lead') & (df_merged['label_v2'] == 'lead')])}

        The 25.2% label change rate confirms that **weak-label operationalization is a
        first-order determinant of label quality** — a finding that has implications for
        any project relying on LLM-generated training data.
        """
        )


# ======================================================================
# TAB 3 — MODEL RESULTS
# ======================================================================

def render_tab3() -> None:
    st.markdown('<p class="main-header">Model Results</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">'
        "Controlled sensitivity analysis: how much does label operationalization quality "
        "matter for downstream classifier performance?"
        "</p>",
        unsafe_allow_html=True,
    )

    # Load data
    test_v1 = load_test_predictions("v1")
    test_v2 = load_test_predictions("v2")
    metrics_v1 = compute_metrics(test_v1)
    metrics_v2 = compute_metrics(test_v2)

    # -- Key finding callout ------------------------------------------------
    st.markdown(
        f"""
    <div class="insight-box" style="border-left-color: #D62728;">
    <strong>Core methodological finding:</strong> Changing only the weak-labeling prompt
    — replacing qualitative category descriptions with operationalized criteria — improved
    test accuracy from <strong>{metrics_v1['accuracy']:.1%}</strong> to
    <strong>{metrics_v2['accuracy']:.1%}</strong>
    (+{(metrics_v2['accuracy'] - metrics_v1['accuracy'])*100:.1f} pp, a
    {(metrics_v2['accuracy']/metrics_v1['accuracy'] - 1)*100:.0f}% relative improvement).
    Dataset, features, model architecture, and hyperparameters were all held constant.
    <strong>Semantic clarity in the labeling function mattered more than any architectural
    or representational change could have.</strong>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # -- Aggregate Metrics Comparison ---------------------------------------
    st.markdown('<p class="section-title">Aggregate Metrics Comparison</p>', unsafe_allow_html=True)

    metrics_comparison = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1",
                "Weighted F1",
            ],
            "Baseline (V1)": [
                metrics_v1["accuracy"],
                metrics_v1["macro_precision"],
                metrics_v1["macro_recall"],
                metrics_v1["macro_f1"],
                metrics_v1["weighted_f1"],
            ],
            "Alternative (V2)": [
                metrics_v2["accuracy"],
                metrics_v2["macro_precision"],
                metrics_v2["macro_recall"],
                metrics_v2["macro_f1"],
                metrics_v2["weighted_f1"],
            ],
            "Delta": [
                metrics_v2["accuracy"] - metrics_v1["accuracy"],
                metrics_v2["macro_precision"] - metrics_v1["macro_precision"],
                metrics_v2["macro_recall"] - metrics_v1["macro_recall"],
                metrics_v2["macro_f1"] - metrics_v1["macro_f1"],
                metrics_v2["weighted_f1"] - metrics_v1["weighted_f1"],
            ],
        }
    )

    # Format display
    metrics_display = metrics_comparison.copy()
    for col in ["Baseline (V1)", "Alternative (V2)"]:
        metrics_display[col] = metrics_display[col].apply(lambda x: f"{x:.4f}")
    metrics_display["Delta"] = metrics_display["Delta"].apply(
        lambda x: f"+{x:.4f}" if x >= 0 else f"{x:.4f}"
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            metrics_display, use_container_width=True, hide_index=True
        )
    with col2:
        st.plotly_chart(plot_metric_delta_chart(metrics_v1, metrics_v2), use_container_width=True)

    st.markdown(
        """
    <div class="insight-box">
    <strong>Interpretation:</strong> The alternative approach improved every aggregate metric
    by 17–23 percentage points. Weighted F1 improved more than macro F1 (+23.4 vs +19.5 pp),
    implying that V2 labels particularly benefited the higher-support classes (senior, intern)
    while still improving minority classes. This demonstrates that the improvement is genuine
    and distributed — not an artifact of a single class.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # -- Per-Class Performance ----------------------------------------------
    st.markdown('<p class="section-title">Per-Class Performance</p>', unsafe_allow_html=True)

    st.plotly_chart(
        plot_per_class_f1_comparison(metrics_v1, metrics_v2), use_container_width=True
    )

    # Build detailed per-class comparison table
    per_class_data = []
    for cat in CLASS_ORDER:
        v1_p = metrics_v1["per_class"][cat]
        v2_p = metrics_v2["per_class"][cat]
        per_class_data.append(
            {
                "Class": cat,
                "V1 Precision": f"{v1_p['precision']:.3f}",
                "V2 Precision": f"{v2_p['precision']:.3f}",
                "V1 Recall": f"{v1_p['recall']:.3f}",
                "V2 Recall": f"{v2_p['recall']:.3f}",
                "V1 F1": f"{v1_p['f1-score']:.3f}",
                "V2 F1": f"{v2_p['f1-score']:.3f}",
                "Delta F1": f"{v2_p['f1-score'] - v1_p['f1-score']:+.3f}",
                "Support (V2)": int(v2_p["support"]),
            }
        )

    st.dataframe(
        pd.DataFrame(per_class_data), use_container_width=True, hide_index=True
    )

    # Per-class insights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
        ##### Largest Improvements

        - **Senior:** F1 0.275 → 0.600 (+0.325). The operationalized prompt successfully
          disambiguated senior from lead — the primary failure mode in V1.
        - **Low-value:** F1 0.242 → 0.583 (+0.341). Stricter absence-based criteria
          produced cleaner low-value labels the classifier could learn.
        - **Template:** F1 0.500 → 0.818 (+0.318). Template repos have a distinctive
          structural signature that operationalized definitions capture effectively.
        """
        )
    with col2:
        st.markdown(
            """
        ##### Stable Performance

        - **Lead:** F1 0.679 → 0.694 (+0.015). Lead-level repos are intrinsically
          distinguishable regardless of labeling philosophy. Operationalized criteria
          reduced over-prediction without sacrificing recall.
        """
        )
    with col3:
        st.markdown(
            """
        ##### Persistent Difficulty

        - **Junior:** F1 0.286 → 0.222 (-0.064). Now the worst-performing class.
          Junior occupies the space between "has some professional practices" (intern)
          and "has established practices" (senior) — the available signals simply don't
          carry enough information to reliably place a repo at this specific intermediate
          point. Only 66 training examples in V2 (8.6% of dataset).
        """
        )

    # -- Confusion Matrices -------------------------------------------------
    st.markdown('<p class="section-title">Confusion Matrix Analysis</p>', unsafe_allow_html=True)

    tab_v1, tab_v2, tab_error = st.tabs(["V1 Confusion Matrix", "V2 Confusion Matrix", "Error Analysis"])

    with tab_v1:
        st.markdown(
            "**Baseline (V1)** — three confusion patterns dominate: senior→lead "
            "(15 cases, 44% of true senior), intern→junior (6 cases), and "
            "low-value→intern (6 cases). Adjacent-class errors = 47.8% of total."
        )
        st.plotly_chart(
            plot_confusion_heatmap(
                metrics_v1["confusion_matrix_normalized"],
                "Confusion Matrix — Baseline (V1)",
            ),
            use_container_width=True,
        )

    with tab_v2:
        st.markdown(
            "**Alternative (V2)** — senior→lead confusion cut nearly in half (15→8). "
            "Low-value recall dramatically improved (19%→78%). Template near-perfect "
            "(9/11 correct). A new error emerges: intern→low-value (6 cases)."
        )
        st.plotly_chart(
            plot_confusion_heatmap(
                metrics_v2["confusion_matrix_normalized"],
                "Confusion Matrix — Alternative (V2)",
            ),
            use_container_width=True,
        )

    with tab_error:
        st.markdown("##### Top Confusion Pairs: V1 vs V2")

        def get_top_errors(test_df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
            errors = test_df[~test_df["correct"]].copy()
            errors["confusion_pair"] = (
                errors["label"].astype(str) + " → " + errors["predicted_label"].astype(str)
            )
            error_counts = (
                errors.groupby("confusion_pair").size().sort_values(ascending=False).head(top_n)
            )
            return error_counts.reset_index(name="count")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**V1 Top Errors**")
            v1_errors = get_top_errors(test_v1)
            st.dataframe(v1_errors, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**V2 Top Errors**")
            v2_errors = get_top_errors(test_v2)
            st.dataframe(v2_errors, use_container_width=True, hide_index=True)

        st.markdown(
            """
        <div class="insight-box">
        <strong>Error pattern shift:</strong> V1 errors were dominated by
        <em>overestimation</em> (senior→lead, intern→junior, low-value→intern). V2 errors
        show a more balanced pattern with some <em>underestimation</em> (intern→low-value).
        The operationalized criteria created sharper boundaries that some borderline repos
        fall on the wrong side of — but the overall error rate is substantially lower.
        </div>
        """,
            unsafe_allow_html=True,
        )

    # -- Sensitivity Analysis -----------------------------------------------
    st.markdown('<p class="section-title">Sensitivity Analysis</p>', unsafe_allow_html=True)

    st.markdown(
        """
    This project is fundamentally a **controlled methodological sensitivity analysis**
    with a single modified variable:

    | Component | Status |
    |-----------|--------|
    | Repository sampling (two rounds, 771 repos) | **Fixed** |
    | Extracted features (10 signals) | **Fixed** |
    | Text representation format (`text_repr`) | **Fixed** |
    | Train/val/test split (539/115/116, seed=42) | **Fixed** |
    | Model architecture (DistilBERT, 66M params) | **Fixed** |
    | Training hyperparameters (batch=16, epochs=8, lr=2e-5) | **Fixed** |
    | Weak-label prompt and category definitions | **Modified** |
    """
    )

    st.markdown(
        f"""
    <div class="insight-box" style="border-left-color: #1F77B4;">
    <strong>Conclusion:</strong> The {(metrics_v2['accuracy']/metrics_v1['accuracy'] - 1)*100:.0f}%
    relative accuracy improvement from a single-variable change — the wording of the labeling
    prompt — demonstrates that <strong>downstream classifier performance is highly sensitive
    to weak-label operationalization quality.</strong> When the LLM annotator operates with
    qualitative, subjective category definitions, the resulting label noise propagates into
    the training data and becomes the dominant source of classifier error.

    This finding challenges the common assumption that model capacity, feature engineering,
    or training-data volume are the primary levers for improving downstream performance.
    In this setting, <strong>semantic clarity in the labeling function mattered more than
    any architectural or representational change could have.</strong>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # -- Junior analysis ----------------------------------------------------
    with st.expander("Why Junior Underperforms in Both Approaches", expanded=False):
        st.markdown(
            """
        The persistent difficulty of junior classification across both V1 and V2
        suggests a deeper problem: **"junior engineer" may not correspond to a distinct
        structural signature in repository data.**

        - Junior is an **interstitial category** squeezed between intern and senior with
          no distinctive structural signature of its own.
        - A repository with some tests and basic CI/CD could be a strong intern project,
          a typical junior project, or a modest senior project — the structural signals
          are identical.
        - Only the *quality* and *consistency* of execution would disambiguate them, which
          the current 10 signals do not capture.
        - In V2, the junior class was reduced to only 66 training examples (8.6% of the
          dataset), with only 10 in the test set — insufficient for the classifier to
          learn a coherent junior prototype.

        **Implication:** Richer repository representations (potentially incorporating
        code-level features beyond structural metadata) may be necessary to resolve the
        most subtle maturity distinctions.
        """
        )


# ======================================================================
# TAB 4 — INTERACTIVE REPOSITORY EXPLORATION
# ======================================================================

def render_tab4() -> None:
    st.markdown(
        '<p class="main-header">Interactive Repository Exploration</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">'
        "Search, filter, and inspect individual repositories — explore signals, "
        "labels, predictions, and how V1 vs V2 approaches classified each project"
        "</p>",
        unsafe_allow_html=True,
    )

    # Load data
    df_full = load_full_repo_data()
    test_v1 = load_test_predictions("v1")
    test_v2 = load_test_predictions("v2")

    # -- Filters ------------------------------------------------------------
    st.markdown("### Filters")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        search_term = st.text_input(
            "Search repository name", "", placeholder="e.g., tensorflow"
        )
    with col2:
        all_languages = sorted(df_full["language"].dropna().unique())
        selected_languages = st.multiselect(
            "Language", all_languages, default=[]
        )
    with col3:
        selected_categories = st.multiselect(
            "V1 Label", CLASS_ORDER, default=[]
        )
    with col4:
        selected_categories_v2 = st.multiselect(
            "V2 Label", CLASS_ORDER, default=[]
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        min_stars = st.number_input("Min stars", 0, value=0)
    with col2:
        min_contributors = st.number_input("Min contributors", 0, value=0)
    with col3:
        min_commits = st.number_input("Min commits (6m)", 0, value=0)

    # -- Filter logic -------------------------------------------------------
    filtered = df_full.copy()

    if search_term:
        filtered = filtered[
            filtered["full_name"].str.contains(search_term, case=False, na=False)
        ]
    if selected_languages:
        filtered = filtered[filtered["language"].isin(selected_languages)]
    if selected_categories:
        filtered = filtered[filtered["label"].isin(selected_categories)]
    if selected_categories_v2 and "label_v2" in filtered.columns:
        filtered = filtered[filtered["label_v2"].isin(selected_categories_v2)]
    filtered = filtered[filtered["stars"] >= min_stars]
    filtered = filtered[filtered["contributors_count"] >= min_contributors]
    filtered = filtered[filtered["commits_6m"] >= min_commits]

    st.markdown(f"**{len(filtered)} repositories match your filters**")

    # Add prediction comparison for test-set repos
    test_repo_ids_v1 = set(test_v1["repo_id"].unique())
    test_repo_ids_v2 = set(test_v2["repo_id"].unique())
    filtered["in_test_v1"] = filtered["repo_id"].isin(test_repo_ids_v1)
    filtered["in_test_v2"] = filtered["repo_id"].isin(test_repo_ids_v2)

    # -- Results display ----------------------------------------------------
    if len(filtered) == 0:
        st.info("No repositories match the current filters. Try broadening your search.")
        return

    # Summary stats for filtered set
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Filtered Repos", len(filtered))
    with col2:
        st.metric("Avg Stars", f"{filtered['stars'].mean():.0f}")
    with col3:
        st.metric("Avg Contributors", f"{filtered['contributors_count'].mean():.1f}")
    with col4:
        st.metric("Avg Commits (6m)", f"{filtered['commits_6m'].mean():.0f}")

    st.markdown("---")

    # -- Paginated results --------------------------------------------------
    repos_per_page = 10
    total_pages = max(1, (len(filtered) + repos_per_page - 1) // repos_per_page)

    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        page = st.number_input(
            "Page", min_value=1, max_value=total_pages, value=1, label_visibility="collapsed"
        )
    with col2:
        st.markdown(
            f"<small>Page {page} of {total_pages} ({len(filtered)} repos total)</small>",
            unsafe_allow_html=True,
        )

    start_idx = (page - 1) * repos_per_page
    end_idx = min(start_idx + repos_per_page, len(filtered))
    page_repos = filtered.iloc[start_idx:end_idx]

    for _, repo in page_repos.iterrows():
        with st.expander(
            f"{repo['full_name']} | {repo.get('language', 'N/A')} | "
            f"V1: **{repo['label']}** | V2: **{repo.get('label_v2', 'N/A')}**",
            expanded=False,
        ):
            # Metadata row
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Stars", f"{int(repo['stars']):,}")
            with col2:
                st.metric("Forks", f"{int(repo['forks']):,}")
            with col3:
                st.metric("Contributors", int(repo["contributors_count"]))
            with col4:
                st.metric("Commits (6m)", int(repo["commits_6m"]))
            with col5:
                st.metric("Age (days)", int(repo["age_days"]))

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Releases", int(repo["releases_count"]))
            with col2:
                st.metric("CI/CD Workflows", int(repo["cicd_workflows"]))
            with col3:
                st.metric("Test Files", int(repo["test_files"]))
            with col4:
                st.metric("README Length", f"{int(repo['readme_length']):,} chars")

            # Label comparison
            st.markdown("##### Labels & Predictions")
            col1, col2, col3 = st.columns(3)
            with col1:
                v1_label = repo["label"]
                st.markdown(
                    f"**V1 Label:** <span style='color:{CLASS_COLORS[v1_label]}'>"
                    f"{v1_label}</span>",
                    unsafe_allow_html=True,
                )
                # Show V1 prediction if available
                v1_pred = test_v1[test_v1["repo_id"] == repo["repo_id"]]
                if len(v1_pred) > 0:
                    pred_label = v1_pred.iloc[0]["predicted_label"]
                    correct = v1_pred.iloc[0]["correct"]
                    icon = "" if correct else ""
                    st.markdown(
                        f"**V1 Prediction:** <span style='color:{CLASS_COLORS[pred_label]}'>"
                        f"{pred_label}</span> {icon}",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("*Not in test set*")
            with col2:
                v2_label = repo.get("label_v2", "N/A")
                if v2_label in CLASS_COLORS:
                    st.markdown(
                        f"**V2 Label:** <span style='color:{CLASS_COLORS[v2_label]}'>"
                        f"{v2_label}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"**V2 Label:** {v2_label}")
                # Show V2 prediction if available
                v2_pred = test_v2[test_v2["repo_id"] == repo["repo_id"]]
                if len(v2_pred) > 0:
                    pred_label = v2_pred.iloc[0]["predicted_label"]
                    correct = v2_pred.iloc[0]["correct"]
                    icon = "" if correct else ""
                    st.markdown(
                        f"**V2 Prediction:** <span style='color:{CLASS_COLORS[pred_label]}'>"
                        f"{pred_label}</span> {icon}",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("*Not in test set*")
            with col3:
                labels_agree = v1_label == v2_label if v2_label in CLASS_ORDER else None
                if labels_agree is True:
                    st.success("Labels agree")
                elif labels_agree is False:
                    st.warning(f"Label changed: {v1_label} → {v2_label}")
                else:
                    st.markdown("*Comparison N/A*")

            # Description & README preview
            if pd.notna(repo.get("description")):
                st.markdown(f"**Description:** {repo['description']}")

            # Show README preview
            if pd.notna(repo.get("readme_preview")) and str(repo["readme_preview"]).strip():
                with st.expander("README Preview", expanded=False):
                    st.text(str(repo["readme_preview"])[:1500])

    # -- Label Agreement Analysis -------------------------------------------
    st.markdown("---")
    st.markdown('<p class="section-title">Label Change Explorer</p>', unsafe_allow_html=True)

    st.markdown(
        "Explore which repositories changed labels between V1 and V2. "
        "This helps understand where the operationalized definitions had the "
        "largest impact on labeling decisions."
    )

    col1, col2 = st.columns(2)
    with col1:
        from_label = st.selectbox(
            "From (V1)", CLASS_ORDER, key="from_label_explorer"
        )
    with col2:
        to_label = st.selectbox(
            "To (V2)", CLASS_ORDER, key="to_label_explorer"
        )

    if from_label == to_label:
        st.info("Select different labels to see migrations between them.")
    else:
        changed_repos = df_full[
            (df_full["label"] == from_label) & (df_full["label_v2"] == to_label)
        ]
        st.markdown(
            f"**{len(changed_repos)} repos** migrated from **{from_label}** → **{to_label}**"
        )
        if len(changed_repos) > 0 and len(changed_repos) <= 20:
            for _, repo in changed_repos.iterrows():
                st.markdown(
                    f"- **{repo['full_name']}** ({repo.get('language', 'N/A')})"
                    f" | {int(repo['stars'])} stars"
                    f" | {int(repo['contributors_count'])} contributors"
                    f" | {int(repo['commits_6m'])} commits (6m)"
                )
        elif len(changed_repos) > 20:
            st.markdown(
                f"*Showing first 20 of {len(changed_repos)} repos...*"
            )
            for _, repo in changed_repos.head(20).iterrows():
                st.markdown(
                    f"- **{repo['full_name']}** ({repo.get('language', 'N/A')})"
                    f" | {int(repo['stars'])} stars"
                    f" | {int(repo['contributors_count'])} contributors"
                    f" | {int(repo['commits_6m'])} commits (6m)"
                )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    # -- header ------------------------------------------------------------
    st.markdown(
        '<p style="font-size:1.6rem; font-weight:700; margin-bottom:0;">'
        " Repository Maturity Classification</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.95rem; color:#666; margin-bottom:1.2rem;">'
        "Weak-supervision NLP pipeline: DeepSeek LLM labeling → DistilBERT fine-tuning → "
        "6-class engineering maturity estimation | "
        "<strong>Controlled sensitivity analysis: Baseline (V1) vs Operationalized (V2)</strong>"
        "</p>",
        unsafe_allow_html=True,
    )

    # -- 4 tabs ------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Problem & Methodology",
            "2. Exploratory Analysis",
            "3. Model Results",
            "4. Interactive Repository Exploration",
        ]
    )

    with tab1:
        render_tab1()
    with tab2:
        render_tab2()
    with tab3:
        render_tab3()
    with tab4:
        render_tab4()

    # -- footer ------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        "<small>GitHub Hiring Repository Intelligence — Track A | "
        "Weak Supervision + DistilBERT | May 2026</small>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
