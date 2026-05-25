"""
Stage 6 — Visualization.

Generates publication-quality figures for the evaluation report.
All charts saved to output/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import CLASS_ORDER, compute_metrics, load_data
from src.utils import load_dotenv

load_dotenv()

OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"

# -- style -----------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)
sns.set_palette("muted")


def confusion_heatmap(cm_norm: np.ndarray, save_path: Path) -> None:
    """Normalized confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    annot = np.array(
        [
            [f"{val:.0%}" if val > 0 else "" for val in row]
            for row in cm_norm
        ]
    )
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        xticklabels=CLASS_ORDER,
        yticklabels=CLASS_ORDER,
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Proportion of true class"},
        ax=ax,
    )
    ax.set_title("Confusion Matrix (Normalized by Row)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[OK] {save_path}")


def per_class_metrics_chart(metrics: dict, save_path: Path) -> None:
    """Grouped bar chart: precision, recall, f1 per class."""
    per_class = metrics["per_class"]
    x = np.arange(len(CLASS_ORDER))
    width = 0.25

    prec_vals = [per_class[c]["precision"] for c in CLASS_ORDER]
    rec_vals = [per_class[c]["recall"] for c in CLASS_ORDER]
    f1_vals = [per_class[c]["f1-score"] for c in CLASS_ORDER]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, prec_vals, width, label="Precision", color="#4C72B0")
    ax.bar(x, rec_vals, width, label="Recall", color="#55A868")
    ax.bar(x + width, f1_vals, width, label="F1", color="#C44E52")

    ax.axhline(y=metrics["accuracy"], color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(
        len(CLASS_ORDER) - 0.5,
        metrics["accuracy"] + 0.02,
        f"Accuracy={metrics['accuracy']:.2f}",
        fontsize=9,
        color="gray",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics vs Overall Accuracy")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[OK] {save_path}")


def class_distribution_chart(df: pd.DataFrame, save_path: Path) -> None:
    """Side-by-side bar chart of true vs predicted class distribution."""
    true_counts = df["label"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    pred_counts = df["predicted_label"].value_counts().reindex(CLASS_ORDER, fill_value=0)

    x = np.arange(len(CLASS_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, true_counts.values, width, label="True", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, pred_counts.values, width, label="Predicted", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylabel("Number of repos")
    ax.set_title("True vs Predicted Class Distribution")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, str(int(h)),
                    ha="center", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, str(int(h)),
                    ha="center", fontsize=8, color="#C44E52")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[OK] {save_path}")


def error_flow_chart(df: pd.DataFrame, save_path: Path) -> None:
    """Horizontal bar chart of top confusion pairs (true→pred)."""
    ct = pd.crosstab(df["label"], df["predicted_label"])
    pairs = []
    for true_label in CLASS_ORDER:
        for pred_label in CLASS_ORDER:
            if true_label != pred_label and ct.loc[true_label, pred_label] > 0:
                pairs.append(
                    {
                        "pair": f"{true_label} → {pred_label}",
                        "count": ct.loc[true_label, pred_label],
                        "true_class": true_label,
                    }
                )
    pairs_df = pd.DataFrame(pairs).sort_values("count", ascending=True)
    pairs_df = pairs_df.tail(12)  # top 12

    colors = [CLASS_ORDER.index(c) for c in pairs_df["true_class"]]
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(pairs_df["pair"], pairs_df["count"], color=[cmap(c / 6) for c in colors])

    for bar, val in zip(bars, pairs_df["count"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)

    ax.set_xlabel("Number of misclassifications")
    ax.set_title("Top Confusion Pairs (True → Predicted)")
    ax.set_xlim(0, pairs_df["count"].max() + 4)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[OK] {save_path}")


def confidence_distribution_chart(df: pd.DataFrame, save_path: Path) -> None:
    """KDE + boxplot of LLM label_confidence by correctness."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # KDE
    correct = df[df["correct"]]["label_confidence"]
    incorrect = df[~df["correct"]]["label_confidence"]

    ax = axes[0]
    ax.hist(correct, bins=8, alpha=0.6, label=f"Correct (n={len(correct)})", color="#55A868")
    ax.hist(incorrect, bins=8, alpha=0.6, label=f"Incorrect (n={len(incorrect)})", color="#C44E52")
    ax.axvline(correct.median(), color="#55A868", linestyle="--", linewidth=1.5)
    ax.axvline(incorrect.median(), color="#C44E52", linestyle="--", linewidth=1.5)
    ax.set_xlabel("LLM Label Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution by Correctness")
    ax.legend(frameon=True, fontsize=9)

    # Boxplot per class
    ax = axes[1]
    plot_data = [
        df[df["label"] == c]["label_confidence"].dropna().values
        for c in CLASS_ORDER
    ]
    bp = ax.boxplot(plot_data, tick_labels=CLASS_ORDER, patch_artist=True)
    for patch, i in zip(bp["boxes"], range(len(CLASS_ORDER))):
        patch.set_facecolor(plt.cm.tab10(i / 6))
        patch.set_alpha(0.7)
    ax.set_ylabel("LLM Label Confidence")
    ax.set_title("Label Confidence by True Class")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[OK] {save_path}")


def per_class_support_chart(metrics: dict, save_path: Path) -> None:
    """Scatter: support vs F1 to show data scarcity effect."""
    per_class = metrics["per_class"]
    supports = [per_class[c]["support"] for c in CLASS_ORDER]
    f1s = [per_class[c]["f1-score"] for c in CLASS_ORDER]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, c in enumerate(CLASS_ORDER):
        ax.scatter(supports[i], f1s[i], s=120, color=plt.cm.tab10(i / 6),
                   edgecolors="white", linewidth=1, zorder=3)
        ax.annotate(c, (supports[i] + 0.4, f1s[i] + 0.01), fontsize=10)

    # Trend line
    z = np.polyfit(supports, f1s, 1)
    x_line = np.linspace(min(supports) - 2, max(supports) + 2, 20)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.6, linewidth=1)

    ax.set_xlabel("Support (test set size)")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 vs Support: Data Scarcity Effect")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[OK] {save_path}")


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    metrics = compute_metrics(df)
    cm_norm = metrics["confusion_matrix_normalized"]

    print("Generating figures...\n")

    confusion_heatmap(cm_norm, FIGURES_DIR / "confusion_matrix.png")
    per_class_metrics_chart(metrics, FIGURES_DIR / "per_class_metrics.png")
    class_distribution_chart(df, FIGURES_DIR / "class_distribution.png")
    error_flow_chart(df, FIGURES_DIR / "confusion_pairs.png")
    confidence_distribution_chart(df, FIGURES_DIR / "label_confidence.png")
    per_class_support_chart(metrics, FIGURES_DIR / "f1_vs_support.png")

    print(f"\nDone. {len(list(FIGURES_DIR.glob('*.png')))} figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
