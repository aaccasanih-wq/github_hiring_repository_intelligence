"""
Stage 6 — Evaluation & Error Analysis.

Sections:
  1. Quantitative Evaluation  — metrics, confusion matrix
  2. Error Analysis            — confusion patterns, confidence, feature analysis
  3. Methodological Diagnosis  — why did the model struggle?
  4. Baseline Limitations       — inherent constraints
  5. Motivation for Alternative Approach
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# --- paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ID_TO_LABEL, LABEL_TO_ID, load_dotenv

load_dotenv()

OUTPUT_DIR = PROJECT_ROOT / "output"
METRICS_DIR = OUTPUT_DIR / "metrics"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = OUTPUT_DIR / "evaluation_report.md"

CLASS_ORDER = ["intern", "junior", "senior", "lead", "template", "low-value"]
N_CLASSES = len(CLASS_ORDER)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(OUTPUT_DIR / "test_predictions.csv")
    df["label"] = pd.Categorical(df["label"], categories=CLASS_ORDER, ordered=True)
    df["predicted_label"] = pd.Categorical(
        df["predicted_label"], categories=CLASS_ORDER, ordered=True
    )
    return df


# ======================================================================
# 1. QUANTITATIVE EVALUATION
# ======================================================================

def compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df["label"].values
    y_pred = df["predicted_label"].values

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    per_class = classification_report(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0, output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    metrics = {
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "weighted_precision": prec_weighted,
        "weighted_recall": rec_weighted,
        "weighted_f1": f1_weighted,
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_norm,
    }
    return metrics


def save_metrics(metrics: dict) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    summary = {
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
    }
    pd.DataFrame([summary]).to_csv(METRICS_DIR / "summary_metrics.csv", index=False)

    # Per-class CSV
    per_class_rows = []
    for label in CLASS_ORDER:
        d = metrics["per_class"][label]
        per_class_rows.append(
            {
                "class": label,
                "label_id": LABEL_TO_ID[label],
                "precision": d["precision"],
                "recall": d["recall"],
                "f1_score": d["f1-score"],
                "support": int(d["support"]),
            }
        )
    pd.DataFrame(per_class_rows).to_csv(
        METRICS_DIR / "per_class_metrics.csv", index=False
    )

    # Confusion matrix CSV
    cm_df = pd.DataFrame(
        metrics["confusion_matrix"], index=CLASS_ORDER, columns=CLASS_ORDER
    )
    cm_df.to_csv(TABLES_DIR / "confusion_matrix.csv")

    # Normalized confusion matrix
    cm_norm_df = pd.DataFrame(
        metrics["confusion_matrix_normalized"], index=CLASS_ORDER, columns=CLASS_ORDER
    )
    cm_norm_df.to_csv(TABLES_DIR / "confusion_matrix_normalized.csv")

    print("[OK] Metrics saved to output/metrics/ and output/tables/")


def print_metrics(metrics: dict) -> None:
    print(f"\n{'='*60}")
    print("1. QUANTITATIVE EVALUATION")
    print(f"{'='*60}")
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro F1:  {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"\nPer-class metrics:")
    print(f"{'Class':<12} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Supp':>6}")
    print("-" * 42)
    for label in CLASS_ORDER:
        d = metrics["per_class"][label]
        print(
            f"{label:<12} {d['precision']:7.3f} {d['recall']:7.3f} "
            f"{d['f1-score']:7.3f} {int(d['support']):6d}"
        )


# ======================================================================
# 2. ERROR ANALYSIS
# ======================================================================

def error_analysis(df: pd.DataFrame, metrics: dict) -> dict:
    print(f"\n{'='*60}")
    print("2. ERROR ANALYSIS")
    print(f"{'='*60}")

    y_true = df["label"].values
    y_pred = df["predicted_label"].values

    # -- Confusion pairs -------------------------------------------------
    cm = metrics["confusion_matrix"]
    confusion_pairs = []
    for i, true_label in enumerate(CLASS_ORDER):
        for j, pred_label in enumerate(CLASS_ORDER):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append(
                    {
                        "true": true_label,
                        "predicted": pred_label,
                        "count": int(cm[i, j]),
                        "pct_of_true": float(cm[i, j] / cm[i].sum()),
                    }
                )
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)

    print("\nTop-10 confusion pairs:")
    for pair in confusion_pairs[:10]:
        print(
            f"  {pair['true']:>10} → {pair['predicted']:<10}  "
            f"{pair['count']:2d}  ({pair['pct_of_true']:.1%} of true {pair['true']})"
        )

    # -- Adjacent vs non-adjacent errors ---------------------------------
    maturity_order = {c: i for i, c in enumerate(CLASS_ORDER)}
    adjacent_errs = 0
    non_adjacent_errs = 0
    total_errs = 0
    for pair in confusion_pairs:
        dist = abs(maturity_order[pair["true"]] - maturity_order[pair["predicted"]])
        c = pair["count"]
        total_errs += c
        if dist == 1:
            adjacent_errs += c
        else:
            non_adjacent_errs += c

    pct_adj = adjacent_errs / total_errs if total_errs else 0
    print(f"\nAdjacent-level errors:   {adjacent_errs}/{total_errs} ({pct_adj:.1%})")
    print(f"Non-adjacent errors:     {non_adjacent_errs}/{total_errs} ({1-pct_adj:.1%})")

    # -- Error type categorization ---------------------------------------
    # senior→lead specifically
    senior_to_lead = next(
        (p for p in confusion_pairs if p["true"] == "senior" and p["predicted"] == "lead"),
        None,
    )
    if senior_to_lead:
        print(
            f"\nsenior→lead: {senior_to_lead['count']} errors "
            f"({senior_to_lead['count']/total_errs:.1%} of all errors)"
        )

    # low-value→{intern,junior}
    lv_to_low = sum(
        p["count"]
        for p in confusion_pairs
        if p["true"] == "low-value" and p["predicted"] in ("intern", "junior")
    )
    print(
        f"low-value→intern/junior: {lv_to_low} errors "
        f"({lv_to_low/total_errs:.1%} of all errors)" if total_errs else ""
    )

    # -- model confusion rate (low-value→senior is classifier, not low-level) --
    low_value_to_senior = sum(
        p["count"]
        for p in confusion_pairs
        if p["true"] == "low-value" and p["predicted"] == "senior"
    )
    print(
        f"low-value→senior: {low_value_to_senior} errors "
        f"({low_value_to_senior/total_errs:.1%} of all errors)" if total_errs else ""
    )

    # -- Class-level recall analysis -------------------------------------
    print("\nClass recall (ability to find all true instances):")
    for label in CLASS_ORDER:
        d = metrics["per_class"][label]
        print(f"  {label:<12} recall={d['recall']:.3f}  support={int(d['support'])}")

    # -- Class-level precision analysis ----------------------------------
    print("\nClass precision (purity of predictions):")
    for label in CLASS_ORDER:
        d = metrics["per_class"][label]
        print(f"  {label:<12} precision={d['precision']:.3f}  support={int(d['support'])}")

    # -- LLM confidence vs correctness -----------------------------------
    df_conf = df.dropna(subset=["label_confidence"])
    correct_conf = df_conf[df_conf["correct"]]["label_confidence"]
    incorrect_conf = df_conf[~df_conf["correct"]]["label_confidence"]
    print(f"\nLLM confidence (label_confidence) vs model correctness:")
    print(f"  Correct:   mean={correct_conf.mean():.3f}  median={correct_conf.median():.3f}")
    print(f"  Incorrect: mean={incorrect_conf.mean():.3f}  median={incorrect_conf.median():.3f}")

    return {
        "confusion_pairs": confusion_pairs,
        "adjacent_errors": adjacent_errs,
        "non_adjacent_errors": non_adjacent_errs,
        "total_errors": total_errs,
        "senior_to_lead": senior_to_lead["count"] if senior_to_lead else 0,
        "lv_to_intern_junior": lv_to_low,
    }


# ======================================================================
# 3. METHODOLOGICAL DIAGNOSIS
# ======================================================================

def methodological_diagnosis(df: pd.DataFrame, metrics: dict, error_info: dict) -> str:
    """Return a markdown string with the diagnosis."""
    cm_norm = metrics["confusion_matrix_normalized"]
    per_class = metrics["per_class"]

    lines = []
    lines.append("## 3. Methodological Diagnosis\n")

    # 3.1 Class imbalance
    supports = {lbl: int(per_class[lbl]["support"]) for lbl in CLASS_ORDER}
    max_sup = max(supports.values())
    min_sup = min(supports.values())
    imbalance_ratio = max_sup / min_sup if min_sup > 0 else float("inf")

    lines.append("### 3.1 Class Imbalance\n")
    lines.append(
        f"The training set has a **{imbalance_ratio:.1f}:1 imbalance ratio** "
        f"(senior={max_sup} vs template={min_sup} in test). "
        "Minority classes (template, junior, intern) have fewer examples for the model to learn "
        "distinctive patterns. Although class-weighted loss was used, weight alone cannot "
        "compensate for the lack of diverse training samples.\n"
    )

    # 3.2 Adjacent class ambiguity
    lines.append("### 3.2 Adjacent-Class Ambiguity\n")
    adj_pct = error_info["adjacent_errors"] / error_info["total_errors"] if error_info["total_errors"] else 0
    lines.append(
        f"**{adj_pct:.0%}** of all errors are between adjacent maturity levels "
        "(intern↔junior, junior↔senior, senior↔lead). "
        "This is expected: the boundaries between adjacent levels are inherently fuzzy. "
        "For example, a senior repo with many contributors and CI/CD might look like a lead repo. "
        "This suggests the 6-class schema has inherent inter-class overlap that even the LLM labeler "
        "may not resolve consistently.\n"
    )

    # 3.3 The senior↔lead confusion
    stl = error_info.get("senior_to_lead", 0)
    lines.append("### 3.3 The senior→lead Confusion Dominates\n")
    lines.append(
        f"The single biggest error pattern is **senior→lead** ({stl} cases, "
        f"{stl/error_info['total_errors']:.0%} of all errors). "
        "lead has the highest recall (90%) because the model over-predicts it — it predicts lead "
        "35 times for only 21 true leads. The model learned that lead-like signals (many "
        "contributors, CI/CD, releases) are strong, but it cannot distinguish senior from lead "
        "when both share these signals. The text_repr may not capture the *qualitative* "
        "difference (code review practices, governance maturity) that separates them.\n"
    )

    # 3.4 low-value is hard
    lv_recall = per_class["low-value"]["recall"]
    lines.append("### 3.4 Low-Value Repos Are Hard to Identify\n")
    lines.append(
        f"low-value has the worst recall ({lv_recall:.0%}). "
        f"The model confuses it most with **intern** ({error_info['lv_to_intern_junior']} cases "
        f"to intern/junior). This suggests that superficial signals in the text_repr "
        "(presence of a README, some code structure) make low-value repos appear more mature "
        "than they really are. Truly low-value repos may be under-represented in the "
        "training data because the sampling strategy targeted repos with *some* activity.\n"
    )

    # 3.5 Model capacity
    lines.append("### 3.5 Model Capacity (DistilBERT)\n")
    lines.append(
        "DistilBERT (66M parameters) is a compressed model that trades 40% size reduction "
        "for ~95% performance retention on GLUE benchmarks. However, this task requires "
        "fine-grained understanding of software engineering signals buried in README text "
        "averaging 717 characters. The model may lack capacity to capture subtle distinctions "
        "(e.g., testing practices, CI/CD configuration quality) from truncated text.\n"
    )

    # 3.6 Text representation limits
    lines.append("### 3.6 Text Representation Limits\n")
    lines.append(
        "The `text_repr` column averages 717 characters (well under DistilBERT's 512-token "
        "limit). While this avoids truncation, it may omit signals that distinguish adjacent "
        "classes: code quality indicators, dependency freshness, architectural patterns. "
        "The structured format (key: value pairs) is readable but may not be the optimal "
        "representation for a transformer — raw README text might carry more signal.\n"
    )

    return "\n".join(lines)


# ======================================================================
# 4. BASELINE LIMITATIONS
# ======================================================================

def baseline_limitations() -> str:
    lines = []
    lines.append("## 4. Baseline Limitations\n")

    limitations = [
        (
            "Weak Supervision Noise",
            "The labels are generated by DeepSeek (an LLM), not human annotators. "
            "LLM labeling accuracy on this task is unknown — the model may be learning "
            "to mimic an LLM's biases rather than true maturity signals. "
            "Without a human-labeled validation set, we cannot measure label quality.",
        ),
        (
            "Single-Model, Single-Run",
            "One DistilBERT model trained with seed=42. No ensemble, no k-fold CV, "
            "no hyperparameter search beyond the defaults. The 40.5% accuracy is a "
            "point estimate with unknown variance.",
        ),
        (
            "Sampling Bias",
            "The Two-Round Informed Sampling targets repos with *some* activity "
            "(stars, commits, recent pushes). This under-samples truly low-value or "
            "abandoned repos, skewing the distribution toward higher maturity levels.",
        ),
        (
            "Text-Only Input",
            "Only `text_repr` is used. Numeric signals (stars, forks, contributors, "
            "age, commit frequency) are excluded to avoid popularity bias — but these "
            "signals are strongly correlated with maturity and could improve "
            "classification if used properly (e.g., as auxiliary features).",
        ),
        (
            "Ordinal Task Treated as Nominal",
            "Maturity levels are ordinal (intern < junior < senior < lead). "
            "The model uses cross-entropy loss, which treats all misclassifications "
            "equally. An ordinal regression or hierarchical approach would penalize "
            "senior→template more heavily than senior→lead.",
        ),
        (
            "Limited Training Data",
            "539 training examples across 6 classes (~90/class avg) is small for "
            "fine-tuning a transformer. Minority classes have as few as 44 examples "
            "(template). Data augmentation or few-shot techniques were not explored.",
        ),
        (
            "No Reproducibility Guarantee",
            "Training ran on Google Colab with a T4 GPU. The exact environment "
            "(CUDA version, Python package versions) was not pinned. Reproducing "
            "the exact weights may require matching the Colab runtime.",
        ),
    ]

    for title, body in limitations:
        lines.append(f"### 4.{limitations.index((title, body))+1} {title}\n")
        lines.append(f"{body}\n")

    return "\n".join(lines)


# ======================================================================
# 5. MOTIVATION FOR ALTERNATIVE APPROACH
# ======================================================================

def motivation_for_alternative_approach() -> str:
    lines = []
    lines.append("## 5. Motivation for an Alternative Approach\n")

    approaches = [
        (
            "Larger Model + Longer Context",
            "Replace DistilBERT with **CodeBERT**, **GraphCodeBERT**, or **RoBERTa-base**. "
            "These models have more capacity and, in the case of CodeBERT, are pre-trained "
            "on code+text pairs — directly relevant to repository understanding. "
            "Additionally, feed the full README (not just the structured text_repr) to "
            "capture richer signals.",
        ),
        (
            "Hierarchical / Ordinal Classification",
            "Replace flat cross-entropy with an **ordinal regression** head "
            "(e.g., Coral loss or cumulative link models). This explicitly models the "
            "ordered nature of maturity levels and should reduce adjacent-class confusion. "
            "Alternatively, a two-stage classifier: first identify template/low-value vs "
            "active repos, then classify maturity level among active repos.",
        ),
        (
            "Multi-Modal Fusion",
            "Combine text features with numeric signals (stars, forks, age, commit "
            "frequency, contributor count) via a late-fusion architecture. The numeric "
            "signals provide strong priors (e.g., 0 stars + 0 commits = likely low-value) "
            "that text alone may miss.",
        ),
        (
            "Human-in-the-Loop Labeling",
            "Curate a small (50-100) human-labeled validation set to measure LLM label "
            "quality. If LLM accuracy is below ~80%, invest in human labeling for at "
            "least the ambiguous cases (senior vs lead, low-value vs intern). Use this "
            "to train a label error detection model.",
        ),
        (
            "Data Augmentation for Minority Classes",
            "Use back-translation, README paraphrasing, or LLM-based synthetic data "
            "generation to boost template, junior, and intern classes. Even 2-3x "
            "augmentation could meaningfully improve recall for those classes.",
        ),
        (
            "Confidence-Based Error Detection",
            "The current model provides no prediction confidence. Adding Monte Carlo "
            "Dropout or a calibration layer would allow flagging low-confidence "
            "predictions for human review — critical if this system is used for hiring "
            "intelligence where wrong classifications have real consequences.",
        ),
        (
            "Ensemble of Weak Supervisors",
            "Instead of relying on a single LLM (DeepSeek) for labels, use 2-3 "
            "different LLMs and take majority vote or model the label uncertainty. "
            "This reduces individual LLM bias and provides a crude measure of label "
            "difficulty per example.",
        ),
    ]

    for title, body in approaches:
        lines.append(f"### 5.{approaches.index((title, body))+1} {title}\n")
        lines.append(f"{body}\n")

    return "\n".join(lines)


# ======================================================================
# REPORT GENERATION
# ======================================================================

def generate_report(
    metrics: dict, error_info: dict, diagnosis: str, limitations: str, motivation: str
) -> str:
    lines = []
    lines.append("# Stage 6 — Evaluation Report\n")
    lines.append(f"**Model:** DistilBERT (distilbert-base-uncased, 66M params)")
    lines.append(f"**Test set:** {metrics['per_class']['macro avg']['support']:.0f} repos")
    lines.append(f"**Date:** 2026-05-24\n")

    # -- Section 1 summary --
    lines.append("## 1. Quantitative Evaluation\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Accuracy | {metrics['accuracy']:.4f} |")
    lines.append(f"| Macro Precision | {metrics['macro_precision']:.4f} |")
    lines.append(f"| Macro Recall | {metrics['macro_recall']:.4f} |")
    lines.append(f"| Macro F1 | {metrics['macro_f1']:.4f} |")
    lines.append(f"| Weighted F1 | {metrics['weighted_f1']:.4f} |\n")

    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|-------|-----------|--------|----|---------|")
    for label in CLASS_ORDER:
        d = metrics["per_class"][label]
        lines.append(
            f"| {label} | {d['precision']:.3f} | {d['recall']:.3f} | "
            f"{d['f1-score']:.3f} | {int(d['support'])} |"
        )
    lines.append("")

    # -- Section 2 summary --
    lines.append("## 2. Error Analysis\n")
    top_3 = error_info["confusion_pairs"][:3]
    lines.append("**Top confusion patterns:**\n")
    for p in top_3:
        lines.append(
            f"- **{p['true']} → {p['predicted']}**: {p['count']} cases "
            f"({p['pct_of_true']:.1%} of true {p['true']})"
        )
    lines.append(
        f"\n**{error_info['adjacent_errors']}/{error_info['total_errors']}** errors "
        f"({error_info['adjacent_errors']/error_info['total_errors']:.1%}) are between "
        f"adjacent maturity levels.\n"
    )

    # -- Sections 3-5 --
    lines.append(diagnosis)
    lines.append(limitations)
    lines.append(motivation)

    return "\n".join(lines)


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    for d in [METRICS_DIR, TABLES_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df)} test predictions")

    # 1. Quantitative Evaluation
    metrics = compute_metrics(df)
    print_metrics(metrics)
    save_metrics(metrics)

    # 2. Error Analysis
    error_info = error_analysis(df, metrics)

    # 3-5. Qualitative sections
    diagnosis = methodological_diagnosis(df, metrics, error_info)
    limitations = baseline_limitations()
    motivation = motivation_for_alternative_approach()

    # Generate report
    report = generate_report(metrics, error_info, diagnosis, limitations, motivation)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n[OK] Report saved to {REPORT_PATH}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Best class: lead (F1={metrics['per_class']['lead']['f1-score']:.3f})")
    worst_class = min(CLASS_ORDER, key=lambda c: metrics["per_class"][c]["f1-score"])
    print(f"Worst class: {worst_class} (F1={metrics['per_class'][worst_class]['f1-score']:.3f})")
    print(f"Top error: senior→lead ({error_info['senior_to_lead']} cases)")
    print(f"Adjacent-level errors: {error_info['adjacent_errors']}/{error_info['total_errors']}")
    print(f"\nOutput files:")
    print(f"  {METRICS_DIR}/summary_metrics.csv")
    print(f"  {METRICS_DIR}/per_class_metrics.csv")
    print(f"  {TABLES_DIR}/confusion_matrix.csv")
    print(f"  {TABLES_DIR}/confusion_matrix_normalized.csv")
    print(f"  {REPORT_PATH}")


if __name__ == "__main__":
    main()
