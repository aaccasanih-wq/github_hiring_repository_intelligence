"""
BERT fine-tuning for repository maturity classification.
Stage 5 of the GitHub Hiring Repository Intelligence pipeline.

Fine-tunes DistilBERT on the text_repr column with class-weighted loss
to handle the imbalanced 6-category dataset (539 train / 115 val / 116 test).

Set APPROACH env var to switch between labeling strategies:
  APPROACH=v1 (default) — baseline prompt
  APPROACH=v2            — alternative operationalized prompt
"""

import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset as HFDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.utils import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    NUM_CLASSES,
    ensure_dir,
    get_project_root,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 27
GRADIENT_ACCUMULATION = 1

# Device detection
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Approach: "v1" (baseline) or "v2" (alternative). Set via APPROACH env var.
APPROACH = os.getenv("APPROACH", "v1")
SUFFIX = "" if APPROACH == "v1" else f"_{APPROACH}"

# Paths
ROOT = get_project_root()
MODEL_DIR = ROOT / "models" / f"trained_models{SUFFIX}"
OUTPUT_DIR = ROOT / "output"
if APPROACH != "v1":
    OUTPUT_DIR = OUTPUT_DIR / APPROACH

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_split(name: str) -> pd.DataFrame:
    """Load a single split CSV (respects APPROACH env var for file naming)."""
    path = ROOT / "data" / "splits" / f"{name}{SUFFIX}.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def tokenize_fn(tokenizer, examples):
    return tokenizer(
        examples["text_repr"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0,
    )
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
    }


# ---------------------------------------------------------------------------
# Weighted Trainer
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train():
    logger.info("===== Stage 5: BERT Fine-Tuning (approach=%s) =====", APPROACH)
    logger.info("Device: %s | Model: %s | Output: %s", DEVICE, MODEL_NAME, OUTPUT_DIR)

    # -- Load data --
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    logger.info(
        "Loaded splits — train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    # -- Class weights --
    class_weights = compute_class_weight(
        "balanced",
        classes=np.arange(NUM_CLASSES),
        y=train_df["label_id"].values,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    logger.info("Class weights: %s", np.round(class_weights.cpu().numpy(), 3))

    # -- Tokenizer --
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # -- HF Datasets --
    train_ds = HFDataset.from_pandas(
        train_df[["text_repr", "label_id"]].rename(columns={"label_id": "labels"})
    )
    val_ds = HFDataset.from_pandas(
        val_df[["text_repr", "label_id"]].rename(columns={"label_id": "labels"})
    )
    test_ds = HFDataset.from_pandas(
        test_df[["text_repr", "label_id"]].rename(columns={"label_id": "labels"})
    )

    train_ds = train_ds.map(lambda x: tokenize_fn(tokenizer, x), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_fn(tokenizer, x), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_fn(tokenizer, x), batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    # -- Model --
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    # -- Training arguments --
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        logging_strategy="steps",
        logging_steps=25,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
    )

    # -- Trainer --
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    # -- Train --
    logger.info("Starting training (%d epochs, batch=%d, grad_accum=%d)...",
                EPOCHS, BATCH_SIZE, GRADIENT_ACCUMULATION)
    trainer.train()

    # -- Save --
    ensure_dir(MODEL_DIR)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    logger.info("Model saved to %s", MODEL_DIR)

    # -- Evaluate on test set --
    logger.info("Evaluating on test set...")
    preds_output = trainer.predict(test_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    report_str = classification_report(
        labels, preds,
        target_names=list(LABEL_TO_ID.keys()),
        zero_division=0,
    )
    logger.info("\n%s", report_str)

    # -- Save outputs --
    ensure_dir(OUTPUT_DIR / "metrics")

    # Classification report
    per_class = classification_report(
        labels, preds,
        target_names=list(LABEL_TO_ID.keys()),
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(per_class).transpose().to_csv(OUTPUT_DIR / "metrics" / "test_metrics.csv")
    logger.info("Metrics saved to %s", OUTPUT_DIR / "metrics" / "test_metrics.csv")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=list(LABEL_TO_ID.keys()), columns=list(LABEL_TO_ID.keys()))
    cm_df.to_csv(OUTPUT_DIR / "metrics" / "confusion_matrix.csv")

    # Predictions
    results_df = test_df.copy()
    results_df["predicted_id"] = preds
    results_df["predicted_label"] = [ID_TO_LABEL[p] for p in preds]
    results_df["correct"] = results_df["label_id"] == results_df["predicted_id"]
    results_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    logger.info("Predictions saved to %s", OUTPUT_DIR / "test_predictions.csv")

    # -- Summary --
    test_acc = accuracy_score(labels, preds)
    logger.info("===== Training complete | Test accuracy: %.4f =====", test_acc)


if __name__ == "__main__":
    train()
