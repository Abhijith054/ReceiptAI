"""
Fine-tune DistilBERT token classification (NER) on CORD receipt data.
Trains on a lightweight subset; saves best checkpoint to models/receipt_ner/.
Prints precision / recall / F1 using seqeval.
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset, DatasetDict
import evaluate

from src.data_processor import (
    LABEL_LIST, LABEL2ID, ID2LABEL,
    load_and_process_cord,
)

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_MODEL = "distilbert-base-uncased"
OUTPUT_DIR = "models/receipt-bert-v1"
DATA_DIR = "data"
MAX_LENGTH = 128


def tokenize_and_align_labels(examples, tokenizer):
    """Word-piece tokenize tokens and propagate NER labels to sub-tokens."""
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
    )

    labels_out = []
    for i, label_ids in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_row = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_row.append(-100)  # special tokens → ignored in loss
            elif word_id != prev_word_id:
                label_row.append(label_ids[word_id])
            else:
                # Sub-token of same word: use I- tag if possible
                tag = LABEL_LIST[label_ids[word_id]]
                if tag.startswith("B-"):
                    tag = "I-" + tag[2:]
                label_row.append(LABEL2ID.get(tag, label_ids[word_id]))
            prev_word_id = word_id
        labels_out.append(label_row)

    tokenized["labels"] = labels_out
    return tokenized


def build_dataset(data_dir: str, tokenizer) -> DatasetDict:
    """Load processed JSON splits and return tokenized HuggingFace DatasetDict."""
    splits = {}
    for split in ("train", "val", "test"):
        fpath = Path(data_dir) / f"{split}.json"
        if not fpath.exists():
            raise FileNotFoundError(
                f"{fpath} not found. Run data_processor.py first:\n"
                f"  python -m src.data_processor"
            )
        with open(fpath) as f:
            records = json.load(f)

        # Filter out empty examples
        records = [r for r in records if r["tokens"]]
        ds = Dataset.from_list(records)
        ds = ds.map(
            lambda ex: tokenize_and_align_labels(ex, tokenizer),
            batched=True,
            remove_columns=["id", "tokens", "ner_tags"],
        )
        splits[split if split != "val" else "validation"] = ds

    return DatasetDict(splits)


def compute_metrics(p, label_list=LABEL_LIST):
    """Compute seqeval precision / recall / F1 / accuracy."""
    seqeval = evaluate.load("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [label_list[l] for l in label if l != -100]
        for label in labels
    ]
    true_preds = [
        [label_list[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def train(
    data_dir: str = DATA_DIR,
    output_dir: str = OUTPUT_DIR,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    download_data: bool = True,
    max_train: int = 600,
    max_val: int = 100,
    max_test: int = 100,
):
    """
    Full training pipeline:
    1. (Optionally) download + process CORD data
    2. Tokenize
    3. Fine-tune DistilBERT
    4. Evaluate and save
    """
    # Step 1 – data
    if download_data:
        print("=== Step 1/4: Downloading and processing data ===")
        load_and_process_cord(data_dir, max_train, max_val, max_test)

    # Step 2 – tokenizer + dataset
    print("=== Step 2/4: Tokenizing dataset ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    ds = build_dataset(data_dir, tokenizer)
    print(f"  Dataset sizes: {ds}")

    # Step 3 – model
    print("=== Step 3/4: Setting up model ===")
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Use CPU-friendly settings; auto-detects GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Training on: {device}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"{output_dir}/logs",
        logging_steps=20,
        report_to="none",  # disable wandb
        save_total_limit=2,
        use_cpu=(device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Step 4 – train
    print("=== Step 4/4: Training ===")
    trainer.train()

    # Evaluate on test set
    print("\n=== Evaluation on test set ===")
    test_results = trainer.evaluate(ds["test"])
    print(json.dumps(test_results, indent=2))

    # Save model + tokenizer
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label config alongside model
    label_cfg = {"label_list": LABEL_LIST, "label2id": LABEL2ID, "id2label": ID2LABEL}
    with open(Path(output_dir) / "label_config.json", "w") as f:
        json.dump(label_cfg, f, indent=2)

    # Save evaluation metrics
    with open(Path(output_dir) / "eval_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\nModel saved to {output_dir}/")
    print(f"F1: {test_results.get('eval_f1', 'N/A'):.4f}")
    return test_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on CORD NER")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--no-download", action="store_true",
                        help="Skip data download (use existing data/ files)")
    parser.add_argument("--max-train", type=int, default=600)
    parser.add_argument("--max-val", type=int, default=100)
    parser.add_argument("--max-test", type=int, default=100)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        download_data=not args.no_download,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
    )
