"""
Data processor for the katanaml/cord dataset.
Reads local CORD JSON annotation files (train/dev/test splits).
Maps CORD labels → simplified BIO NER tags and saves processed splits.

Dataset structure (after unzip):
  data/cord_dataset/CORD/
    train/json/*.json
    dev/json/*.json
    test/json/*.json

Each JSON file has:
  valid_line[]:
    category: str  (e.g. "total.total_price", "menu.nm")
    words[]:
      text: str
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Label definitions ────────────────────────────────────────────────────────

LABEL_LIST = ["O", "B-TOTAL", "I-TOTAL", "B-DATE", "I-DATE",
              "B-VENDOR", "I-VENDOR", "B-ID", "I-ID"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# CORD category → simplified field
CATEGORY_TO_FIELD = {
    # Total / amount fields
    "total.total_price":       "TOTAL",
    "total.total_etc":         "TOTAL",
    "total.cashprice":         "TOTAL",
    "total.changeprice":       "TOTAL",
    "total.creditcardprice":   "TOTAL",
    "total.emoneyprice":       "TOTAL",
    "total.menuqty_cnt":       "TOTAL",
    "total.menutype_cnt":      "TOTAL",
    "sub_total.subtotal_price":"TOTAL",
    "sub_total.tax_price":     "TOTAL",
    "sub_total.discount_price":"TOTAL",
    "sub_total.service_price": "TOTAL",
    # Vendor / store name (store name is usually the first line or "etc")
    "sub_total.etc":           "VENDOR",
}

# These cord labels are "ignored" (mapped to O) for our task
IGNORE_CATEGORIES = {
    "menu.nm", "menu.cnt", "menu.price", "menu.unitprice",
    "menu.num", "menu.discountprice", "menu.sub_nm",
    "menu.sub_cnt", "menu.sub_price",
}

CORD_DATASET_DIR = "data/cord_dataset/CORD"


def get_field_for_category(category: str) -> Optional[str]:
    """Return simplified field name for a CORD category, or None for O."""
    return CATEGORY_TO_FIELD.get(category)


def load_cord_split(split_dir: str, max_samples: int = -1) -> List[Dict]:
    """
    Load all JSON files from a CORD split directory.

    Args:
        split_dir: path to CORD/train, CORD/dev, or CORD/test
        max_samples: limit number of receipts (-1 = all)

    Returns:
        List of {"id", "tokens", "ner_tags"} dicts
    """
    ann_dir = Path(split_dir) / "json"
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

    examples = []
    files = sorted(ann_dir.glob("*.json"))

    for file_idx, json_file in enumerate(files):
        if max_samples > 0 and len(examples) >= max_samples:
            break

        with open(json_file, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        tokens: List[str] = []
        ner_tags: List[int] = []

        valid_lines = data.get("valid_line", [])
        for line in valid_lines:
            category = line.get("category", "O")
            field = get_field_for_category(category)

            words = line.get("words", [])
            for word_idx, word in enumerate(words):
                text = word.get("text", "").strip()
                if not text:
                    continue

                tokens.append(text)

                if field is None:
                    ner_tags.append(LABEL2ID["O"])
                elif word_idx == 0:
                    ner_tags.append(LABEL2ID[f"B-{field}"])
                else:
                    ner_tags.append(LABEL2ID[f"I-{field}"])

        if not tokens:
            continue

        examples.append({
            "id": json_file.stem,
            "tokens": tokens,
            "ner_tags": ner_tags,
        })

    return examples


def load_and_process_cord(
    output_dir: str = "data",
    cord_dir: str = CORD_DATASET_DIR,
    max_train: int = 800,
    max_val: int = 100,
    max_test: int = 100,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Process the local katanaml/cord dataset into NER examples.

    Args:
        output_dir:  where to save processed JSON files
        cord_dir:    path to the unzipped CORD directory
        max_train / max_val / max_test: subset sizes

    Returns:
        (train_examples, val_examples, test_examples)
    """
    cord_path = Path(cord_dir)
    if not cord_path.exists():
        raise FileNotFoundError(
            f"CORD dataset not found at {cord_dir}.\n"
            "Run: git clone https://huggingface.co/datasets/katanaml/cord data/cord_dataset\n"
            "Then download and unzip dataset.zip inside data/cord_dataset/"
        )

    split_map = {
        "train": ("train", max_train),
        "val":   ("dev",   max_val),
        "test":  ("test",  max_test),
    }

    results = {}
    for out_name, (folder, limit) in split_map.items():
        split_dir = cord_path / folder
        print(f"  Loading {out_name} from {split_dir}…")
        examples = load_cord_split(str(split_dir), max_samples=limit)
        results[out_name] = examples
        print(f"    → {len(examples)} examples loaded")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for out_name, examples in results.items():
        fname = "val.json" if out_name == "val" else f"{out_name}.json"
        fpath = Path(output_dir) / fname
        with open(fpath, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"  Saved {len(examples)} records → {fpath}")

    # Save label config
    label_cfg = {"label_list": LABEL_LIST, "label2id": LABEL2ID, "id2label": ID2LABEL}
    with open(Path(output_dir) / "label_config.json", "w") as f:
        json.dump(label_cfg, f, indent=2)

    return results["train"], results["val"], results["test"]


# ─── Regex-based fallback extraction ─────────────────────────────────────────

TOTAL_RE = re.compile(
    r"(?:grand\s*)?(?:total|tota|t0tal|amount|subtotal|sum)[^\d]*([\d,\.\s]+)",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
VENDOR_RE = re.compile(r"^([A-Z0-9][A-Za-z0-9\-\s&\'\.]{2,40})$", re.MULTILINE)
ID_RE = re.compile(
    r"(?:receipt|invoice|no|id|#)[^\w]*([A-Z0-9\-]{4,20})",
    re.IGNORECASE,
)


def regex_extract(text: str) -> Dict:
    """Fast regex-based extraction as fallback when model isn't available."""
    result: Dict[str, Optional[str]] = {
        "total_amount": None,
        "date": None,
        "vendor_name": None,
        "receipt_id": None,
    }

    # Clean currency symbols before searching to avoid breaking the word boundary
    clean_text = re.sub(r'[Rr][Pp]\.?|\$|€|£|¥', '', text)

    m = TOTAL_RE.search(clean_text)
    if m:
        # Get the first captured group and strip spaces/extra characters
        val = m.group(1).strip()
        result["total_amount"] = val
    else:
        # Fallback: Find anything that looks like a high-value amount (at least 2 digits)
        # Handle formats like 22,000 or 22.000 or 22 000 
        candidates = re.findall(r'\b\d{1,3}(?:[.,\s]\d{3})+(?:\.\d{2})?\b|\b\d{1,}\.\d{2}\b', clean_text)
        if candidates:
            # Grand total is usually the last/highest price mentioned relative to label
            result["total_amount"] = candidates[-1].replace(" ", "")
        else:
            # Last resort: Any number with a decimal/comma at the end of the text
            last_nums = re.findall(r'(\d+[.,\s]\d+)', clean_text)
            if last_nums:
                result["total_amount"] = last_nums[-1].replace(" ", "")


    m = DATE_RE.search(text)
    if m:
        result["date"] = m.group(1)

    # Vendor Name heuristic: Usually the first or second substantive line
    # that isn't a phone number, date, or long string of digits.
    lines = [L.strip() for L in text.split('\n') if len(L.strip()) >= 3]
    for line in lines[:5]:
        # Must contain at least 3 actual letters to be considered a vendor name
        if not re.search(r'[A-Za-z]{3,}', line):
            continue
            
        # Reject lines that perfectly match common item formats: "1 Ayam Pop" or contain prices
        if re.search(r'^\d{1,2}\s+[a-zA-Z]', line) or re.search(r'\d{2,}[\.,]\d{2,}', line):
            continue
            
        # If it doesn't have too many numbers (like a phone/tax ID), it's probably the vendor brand.
        if not re.search(r'\d{5,}', line) and not TOTAL_RE.search(line):
            result["vendor_name"] = line
            break

    m = ID_RE.search(text)
    if m:
        result["receipt_id"] = m.group(1)

    return result


if __name__ == "__main__":
    print("Processing katanaml/cord dataset…")
    train, val, test = load_and_process_cord()
    print(f"\nDone. Train={len(train)}, Val={len(val)}, Test={len(test)}")
    if train:
        ex = train[0]
        print(f"\nSample tokens  : {ex['tokens'][:10]}")
        print(f"Sample ner_tags: {[LABEL_LIST[t] for t in ex['ner_tags'][:10]]}")
