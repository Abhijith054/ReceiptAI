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


# ─── Improved Regex & Cleaning ────────────────────────────────────────────────

def normalize_ocr(text: str) -> str:
    """Clean and normalize OCR text for better extraction."""
    if not text: return ""
    # Remove extra whitespace and standardized common artifacts
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

TOTAL_KEYWORDS = ["total", "subtotal", "grand total", "amount due", "net amt", "total price"]
IGNORE_KEYWORDS = ["cash", "change", "tax", "vat", "discount", "promo", "saving", "bal", "due"]

def parse_amount(text: str) -> Optional[float]:
    """Convert string amount to float, handling common currency separators."""
    if not text: return None
    try:
        # Remove currency symbols and non-numeric except . and ,
        clean = re.sub(r'[^\d.,]', '', text)
        if not clean: return None
        
        # Handle cases like 197.450 -> 197450 (if thousand separator) vs 197.45 (decimal)
        # Heuristic: if there are 3 digits after the separator, it's likely a thousands separator (RP/Euro style)
        if ',' in clean and '.' in clean:
            # Traditional US: 1,234.56
            clean = clean.replace(',', '')
        elif ',' in clean:
            # Euro style: 1.234,56 or 1234,56
            if len(clean.split(',')[-1]) == 3:
                clean = clean.replace(',', '') # thousands
            else:
                clean = clean.replace(',', '.') # decimal
        elif '.' in clean:
            # Period only: 1.234.567 or 1234.56
            parts = clean.split('.')
            if len(parts[-1]) == 3 and len(parts) > 1:
                clean = clean.replace('.', '') # thousands
        
        return float(clean)
    except:
        return None

def regex_extract(text: str) -> Dict:
    """Enhanced regex extraction with smart total detection."""
    result = {"vendor": None, "date": None, "total_amount": None}
    
    # 1. Date Extraction (Multiple Formats)
    date_patterns = [
        r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})', # 25/12/2023
        r'(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})', # 2023/12/25
        r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4})', # 25 Dec 2023
    ]
    for pattern in date_patterns:
        m = re.search(pattern, text, re.I)
        if m:
            result["date"] = m.group(1)
            break

    # 2. Smart Total Detection
    lines = text.split('\n')
    candidates = []
    
    # regex for numbers like 10,000.00 or 10.000
    price_pattern = re.compile(r'(\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{2})?|\d{1,}[.,]\d{2})')

    for i, line in enumerate(lines):
        line_lower = line.lower()
        # Find lines with total keywords
        if any(kw in line_lower for kw in TOTAL_KEYWORDS):
            # Skip lines with ignore keywords
            if any(ik in line_lower for ik in IGNORE_KEYWORDS if ik not in line_lower):
                 # Wait, 'cash' might be on the same line. If line has both 'total' and 'cash', it's usually cash tendered.
                 if "cash" in line_lower or "change" in line_lower:
                     continue
            
            # Find numbers in this line or next line
            nums = price_pattern.findall(line)
            if not nums and i + 1 < len(lines):
                nums = price_pattern.findall(lines[i+1])
            
            for n in nums:
                val = parse_amount(n)
                if val: candidates.append(val)

    if candidates:
        # Priority: usually the largest number found near 'total' keywords
        result["total_amount"] = max(candidates)
    else:
        # Global fallback: largest number in whole text (risky)
        all_nums = price_pattern.findall(text)
        vals = [parse_amount(n) for n in all_nums if parse_amount(n)]
        # Filter out numbers that look like dates or phone numbers (e.g. > 10 chars)
        vals = [v for v in vals if v < 10000000] # threshold for noise
        if vals:
            result["total_amount"] = max(vals)

    # 3. Vendor Name Extraction
    # Filter out noise lines
    noise = ["receipt", "invoice", "terminal", "welcome", "thank you", "cashier", "date", "time", "order", "reprint"]
    substantive_lines = []
    for line in lines[:8]:
        L = line.strip()
        if len(L) < 3: continue
        if any(n in L.lower() for n in noise): continue
        if re.search(r'\d{5,}', L): continue # likely ID or phone
        substantive_lines.append(L)
    
    if substantive_lines:
        result["vendor"] = substantive_lines[0]

    return result


if __name__ == "__main__":
    print("Processing katanaml/cord dataset…")
    train, val, test = load_and_process_cord()
    print(f"\nDone. Train={len(train)}, Val={len(val)}, Test={len(test)}")
    if train:
        ex = train[0]
        print(f"\nSample tokens  : {ex['tokens'][:10]}")
        print(f"Sample ner_tags: {[LABEL_LIST[t] for t in ex['ner_tags'][:10]]}")
