import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─── Label definitions ────────────────────────────────────────────────────────
LABEL_LIST = ["O", "B-TOTAL", "I-TOTAL", "B-DATE", "I-DATE",
              "B-VENDOR", "I-VENDOR", "B-ID", "I-ID"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ─── Heuristics & Keywords ───────────────────────────────────────────────────
TOTAL_KEYWORDS = ["total", "subtotal", "grand total", "net", "amount due", "payable"]
IGNORE_KEYWORDS = ["cash", "change", "tax", "vat", "disc", "promo", "saving", "bal", "due"]
NOISE_WORDS = ["receipt", "invoice", "terminal", "welcome", "thank you", "cashier", "date", "time", "order", "reprint"]

def clean_text(text: str) -> str:
    """Normalize OCR text: merge lines, remove excessive whitespace/symbols."""
    if not text: return ""
    # Normalize unicode and whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # Merge broken lines into a single list while keeping layout for vendor detection
    lines = [L.strip() for L in text.split('\n') if L.strip()]
    return "\n".join(lines)

def parse_amount(text: str) -> Optional[float]:
    """Robust currency parser with thousands-separator disambiguation.
    Handles both Western (1,234.56) and Indonesian (1.234,56 or 680.500) formats.
    """
    if not text: return None
    try:
        clean = re.sub(r'[^\d.,]', '', text)
        if not clean: return None

        has_dot = '.' in clean
        has_comma = ',' in clean

        if has_dot and has_comma:
            # Both separators present — determine which is decimal
            last_dot = clean.rfind('.')
            last_comma = clean.rfind(',')
            if last_dot > last_comma:
                # Western: 1,234.56 — comma is thousands, dot is decimal
                clean = clean.replace(',', '')
            else:
                # Indonesian: 1.234,56 — dot is thousands, comma is decimal
                clean = clean.replace('.', '').replace(',', '.')
        elif has_dot and not has_comma:
            parts = clean.split('.')
            # All parts after the first have exactly 3 digits → thousands separators
            # e.g. 680.500 or 1.234.567
            if all(len(p) == 3 for p in parts[1:]):
                clean = clean.replace('.', '')  # strip all dots → integer
            elif len(parts) == 2 and len(parts[1]) == 2:
                pass  # decimal: 12.50
            else:
                clean = clean.replace('.', '')  # fallback: treat as thousands
        elif has_comma and not has_dot:
            parts = clean.split(',')
            if all(len(p) == 3 for p in parts[1:]):
                clean = clean.replace(',', '')  # thousands: 680,500
            else:
                clean = clean.replace(',', '.')  # decimal: 680,5

        return float(clean)
    except:
        return None

def rule_based_extract(text: str) -> Dict[str, Any]:
    """Smart extraction following specific priority rules for Vendor and Date (Rule 2 & 3)."""
    result: Dict[str, Any] = {"vendor": "", "date": "", "total_amount": 0.0}
    lines = text.split('\n')
    
    # ─── VENDOR EXTRACTION (Rule 2) ───
    # Focus on top 5 lines for candidate vendor names
    vendor_candidates = []
    print(f"[DEBUG] Analyzing Top Lines for Vendor: {lines[:5]}")
    for line in lines[:5]:
        L = line.strip()
        # Ignore if too short, contains dates/years, or noise keywords
        if len(L) < 4: continue
        if any(w in L.lower() for w in NOISE_WORDS): continue
        if re.search(r'\d{4,}', L): continue # Skip if has 4+ digits (likely address/year)
        
        # Check alphabet density (Rule 2: Filter lines with mostly alphabets)
        alphas = sum(1 for c in L if c.isalpha())
        if alphas / len(L) > 0.4:
            vendor_candidates.append(L)
    
    # Pick the longest meaningful text (Rule 2)
    if vendor_candidates:
        result["vendor"] = max(vendor_candidates, key=len)
        print(f"[DEBUG] Vendor Candidates Found: {vendor_candidates} -> Selected: {result['vendor']}")

    # ─── DATE EXTRACTION (Rule 3 & 4) ───
    date_patterns = [
        r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',       # 25/12/2023 or 25.12.23
        r'(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})',       # 2023/12/25
        r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})' # 25 Dec 2023
    ]
    
    date_keywords = ["date", "tgl", "transaction", "time", "tanggal"]
    found_dates = []
    
    # Contextual Search: Look for keywords near potential matches
    for pattern in date_patterns:
        matches = re.finditer(pattern, text, re.I)
        for m in matches:
            d_str = m.group(1)
            # Validation Rule 3: Must be complete (not just '2-')
            if len(d_str) < 6: continue
            
            # Check context window (+/- 40 chars) for keywords
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            window = text[start:end].lower()
            
            prio = 0
            if any(kw in window for kw in date_keywords): prio = 1
            found_dates.append((d_str, prio))

    if found_dates:
        # Prefer dates with keywords nearby
        found_dates.sort(key=lambda x: x[1], reverse=True)
        result["date"] = found_dates[0][0].replace('.', '/')
        print(f"[DEBUG] Found Dates: {found_dates} -> Selected: {result['date']}")

    # ─── TOTAL AMOUNT (Prioritised Heuristic) ───
    # Matches: 680,500 / 680.500 / 680500 / 1,234.56 / 1.234,56
    price_pattern = re.compile(
        r'(?<![\d])(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{2})?|\d{1,3}(?:[.,]\d{3})+|\d+[.,]\d{2})(?![\d])'
    )

    # Score each candidate: prefer lines containing "total" (not "subtotal")
    scored_matches = []  # list of (score, value)
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in TOTAL_KEYWORDS):
            if any(ik in line_lower for ik in IGNORE_KEYWORDS): continue
            # Exact "total" match scores higher than "subtotal"
            score = 2 if re.search(r'\btotal\b', line_lower) else 1
            matches = price_pattern.findall(line)
            for m in matches:
                val = parse_amount(m)
                if val and val > 0:
                    scored_matches.append((score, val))
                    print(f"[DEBUG] Total candidate: '{m}' → {val} (score={score}) from line: '{line.strip()}'")

    if scored_matches:
        # Pick highest score first, then highest value among ties
        scored_matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
        result["total_amount"] = scored_matches[0][1]
        print(f"[DEBUG] Selected total_amount: {result['total_amount']}")
    else:
        # Fallback: largest plausible number in the whole document
        all_nums = [parse_amount(m) for m in price_pattern.findall(text)]
        plausible = [v for v in all_nums if v is not None and 1.0 < v < 100_000_000]
        if plausible:
            result["total_amount"] = max(plausible)
            print(f"[DEBUG] Fallback total_amount (max in doc): {result['total_amount']}")

    return result

# ─── Training Support (CORD Dataset Processor) ──────────────────────────

CATEGORY_TO_FIELD = {
    "total.total_price": "TOTAL", "total.total_etc": "TOTAL", "total.cashprice": "TOTAL",
    "sub_total.subtotal_price":"TOTAL", "sub_total.tax_price": "TOTAL",
    "sub_total.etc": "VENDOR", # Often contains store name in CORD
}

def get_field_for_category(category: str) -> Optional[str]:
    return CATEGORY_TO_FIELD.get(category)

def load_cord_split(split_dir: str, max_samples: int = -1) -> List[Dict]:
    ann_dir = Path(split_dir) / "json"
    if not ann_dir.exists(): return []
    examples = []
    files = sorted(ann_dir.glob("*.json"))
    for json_file in files:
        if max_samples > 0 and len(examples) >= max_samples: break
        with open(json_file, encoding="utf-8") as f:
            try: data = json.load(f)
            except: continue
        tokens, ner_tags = [], []
        valid_lines = data.get("valid_line", [])
        for line in valid_lines:
            cat = line.get("category", "O")
            field = get_field_for_category(cat)
            words = line.get("words", [])
            for i, w_obj in enumerate(words):
                text = w_obj.get("text", "").strip()
                if not text: continue
                tokens.append(text)
                if field == "TOTAL": tag = "B-TOTAL" if i == 0 else "I-TOTAL"
                elif field == "VENDOR": tag = "B-VENDOR" if i == 0 else "I-VENDOR"
                else: tag = "O"
                ner_tags.append(LABEL2ID.get(tag, 0))
        if tokens: examples.append({"id": json_file.stem, "tokens": tokens, "ner_tags": ner_tags})
    return examples

def load_and_process_cord(data_dir: str = "data", max_train: int = 400, max_val: int = 50, max_test: int = 50):
    cord_path = Path(data_dir) / "cord_dataset" / "CORD"
    output_dir = Path(data_dir)
    splits = {"train": ("train", max_train), "val": ("dev", max_val), "test": ("test", max_test)}
    results = {}
    for out_name, (folder, limit) in splits.items():
        results[out_name] = load_cord_split(str(cord_path / folder), max_samples=limit)
        fname = "val.json" if out_name == "val" else f"{out_name}.json"
        with open(output_dir / fname, "w") as f: json.dump(results[out_name], f, indent=2)
    return results["train"], results["val"], results["test"]

if __name__ == "__main__":
    print("AI Data Processor: Active.")
