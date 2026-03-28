"""
Information extractor: loads the fine-tuned DistilBERT token classifier
and runs inference to extract key receipt fields.
Falls back to regex extraction if model is not yet trained.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.data_processor import (
    LABEL_LIST, LABEL2ID, ID2LABEL,
    regex_extract,
)

MODEL_DIR = "models/receipt_ner"


class ReceiptExtractor:
    """
    Extracts structured information from receipt text.

    Priority:
    1. Fine-tuned DistilBERT token classifier (if model exists)
    2. Regex fallback (always available)
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        self._pipeline = None
        self._try_load_model()

    def _try_load_model(self):
        """Attempt to load the fine-tuned model (silent if not available)."""
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForTokenClassification,
                pipeline,
            )

            model_path = Path(self.model_dir)
            if not model_path.exists() or not (model_path / "config.json").exists():
                print(f"[Extractor] Model not found at {self.model_dir}. Using regex fallback.")
                return

            print(f"[Extractor] Loading fine-tuned model from {self.model_dir}…")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            # Hotfix: DistilBERT does not expect token_type_ids
            if self._tokenizer and hasattr(self._tokenizer, "model_input_names"):
                if "token_type_ids" in self._tokenizer.model_input_names:
                    self._tokenizer.model_input_names.remove("token_type_ids")

            self._model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
            self._pipeline = pipeline(
                "token-classification",
                model=self._model,
                tokenizer=self._tokenizer,
                aggregation_strategy="simple",
            )
            self._model_loaded = True
            print("[Extractor] Model loaded ✓")

        except Exception as e:
            print(f"[Extractor] Could not load model ({e}). Using regex fallback.")

    @staticmethod
    def _clean_amount(text: str) -> str:
        """Extract the best numeric value from a NER span like 'total amount : $ 36. 30'."""
        # Find all continuous digit sequences that may include commas, dots, or spaces
        text_clean = text.replace(" ", "")
        nums = re.findall(r'\d+(?:[.,]\d+)*', text_clean)
        if not nums:
            return text.strip()
            
        # Filter out numbers that look like phone numbers or IDs:
        # A valid price usually doesn't have 8+ continuous digits without punctuation.
        valid_nums = []
        for n in nums:
            raw_len = len(n.replace(',', '').replace('.', ''))
            has_punct = (',' in n or '.' in n)
            if raw_len < 8 or has_punct:
                valid_nums.append(n)
        
        if valid_nums:
            # Prefer numbers with punctuation if multiple candidates exist
            with_punct = [n for n in valid_nums if ',' in n or '.' in n]
            if with_punct:
                return with_punct[-1]
            return valid_nums[-1]
        
        return nums[-1] if nums else text.strip()



    def _ner_to_fields(self, ner_results: List[Dict]) -> Dict:
        """Convert pipeline NER output to structured receipt fields."""
        fields: Dict[str, Optional[str]] = {
            "total_amount": None,
            "date": None,
            "vendor_name": None,
            "receipt_id": None,
        }

        # Collect spans by entity group
        groups: Dict[str, List[str]] = {}
        for span in ner_results:
            group = span.get("entity_group", span.get("entity", "O"))
            # Normalize: B-TOTAL → TOTAL
            group = group.replace("B-", "").replace("I-", "")
            word = span.get("word", "").replace(" ##", "").strip()
            if word:
                groups.setdefault(group, []).append(word)

        for group, words in groups.items():
            text_val = " ".join(words)
            if group == "TOTAL":
                amt = self._clean_amount(text_val)
                # If the model extracted an amount but it has no price punctuation (commas/decimals),
                # it's likely a false positive (like a terminal ID). We'll set it to None to let 
                # the regex fallback find a real monetary value from the raw text instead.
                if amt and not ('.' in amt or ',' in amt):
                    amt = None
                fields["total_amount"] = amt
            elif group == "DATE":
                fields["date"] = text_val
            elif group == "VENDOR":
                fields["vendor_name"] = text_val
            elif group == "ID":
                fields["receipt_id"] = text_val

        return fields

    def extract(self, text: str) -> Dict:
        """
        Extract key fields from receipt OCR text.

        Returns:
            {
                "total_amount": str | null,
                "date": str | null,
                "vendor_name": str | null,
                "receipt_id": str | null,
                "raw_text": str,
                "method": "model" | "regex"
            }
        """
        if not text or not text.strip():
            return {
                "total_amount": None,
                "date": None,
                "vendor_name": None,
                "receipt_id": None,
                "raw_text": text,
                "method": "empty",
            }

        # Try model first
        if self._model_loaded:
            try:
                ner_results = self._pipeline(text[:512])  # truncate for safety
                fields = self._ner_to_fields(ner_results)
                # Prioritize high-precision Regex for mathematically structured fields
                regex_fields = regex_extract(text)
                
                # These fields are highly deterministic or structural, Regex is much safer
                for key in ("total_amount", "date", "vendor_name", "receipt_id"):
                    if regex_fields[key] is not None:
                        fields[key] = regex_fields[key]
                
                fields["raw_text"] = text
                fields["method"] = "model"
                return fields
            except Exception as e:
                err_msg = str(e)
                print(f"[Extractor] Model inference failed ({e}), falling back to regex.")

        # Regex fallback
        fields = regex_extract(text)
        fields["raw_text"] = text
        fields["method"] = f"REGEX (error: {err_msg})" if "err_msg" in locals() else "REGEX"
        return fields

    def extract_from_cord_sample(self, cord_annotation: Dict) -> Dict:
        """
        Extract from a raw CORD annotation dict (ground_truth JSON).
        Used during evaluation to work from the dataset directly.
        """
        annotation = cord_annotation
        if isinstance(annotation, str):
            try:
                annotation = json.loads(annotation)
            except Exception:
                annotation = {}

        # Build plain text from all valid_line words
        lines = annotation.get("valid_line", [])
        tokens = []
        for line in lines:
            for word in line.get("words", []):
                t = word.get("quad_text", "").strip()
                if t:
                    tokens.append(t)
        text = " ".join(tokens)
        return self.extract(text)


# Singleton instance (lazy initialization)
_extractor_instance: Optional[ReceiptExtractor] = None


def get_extractor() -> ReceiptExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = ReceiptExtractor()
    return _extractor_instance
