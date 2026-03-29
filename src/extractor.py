"""
Receipt Intelligence Engine: 
1. Extraction: Uses your FINE-TUNED DistilBERT model (trained on CORD).
2. Chat: Uses LLM (Groq) for high-reasoning natural language queries.
"""

import json
import os
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
    Hybrid Intelligent Extractor.
    - Uses local fine-tuned BERT for initial data retrieval.
    - Uses Llama-3 (Groq) for advanced chat reasoning.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        self._pipeline = None
        self._try_load_model()

    def _try_load_model(self):
        """Load your fine-tuned DistilBERT model from the models/ folder."""
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForTokenClassification,
                pipeline,
            )

            model_path = Path(self.model_dir)
            if not model_path.exists() or not (model_path / "config.json").exists():
                print(f"[Extractor] Warning: Fine-tuned model not found at {self.model_dir}.")
                return

            print(f"[Extractor] Initializing Fine-tuned NER model…")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            # Hotfix: Ensure token_type_ids are not enabled for DistilBERT
            if self._tokenizer and hasattr(self._tokenizer, "model_input_names"):
                if "token_type_ids" in self._tokenizer.model_input_names:
                    self._tokenizer.model_input_names.remove("token_type_ids")

            self._model = AutoModelForTokenClassification.from_pretrained(
                self.model_dir, 
                low_cpu_mem_usage=True,
                # use_safetensors=True is default if available
            )
            self._pipeline = pipeline(
                "token-classification",
                model=self._model,
                tokenizer=self._tokenizer,
                aggregation_strategy="simple",
            )
            self._model_loaded = True
            print("[Extractor] Trained Model System: ONLINE ✓")

        except Exception as e:
            print(f"[Extractor] Load failed: {e}")

    @staticmethod
    def _clean_ner_val(text: str) -> str:
        """Sanitize word-piece remnants and artifacts from BERT spans."""
        return text.replace(" ##", "").strip()

    def _format_ner_results(self, ner_results: List[Dict]) -> Dict:
        """Convert BERT spans into structured receipt fields."""
        fields = {"total_amount": None, "date": None, "vendor_name": None, "receipt_id": None}
        
        for span in ner_results:
            group = span.get("entity_group", "").upper()
            word = self._clean_ner_val(span.get("word", ""))
            
            if not word: continue
            
            if "TOTAL" in group:
                fields["total_amount"] = word
            elif "DATE" in group:
                fields["date"] = word
            elif "VENDOR" in group:
                fields["vendor_name"] = word
            elif "ID" in group:
                fields["receipt_id"] = word
        return fields

    def extract(self, text: str) -> Dict:
        """
        Primary Extraction Loop:
        1. Trained DistilBERT Model (Primary Intelligence)
        2. Regex (Precision Fallback)
        """
        if not text:
            return {"total_amount": None, "date": None, "vendor_name": None, "receipt_id": None, "method": "empty"}

        # Phase 1: Use your Trained Model
        res = {}
        method = "regex"
        
        if self._model_loaded:
            try:
                ner_out = self._pipeline(text[:512]) # limit context window
                res = self._format_ner_results(ner_out)
                method = "Trained DistilBERT (NER)"
            except Exception:
                pass

        # Phase 2: Precision verification with internal patterns
        fallback = regex_extract(text)
        
        # Merge results: favor model for entities, regex for math-heavy fields
        final = {
            "vendor_name": res.get("vendor_name") or fallback.get("vendor_name"),
            "date": res.get("date") or fallback.get("date"),
            "total_amount": res.get("total_amount") or fallback.get("total_amount"),
            "receipt_id": res.get("receipt_id") or fallback.get("receipt_id"),
            "raw_text": text,
            "method": method
        }
        
        return final


# Singleton
_extractor_instance: Optional[ReceiptExtractor] = None

def get_extractor() -> ReceiptExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = ReceiptExtractor()
    return _extractor_instance
