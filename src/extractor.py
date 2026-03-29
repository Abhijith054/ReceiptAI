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
        
        # Check if we should skip to avoid OOM
        self.skip_local = os.environ.get("SKIP_LOCAL_MODEL", "0") == "1"

    def _try_load_model(self):
        """Load your fine-tuned DistilBERT model. Lazy-called if needed."""
        if self._model_loaded: return
        if self.skip_local:
            print("[Extractor] SKIP_LOCAL_MODEL=1 detected. Using Regex/AI fallback flow.")
            return

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

            print(f"[Extractor] Initializing Fine-tuned NER model (Memory-intensive)…")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            # Hotfix: Ensure token_type_ids are not enabled for DistilBERT
            if self._tokenizer and hasattr(self._tokenizer, "model_input_names"):
                if "token_type_ids" in self._tokenizer.model_input_names:
                    self._tokenizer.model_input_names.remove("token_type_ids")

            # Load with minimal settings to save memory
            self._model = AutoModelForTokenClassification.from_pretrained(
                self.model_dir,
                low_cpu_mem_usage=True
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
            print(f"[Extractor] Load failed (likely OOM or missing dependencies): {e}")
            self.skip_local = True

    def _extract_with_groq(self, text: str) -> Optional[Dict]:
        """Use High-Performance Groq LLM (Llama-3) for extraction if local model fails."""
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key or "YOUR_GROQ" in api_key.upper():
            return None

        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            
            prompt = f"""Extract structured data from this receipt OCR text.
Return ONLY a JSON object with these EXACT keys:
{{
  "vendor": "Store name string (clean and title case)",
  "date": "YYYY-MM-DD or null if not found",
  "total_amount": numerical_value (float/int)
}}

Rules:
1. No currency symbols.
2. If total is not found, return 0.
3. Output MUST be valid JSON.

OCR Text:
{text[:2000]}

JSON Output:"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content
            return json.loads(raw)
        except Exception as e:
            print(f"[Extractor] Groq extraction failed: {e}")
            return None

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
        Final Production Extraction Pipeline:
        1. Normalization
        2. Local Model (Primary if loaded)
        3. Smart Regex (Secondary)
        4. Groq LLM (High-Confidence Fallback)
        5. Validation & Strict Formatting
        """
        # Trigger lazy load
        self._try_load_model()
        
        from src.data_processor import normalize_ocr, regex_extract
        
        # 0. Normalize & Log
        clean_text = normalize_ocr(text)
        print(f"[Extractor] Starting extraction for OCR length: {len(text)}")
        
        # Initialize result with defaults
        res = {"vendor": None, "date": None, "total_amount": None}
        method = "regex"

        # 1. Local Model Attempt (BERT)
        if self._model_loaded:
            try:
                ner_out = self._pipeline(text[:1024])
                bert_res = self._format_ner_results(ner_out)
                # Map old keys to new keys
                res["vendor"] = bert_res.get("vendor_name")
                res["date"] = bert_res.get("date")
                res["total_amount"] = bert_res.get("total_amount")
                method = "Trained DistilBERT (NER)"
            except Exception as e:
                print(f"[Extractor] BERT failed: {e}")

        # 2. Smart Regex Attempt
        reg_res = regex_extract(text)
        # Fill gaps from regex
        if not res["vendor"]: res["vendor"] = reg_res.get("vendor")
        if not res["date"]: res["date"] = reg_res.get("date")
        if not res["total_amount"]: res["total_amount"] = reg_res.get("total_amount")

        # 3. Validation Layer
        def is_valid(data):
            try:
                total = float(data.get("total_amount") or 0)
                return total > 0 and data.get("vendor") and data.get("date")
            except: return False

        # 4. LLM Fallback (Groq) - Triggered if data is invalid or missing
        if not is_valid(res):
            print("[Extractor] Confidence low. Falling back to Groq AI…")
            groq_res = self._extract_with_groq(text)
            if groq_res:
                # Prioritize Groq's high-intelligence results
                res["vendor"] = groq_res.get("vendor_name") or groq_res.get("vendor")
                res["date"] = groq_res.get("date")
                res["total_amount"] = groq_res.get("total_amount")
                method = "Groq High-Intelligence AI"

        # 5. Final Sanitization
        try:
            # Ensure total_amount is a number (number type in JSON)
            if res["total_amount"]:
                # strip non-numeric or currency symbols if any left
                if isinstance(res["total_amount"], str):
                    s = re.sub(r'[^\d.]', '', res["total_amount"])
                    res["total_amount"] = float(s) if s else 0.0
                else:
                    res["total_amount"] = float(res["total_amount"])
            else:
                res["total_amount"] = 0.0
        except:
            res["total_amount"] = 0.0

        print(f"[Extractor] Final Result: {res} | Method: {method}")
        
        return {
            "vendor": res["vendor"] or "Unknown Store",
            "date": res["date"], # LLM or regex will return YYYY-MM-DD or None
            "total_amount": res["total_amount"],
            "raw_text": text,
            "method": method
        }


# Singleton
_extractor_instance: Optional[ReceiptExtractor] = None

def get_extractor() -> ReceiptExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = ReceiptExtractor()
    return _extractor_instance
