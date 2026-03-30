import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Local imports
from src import data_processor
from src.data_processor import (
    clean_text,
    rule_based_extract
)

MODEL_DIR = "models/receipt-bert-v1"

class ReceiptExtractor:
    """
    Hybrid Intelligent Extractor.
    - Uses local fine-tuned BERT for initial data retrieval.
    - Uses Llama-3 (Groq) for advanced zero-shot refinement if needed.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        self._pipeline = None
        
        # Check if we should skip to avoid OOM
        self.skip_local = os.environ.get("SKIP_LOCAL_MODEL", "0") == "1"
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.use_groq = bool(self.groq_api_key and "YOUR_GROQ" not in self.groq_api_key.upper())
        
        if self.use_groq:
            print("[Extractor] Intelligence Mode: HYBRID (Groq Enabled)")
        else:
            print("[Extractor] Intelligence Mode: LOCAL (BERT/Regex Mode)")

    def _try_load_model(self):
        """Load fine-tuned DistilBERT model. Lazy-called only when available."""
        if self._model_loaded: return
        if self.skip_local:
            print("[Extractor] Local model skipped (SKIP_LOCAL_MODEL=1). Using Groq + rules.")
            return

        # Guard: torch/transformers may not be installed (e.g. HF free tier)
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        except ImportError:
            print("[Extractor] torch/transformers not installed. Using Groq + rule-based extraction.")
            self.skip_local = True
            return

        try:
            model_path = Path(self.model_dir)
            if not model_path.exists() or not (model_path / "config.json").exists():
                print(f"[Extractor] Model directory '{model_path}' not found. Using Groq + rules.")
                self.skip_local = True
                return

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            raw_model = AutoModelForTokenClassification.from_pretrained(
                self.model_dir,
                low_cpu_mem_usage=True,
            )
            self._model = torch.quantization.quantize_dynamic(
                raw_model, {torch.nn.Linear}, dtype=torch.qint8
            )
            self._pipeline = pipeline(
                "ner",
                model=self._model,
                tokenizer=self._tokenizer,
                aggregation_strategy="simple",
                device=-1,  # CPU only
            )
            self._model_loaded = True
            print("[Extractor] BERT Model System: ONLINE ✓")
        except Exception as e:
            print(f"[Extractor] Error loading local model: {e}. Falling back to rules.")
            self.skip_local = True

    def _extract_with_groq(self, text: str) -> Optional[Dict]:
        """Use High-Performance Groq LLM (Llama-3) for extraction fallback (Rule 5)."""
        if not self.use_groq: return None

        try:
            from groq import Groq
            # Apply strict 5-second timeout to prevent stalling the backend
            client = Groq(api_key=self.groq_api_key, timeout=5.0)
            
            # Specific Prompt (Rule 5)
            prompt = f"""
            Extract vendor name and date from this receipt.
            Return JSON:
            {{
              "vendor": "string",
              "date": "YYYY-MM-DD or null"
            }}
            
            OCR Text:
            ---
            {text[:2500]}
            ---
            """

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                # response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content
            # Clean potential markdown wrap
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            return json.loads(raw)
        except Exception as e:
            print(f"[Extractor] Groq Fallback Failed: {e}")
            return None

    def extract(self, ocr_text: str) -> Dict[str, Any]:
        """Hybrid extraction entry point with robust validation (Rule 6)."""
        text = clean_text(ocr_text)
        print(f"[DEBUG] Raw OCR (Normalized):\n{text[:500]}...") # Rule 8
        
        res: Dict[str, Any] = {"vendor": "", "date": "", "total_amount": 0.0, "raw_text": text}
        method = "Heuristic (Regex/Rules)"

        # 1. Primary: Fine-tuned DistilBERT (NER)
        self._try_load_model()
        if self._model_loaded:
            try:
                ner_results = self._pipeline(text)
                format_res = self._format_ner_results(ner_results)
                if format_res.get("vendor"): res["vendor"] = format_res["vendor"]
                if format_res.get("date"): res["date"] = format_res["date"]
                if format_res.get("total_amount"): res["total_amount"] = format_res["total_amount"]
                method = "Model (BERT)"
            except: pass

        # 2. Rule-based Extraction (Fallback)
        reg_res = rule_based_extract(text)
        print(f"[DEBUG] Heuristic Candidates - Vendor: '{reg_res.get('vendor')}', Date: '{reg_res.get('date')}'") # Rule 8
        
        # Merge if models got junk
        if not res.get("vendor") or self._is_invalid_vendor(res["vendor"]):
            if reg_res.get("vendor"): res["vendor"] = reg_res["vendor"]
            
        if not res.get("date") or self._is_invalid_date(res["date"]):
            if reg_res.get("date"): res["date"] = reg_res["date"]

        # Always prefer the larger total: rule-based regex is more reliable for
        # currency amounts than BERT, which can pick up wrong small numbers.
        rule_total = reg_res.get("total_amount") or 0.0
        bert_total = res.get("total_amount") or 0.0
        if rule_total > bert_total:
            res["total_amount"] = rule_total
            print(f"[Extractor] Rule-based total ({rule_total}) overrides BERT total ({bert_total})")

        # 3. Validation Layer & LLM Fallback (Rule 5 & 6)
        is_invalid_v = self._is_invalid_vendor(res.get("vendor"))
        is_invalid_d = self._is_invalid_date(res.get("date"))
        
        if self.use_groq and (is_invalid_v or is_invalid_d):
            print(f"[Extractor] Validation failed (V:{is_invalid_v}, D:{is_invalid_d}). Triggering LLM Fallback...")
            groq_res = self._extract_with_groq(text)
            if groq_res:
                if is_invalid_v and groq_res.get("vendor"):
                    res["vendor"] = groq_res["vendor"]
                if is_invalid_d and groq_res.get("date"):
                    res["date"] = groq_res["date"]
                method += " + AI Healing"

        res["method"] = method
        return res

    def _is_invalid_vendor(self, vendor: Any) -> bool:
        """Rule 6: Must be > 3 chars and not all numeric."""
        v = str(vendor or "").strip()
        if len(v) <= 3: return True
        if v.isdigit(): return True
        return False

    def _is_invalid_date(self, date: Any) -> bool:
        """Rule 6: Must be complete format (> 8 chars, with separators)."""
        d = str(date or "").strip()
        if not d or d.lower() == "null": return True
        if len(d) < 8: return True
        if not any(c in d for c in ['-', '/', '.']): return True
        return False

    def _format_ner_results(self, ner_results: List[Dict]) -> Dict:
        """Helper to parse BERT spans."""
        res = {"total_amount": 0.0, "vendor": "", "date": ""}
        for span in ner_results:
            group = span.get("entity_group", "").upper()
            word = span.get("word", "").replace(" ##", "").strip()
            if "TOTAL" in group:
                try:
                    val = data_processor.parse_amount(word)
                    if val and val > res["total_amount"]: res["total_amount"] = val
                except: pass
            elif "ORG" in group or "VENDOR" in group:
                if not res["vendor"]: res["vendor"] = word
            elif "DATE" in group:
                if not res["date"]: res["date"] = word
        return res

# Global Singleton
_extractor_instance = None

def get_extractor() -> ReceiptExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = ReceiptExtractor()
    return _extractor_instance
