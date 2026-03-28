"""
Question Answering engine.
Uses Groq LLM (free tier) or a rule-based fallback to answer
natural language questions grounded in extracted receipt data.
"""

import json
import os
import re
from typing import Dict, List, Optional

from src.storage import get_storage, RecordStorage


# ─── Rule-based QA ───────────────────────────────────────────────────────────

QUESTION_PATTERNS = {
    "total": [
        r"\btotal\b", r"amount", r"how much", r"price", r"cost", r"pay",
        r"grand total", r"sum",
    ],
    "date": [r"\bdate\b", r"when", r"day", r"time", r"month", r"year"],
    "vendor": [
        r"vendor", r"store", r"shop", r"restaurant", r"merchant",
        r"where", r"name", r"company",
    ],
    "receipt_id": [r"id\b", r"receipt number", r"invoice", r"number", r"#"],
}


def _match_field(question: str) -> Optional[str]:
    q = question.lower()
    for field, patterns in QUESTION_PATTERNS.items():
        for p in patterns:
            if re.search(p, q):
                return field
    return None


def rule_based_answer(question: str, record: Dict) -> str:
    """Answer a question about a record using simple pattern matching."""
    ex = record.get("extracted", {})
    field = _match_field(question)

    if field == "total":
        val = ex.get("total_amount")
        return f"The total amount is **{val}**." if val else "Total amount not found in this receipt."
    elif field == "date":
        val = ex.get("date")
        return f"The receipt date is **{val}**." if val else "Date not found in this receipt."
    elif field == "vendor":
        val = ex.get("vendor_name")
        return f"The vendor is **{val}**." if val else "Vendor name not found in this receipt."
    elif field == "receipt_id":
        val = ex.get("receipt_id")
        return f"The receipt ID is **{val}**." if val else "Receipt ID not found."
    else:
        # Generic: return all extracted fields
        parts = []
        if ex.get("vendor_name"):
            parts.append(f"Vendor: {ex['vendor_name']}")
        if ex.get("date"):
            parts.append(f"Date: {ex['date']}")
        if ex.get("total_amount"):
            parts.append(f"Total: {ex['total_amount']}")
        if ex.get("receipt_id"):
            parts.append(f"Receipt ID: {ex['receipt_id']}")
        if parts:
            return "Here is what I found:\n" + "\n".join(f"• {p}" for p in parts)
        return "I could not find relevant information in this receipt."


# ─── LLM-based QA (Groq) ────────────────────────────────────────────────────

def groq_answer(question: str, record: Dict, api_key: str) -> str:
    """Use Groq LLM to answer the question, grounded in extracted data."""
    try:
        from groq import Groq

        ex = record.get("extracted", {})
        context = json.dumps(ex, indent=2)

        prompt = f"""You are a helpful assistant that answers questions about receipt data.
You have access to the following extracted receipt information:

{context}

Answer the user's question based ONLY on the information above.
If the information is not available, say so clearly.
Be concise and direct.

Question: {question}
Answer:"""

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[LLM error: {e}] Falling back to rule-based answer.\n\n" + rule_based_answer(question, record)


# ─── Multi-document QA ───────────────────────────────────────────────────────

def groq_multi_answer(question: str, records: List[Dict], api_key: str) -> str:
    """Answer a question across multiple receipt records."""
    try:
        from groq import Groq

        summaries = []
        for r in records[:5]:  # limit context
            ex = r.get("extracted", {})
            summaries.append(
                f"[{r['doc_id']}] Vendor: {ex.get('vendor_name','?')}, "
                f"Date: {ex.get('date','?')}, Total: {ex.get('total_amount','?')}"
            )

        context = "\n".join(summaries)
        prompt = f"""You are a helpful assistant that analyzes multiple receipts.

Available receipts:
{context}

Answer the following question based ONLY on the receipt data above.
Be concise and direct.

Question: {question}
Answer:"""

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[LLM error: {e}]"


# ─── Main QA Engine ──────────────────────────────────────────────────────────

class QAEngine:
    def __init__(self, storage: Optional[RecordStorage] = None):
        self.storage = storage or get_storage()
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.use_llm = bool(self.groq_api_key and self.groq_api_key != "your_groq_api_key_here")
        if self.use_llm:
            print("[QA] Using Groq LLM for question answering.")
        else:
            print("[QA] Using rule-based QA (set GROQ_API_KEY for LLM answers).")

    def answer(
        self,
        question: str,
        doc_id: Optional[str] = None,
    ) -> Dict:
        """
        Answer a natural language question.

        Args:
            question: the user's question
            doc_id: specific document to query; if None, queries all documents

        Returns:
            {"answer": str, "doc_id": str|None, "sources": list}
        """
        if doc_id:
            record = self.storage.get_record(doc_id)
            if not record:
                return {
                    "answer": f"No document found with ID {doc_id}.",
                    "doc_id": doc_id,
                    "sources": [],
                }
            if self.use_llm:
                answer = groq_answer(question, record, self.groq_api_key)
            else:
                answer = rule_based_answer(question, record)
            return {
                "answer": answer,
                "doc_id": doc_id,
                "sources": [record.get("filename", doc_id)],
            }

        # No doc_id → query all documents
        all_records = self.storage.list_all()
        if not all_records:
            return {
                "answer": "No documents have been processed yet. Please upload a receipt first.",
                "doc_id": None,
                "sources": [],
            }

        if len(all_records) == 1:
            record = all_records[0]
            if self.use_llm:
                answer = groq_answer(question, record, self.groq_api_key)
            else:
                answer = rule_based_answer(question, record)
            return {
                "answer": answer,
                "doc_id": record["doc_id"],
                "sources": [record.get("filename", record["doc_id"])],
            }

        # Multiple docs
        if self.use_llm:
            answer = groq_multi_answer(question, all_records, self.groq_api_key)
        else:
            # Summarize across all documents
            lines = [f"Found {len(all_records)} receipts:"]
            for r in all_records[:10]:
                ex = r.get("extracted", {})
                lines.append(
                    f"• [{r['doc_id']}] {ex.get('vendor_name','Unknown')} – "
                    f"{ex.get('date','N/A')} – Total: {ex.get('total_amount','N/A')}"
                )
            answer = "\n".join(lines)

        return {
            "answer": answer,
            "doc_id": None,
            "sources": [r.get("filename", r["doc_id"]) for r in all_records[:5]],
        }


# Singleton
_qa_engine: Optional[QAEngine] = None


def get_qa_engine() -> QAEngine:
    global _qa_engine
    if _qa_engine is None:
        _qa_engine = QAEngine()
    return _qa_engine
