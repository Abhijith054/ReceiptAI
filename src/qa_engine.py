import json
import os
import re
from typing import Dict, List, Optional

from src.storage import get_storage, RecordStorage

# ─── LLM-based QA (Groq) ────────────────────────────────────────────────────

def groq_answer(question: str, records: List[Dict], api_key: str) -> str:
    """Use Groq LLM to answer the question, grounded in extracted data."""
    try:
        from groq import Groq
        
        # Format the context from multiple records
        context_parts = []
        for r in records:
            ex = r.get("extracted", {})
            context_parts.append(
                f"Document ID: {r['doc_id']}\n"
                f"Vendor: {ex.get('vendor_name', 'Unknown')}\n"
                f"Date: {ex.get('date', 'Unknown')}\n"
                f"Total Amount: {ex.get('total_amount', 'Unknown')}\n"
                f"Receipt ID: {ex.get('receipt_id', 'Unknown')}\n"
                f"Filename: {r.get('filename', 'Unknown')}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        prompt = f"""You are a high-performance ReceiptAI assistant.
Your task is to answer natural language questions based ONLY on the receipt data provided below.

Available Data:
{context}

Guidelines:
1. If the user asks for a specific vendor or date, focus ONLY on those records.
2. If the user asks for a sum (e.g. "Total spent"), calculate it accurately from the available data.
3. If the data is missing, state it clearly.
4. Keep your answer professional, concise, and helpful.

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
        return f"[System Error: {e}] I encountered an issue analyzing the database. Please check your API key."

# ─── Rule-based Fallback ───────────────────────────────────────────────────

def rule_based_answer(question: str, record: Dict) -> str:
    """Fallback answer using simple pattern matching if no LLM key is set."""
    ex = record.get("extracted", {})
    q = question.lower()
    
    if "total" in q or "much" in q:
        val = ex.get("total_amount")
        return f"The total amount on this receipt is **{val}**." if val else "Total amount not found."
    elif "date" in q:
        val = ex.get("date")
        return f"This receipt is dated **{val}**." if val else "Date not found."
    elif "vendor" in q or "where" in q:
        val = ex.get("vendor_name")
        return f"The vendor is **{val}**." if val else "Vendor not found."
    
    return "I found this receipt but I'm not sure what you're asking. Try asking about the 'total', 'date', or 'vendor'."


# ─── Main QA Engine ──────────────────────────────────────────────────────────

class QAEngine:
    def __init__(self, storage: Optional[RecordStorage] = None):
        self.storage = storage or get_storage()
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.use_llm = bool(self.groq_api_key and "YOUR_GROQ" not in self.groq_api_key.upper())
        
        if self.use_llm:
            print("[QA] Intelligence Engine: ACTIVE (Using Groq LLM)")
        else:
            print("[QA] Intelligence Engine: FALLBACK (Using Rule-based matches)")

    def answer(self, question: str, doc_id: Optional[str] = None) -> Dict:
        """
        Smart Answer Engine with RAG (Retrieval-Augmented Generation).
        """
        # Phase 1: Context Retrieval
        records_to_query = []
        
        if doc_id:
            # Single doc focus (User click)
            record = self.storage.get_record(doc_id)
            if record:
                records_to_query = [record]
        else:
            # Global search (Smart RAG)
            # Try to find relevant docs by searching for words in the question
            # (e.g. if question contains "Walmart", find Walmart receipts)
            keywords = [w.strip() for w in question.split() if len(w) > 3]
            search_results = []
            for kw in keywords:
                search_results.extend(self.storage.search(kw))
            
            # De-duplicate results
            seen_ids = set()
            for r in search_results:
                if r["doc_id"] not in seen_ids:
                    records_to_query.append(r)
                    seen_ids.add(r["doc_id"])
            
            # If no matches found via keyword, default to last 10 receipts
            if not records_to_query:
                records_to_query = self.storage.list_all(limit=10)

        # Phase 2: Generating the Answer
        if not records_to_query:
            return {
                "answer": "I couldn't find any documents to analyze. Please upload a receipt first!",
                "sources": []
            }

        if self.use_llm:
            answer = groq_answer(question, records_to_query, self.groq_api_key)
        else:
            # Rule based only supports single doc logic currently
            answer = rule_based_answer(question, records_to_query[0])
            if len(records_to_query) > 1:
                answer += f"\n\n(I analyzed document '{records_to_query[0].get('filename')}' out of {len(records_to_query)} available.)"

        return {
            "answer": answer,
            "sources": [r.get("filename", r["doc_id"]) for r in records_to_query[:5]]
        }


# Singleton Pattern
_qa_engine: Optional[QAEngine] = None

def get_qa_engine() -> QAEngine:
    global _qa_engine
    if _qa_engine is None:
        _qa_engine = QAEngine()
    return _qa_engine
