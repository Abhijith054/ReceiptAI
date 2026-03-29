import json
import os
import re
from typing import Any, Dict, List, Optional

# Local imports
from src.storage import get_storage, RecordStorage

class QAEngine:
    """
    Intelligent Answer Engine with support for:
    - RAG (Retrieval-Augmented Generation) based on extracted receipt data.
    - Conversational Memory (via history).
    - Rule-based fallback if LLM is unavailable.
    """

    def __init__(self, storage: Optional[RecordStorage] = None):
        self.storage = storage or get_storage()
        
        # Check for High-Intelligence Mode (Groq)
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.use_llm = bool(self.groq_api_key and "YOUR_GROQ" not in self.groq_api_key.upper())
        
        if self.use_llm:
            print("[QA] Intelligence Engine: ACTIVE (Using Groq LLM)")
        else:
            print("[QA] Intelligence Engine: FALLBACK (Using Rule-based matches)")

    def answer(self, question: str, doc_id: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Main entry point for answering questions.
        If history is provided, the AI will use it for multi-turn context.
        """
        # Always fetch 10 most recent for global context awareness
        all_recent = self.storage.list_all(limit=10)
        focused_record = self.storage.get_record(doc_id) if doc_id else None
        
        # If no specific doc focused, we just use the global list
        other_records = [r for r in all_recent if r['doc_id'] != doc_id] if focused_record else all_recent

        # Generate the context string for the prompt
        context_parts = []
        if focused_record:
            ex = focused_record.get("extracted", {})
            context_parts.append(f"TARGET DOC (ID {doc_id}): {focused_record.get('filename')}\nData: {json.dumps(ex)}")
        
        summaries = []
        for r in other_records[:5]:
            ex = r.get("extracted", {})
            summaries.append(f"- {r.get('filename')}: Vendor={ex.get('vendor')}, Total={ex.get('total_amount')}")
        if summaries:
            context_parts.append("OTHER RECENT BILLS:\n" + "\n".join(summaries))
        
        context_str = "\n---\n".join(context_parts)

        # Intelligence Handover
        if self.use_llm:
            try:
                from groq import Groq
                client = Groq(api_key=self.groq_api_key)
                
                system_prompt = f"""You are an intelligent ReceiptAI assistant.

GENERAL INSTRUCTIONS:
- If the user greets you, respond politely.
- Otherwise, answer the user’s question strictly using the provided document data.

DATA:
{context_str}

STRICT RULES:
- If asked about "all" documents, summarize totals or vendors from the entire DATA.
- Otherwise, focus on the TARGET DOC.
- If the answer is genuinely missing from the data, say 'Not available'.
- Return a short, direct answer in natural language. Do NOT output raw JSON."""

                messages = [{"role": "system", "content": system_prompt}]
                if history:
                    messages.extend(history[-5:] if len(history) > 5 else history)
                messages.append({"role": "user", "content": question})

                resp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=300
                )
                answer_text = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"[QA] Groq Intelligence Error: {e}")
                answer_text = "I encountered an issue analyzing the database. Falling back to local search..."
        else:
            # Rule based fallback (if no Groq)
            target = focused_record or (all_recent[0] if all_recent else None)
            if target:
                answer_text = f"I analyzed '{target.get('filename')}' and found it is from {target.get('extracted', {}).get('vendor')} for a total amount of {target.get('extracted', {}).get('total_amount')}."
            else:
                answer_text = "No records found in my database yet. Please upload a receipt!"

        return {
            "answer": answer_text,
            "sources": [focused_record.get('filename')] if focused_record else [r.get('filename') for r in all_recent[:2]]
        }

# Global Singleton
_qa_instance = None

def get_qa_engine() -> QAEngine:
    global _qa_instance
    if _qa_instance is None:
        _qa_instance = QAEngine()
    return _qa_instance
