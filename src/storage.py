"""
Storage layer for extracted receipt records.
Uses JSON-Lines format (one JSON object per line) in data/extracted_records.jsonl.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

RECORDS_FILE = "data/extracted_records.jsonl"


class RecordStorage:
    """Persist and retrieve extracted receipt records."""

    def __init__(self, records_file: str = RECORDS_FILE):
        self.records_file = Path(records_file)
        self.records_file.parent.mkdir(parents=True, exist_ok=True)
        # In-memory cache
        self._cache: Dict[str, Dict] = {}
        self._load_all()

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _load_all(self):
        """Load all records from disk into the in-memory cache."""
        self._cache = {}
        if self.records_file.exists():
            with open(self.records_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            self._cache[record["doc_id"]] = record
                        except (json.JSONDecodeError, KeyError):
                            pass

    def _flush(self):
        """Write all in-memory records back to disk."""
        with open(self.records_file, "w") as f:
            for record in self._cache.values():
                f.write(json.dumps(record) + "\n")

    # ─── Public API ──────────────────────────────────────────────────────────

    def save_record(
        self,
        extracted: Dict,
        doc_id: Optional[str] = None,
        filename: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict:
        """
        Save an extracted record.

        Args:
            extracted: Dict from extractor.extract()
            doc_id: optional custom ID; auto-generated if None
            filename: original uploaded filename (for display)
            session_id: the active chat session ID this document belongs to

        Returns:
            The saved record dict (with doc_id and timestamp added).
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8].upper()

        record = {
            "doc_id": doc_id,
            "session_id": session_id or "default_session",
            "filename": filename or f"receipt_{doc_id}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "extracted": {
                "total_amount": extracted.get("total_amount"),
                "date": extracted.get("date"),
                "vendor_name": extracted.get("vendor_name"),
                "receipt_id": extracted.get("receipt_id"),
            },
            "raw_text": extracted.get("raw_text", ""),
            "method": extracted.get("method", "unknown"),
        }

        self._cache[doc_id] = record
        self._flush()
        return record

    def get_record(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a single record by doc_id."""
        return self._cache.get(doc_id)

    def list_all(self, limit: int = 100, session_id: Optional[str] = None) -> List[Dict]:
        """Return all records, newest first, optionally filtered by session_id."""
        records = list(self._cache.values())
        if session_id:
            records = [r for r in records if r.get("session_id") == session_id]
        records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return records[:limit]

    def delete_record(self, doc_id: str) -> bool:
        """Delete a record by doc_id."""
        if doc_id in self._cache:
            del self._cache[doc_id]
            self._flush()
            return True
        return False

    def search(self, query: str) -> List[Dict]:
        """
        Simple fuzzy search over vendor name, date, total.
        Used by the QA engine to locate relevant documents.
        """
        query_lower = query.lower()
        results = []
        for record in self._cache.values():
            ex = record.get("extracted", {})
            haystack = " ".join(
                str(v) for v in ex.values() if v
            ).lower()
            if query_lower in haystack or any(
                w in haystack for w in query_lower.split()
                if len(w) > 2
            ):
                results.append(record)
        return results


# Singleton
_storage_instance: Optional[RecordStorage] = None


def get_storage() -> RecordStorage:
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = RecordStorage()
    return _storage_instance
