import os
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Constants
RECORDS_FILE = "data/extracted_records.jsonl"

class RecordStorage:
    """Simple JSONL-based storage for extracted receipt records."""

    def __init__(self, records_file: str = RECORDS_FILE):
        env_path = os.environ.get("STORAGE_FILE")
        self.records_file = Path(env_path) if env_path else Path(records_file)
        
        # Ensure data directory exists
        if not self.records_file.parent.exists():
            try:
                self.records_file.parent.mkdir(parents=True, exist_ok=True)
            except:
                pass

        self._cache: Dict[str, Dict] = {}
        self._otps: Dict[str, Dict] = {} # {email: {otp_hash, expires_at, attempts}}
        self._load()

    def save_otp(self, email: str, otp_hash: str, expires_at: datetime):
        print(f"[Storage] Saving OTP for {email}")
        self._otps[email] = {
            "otp_hash": otp_hash,
            "expires_at": expires_at,
            "attempts": 0
        }

    def get_otp(self, email: str) -> Optional[Dict]:
        return self._otps.get(email)

    def increment_otp_attempts(self, email: str):
        if email in self._otps:
            self._otps[email]["attempts"] += 1

    def _load(self):
        """Load records from disk into memory cache."""
        self._cache = {}
        if self.records_file.exists():
            try:
                with open(self.records_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                if "doc_id" in data:
                                    self._cache[data["doc_id"]] = data
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                print(f"[Storage] Load Error: {e}")

    def _flush(self):
        """Write memory cache back to disk."""
        try:
            with open(self.records_file, "w") as f:
                for record in self._cache.values():
                    f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[Storage] Flush Error: {e}")

    def save_record(self, extracted: Dict, doc_id: str, filename: Optional[str] = None, session_id: Optional[str] = None) -> Dict:
        """Create or update a record."""
        record = {
            "doc_id": doc_id,
            "session_id": session_id or "default_session",
            "filename": filename or f"receipt_{doc_id}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "extracted": {
                "total_amount": extracted.get("total_amount", 0.0),
                "date": extracted.get("date", ""),
                "vendor": extracted.get("vendor", "") or extracted.get("vendor_name", ""),
                "receipt_id": extracted.get("receipt_id", ""),
            },
            "raw_text": extracted.get("raw_text", ""),
            "method": extracted.get("method", "unknown"),
            "image_data": extracted.get("image_data"),
        }
        self._cache[doc_id] = record
        self._flush()
        return record

    def get_record(self, doc_id: str) -> Optional[Dict]:
        return self._cache.get(doc_id)

    def list_all(self, limit: int = 50, session_id: Optional[str] = None) -> List[Dict]:
        records = list(self._cache.values())
        if session_id:
            records = [r for r in records if r.get("session_id") == session_id]
        
        # Sort by timestamp descending
        records.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return records[:limit]

    def search(self, query: str) -> List[Dict]:
        query = query.lower()
        results = []
        for r in self._cache.values():
            text = (r.get("raw_text") or "").lower()
            vendor = (r.get("extracted", {}).get("vendor") or "").lower()
            if query in text or query in vendor:
                results.append(r)
        return results

    def delete_record(self, doc_id: str) -> bool:
        if doc_id in self._cache:
            del self._cache[doc_id]
            self._flush()
            return True
        return False


# Global Singleton
_storage_instance = None

def get_storage() -> RecordStorage:
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = RecordStorage()
    return _storage_instance
