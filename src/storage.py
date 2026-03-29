import json
import os
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Constants
RECORDS_FILE = "data/extracted_records.jsonl"
MONGO_URI = os.environ.get("MONGODB_URI")


class RecordStorage:
    """Interface and Local File Storage (JSON-Lines)."""
    """Persist and retrieve extracted receipt records."""

    def __init__(self, records_file: str = RECORDS_FILE):
        # Allow environment override (essential for Vercel /tmp usage)
        env_path = os.environ.get("STORAGE_FILE")
        self.records_file = Path(env_path) if env_path else Path(records_file)
        
        # Only attempt mkdir if the directory doesn't already exist and is writable
        if not self.records_file.parent.exists():
            try:
                self.records_file.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                # Silently proceed (e.g. read-only fs)
                pass
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
            "image_data": extracted.get("image_data"), # Optional Base64 or URL
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
        """Simple fuzzy search over vendor name, date, total."""
        query_lower = query.lower()
        results = []
        for record in self._cache.values():
            ex = record.get("extracted", {})
            haystack = " ".join(str(v) for v in ex.values() if v).lower()
            if query_lower in haystack:
                results.append(record)
        return results


class SQLAlchemyStorage:
    """Production PostgreSQL/Supabase Storage via SQLAlchemy."""

    def __init__(self, url: str):
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker
        
        # Clean up URL
        url = url.strip().replace("[", "").replace("]", "")
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        elif url.startswith("file:"):
            # Handle Prisma/SQLite local file URLs
            url = url.replace("file:", "sqlite:///", 1)
            
        self.engine = create_engine(url, pool_pre_ping=True)
        self.Session = sessionmaker(bind=self.engine)
        self._init_db()
        print(f"[Storage] SQL Database connected.")

    def _init_db(self):
        from sqlalchemy import text
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS receipt_records (
                    doc_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    filename TEXT,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    extracted JSONB,
                    method TEXT,
                    image_data TEXT
                );
            """))
            conn.commit()

    def save_record(
        self,
        extracted: Dict,
        doc_id: Optional[str] = None,
        filename: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict:
        from sqlalchemy import text
        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8].upper()
            
        # Standardize record for return and DB
        record = {
            "doc_id": doc_id,
            "session_id": session_id or "default_session",
            "filename": filename or f"receipt_{doc_id}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "extracted": {
                "total_amount": extracted.get("total_amount"),
                "date": extracted.get("date"),
                "vendor_name": extracted.get("vendor_name") or extracted.get("vendor"),
                "receipt_id": extracted.get("receipt_id"),
            },
            "method": extracted.get("method", "sql"),
            "image_data": extracted.get("image_data"), 
        }

        with self.Session() as session:
            session.execute(text("""
                INSERT INTO receipt_records (doc_id, session_id, filename, timestamp, extracted, method, image_data)
                VALUES (:doc_id, :session_id, :filename, :timestamp, :extracted, :method, :image_data)
                ON CONFLICT (doc_id) DO UPDATE SET
                    session_id = EXCLUDED.session_id,
                    filename = EXCLUDED.filename,
                    timestamp = EXCLUDED.timestamp,
                    extracted = EXCLUDED.extracted,
                    method = EXCLUDED.method,
                    image_data = EXCLUDED.image_data;
            """), {
                "doc_id": record["doc_id"],
                "session_id": record["session_id"],
                "filename": record["filename"],
                "timestamp": record["timestamp"],
                "extracted": json.dumps(record["extracted"]),
                "method": record["method"],
                "image_data": record["image_data"]
            })
            session.commit()
        return record

    def get_record(self, doc_id: str) -> Optional[Dict]:
        from sqlalchemy import text
        with self.Session() as session:
            res = session.execute(text("SELECT * FROM receipt_records WHERE doc_id = :d"), {"d": doc_id}).fetchone()
            if not res: return None
            return self._row_to_dict(res)

    def list_all(self, limit: int = 100, session_id: Optional[str] = None) -> List[Dict]:
        from sqlalchemy import text
        query = "SELECT * FROM receipt_records"
        params = {"l": limit}
        if session_id:
            query += " WHERE session_id = :s"
            params["s"] = session_id
        query += " ORDER BY timestamp DESC LIMIT :l"
        
        with self.Session() as session:
            rows = session.execute(text(query), params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def delete_record(self, doc_id: str) -> bool:
        from sqlalchemy import text
        with self.Session() as session:
            res = session.execute(text("DELETE FROM receipt_records WHERE doc_id = :d"), {"d": doc_id})
            session.commit()
            return res.rowcount > 0

    def _row_to_dict(self, row) -> Dict:
        # row indexes: 0:doc_id, 1:session_id, 2:filename, 3:timestamp, 4:extracted, 5:method, 6:image_data
        # Note: SQLAlchemy row objects can be indexed or have attributes
        return {
            "doc_id": row[0],
            "session_id": row[1],
            "filename": row[2],
            "timestamp": row[3].isoformat() if hasattr(row[3], "isoformat") else row[3],
            "extracted": json.loads(row[4]) if isinstance(row[4], str) else row[4],
            "method": row[5],
            "image_data": row[6]
        }


# Singleton
_storage_instance: Optional[RecordStorage] = None

def get_storage() -> RecordStorage:
    global _storage_instance
    if _storage_instance is None:
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
             # Fallback
             db_url = "postgresql://postgres:vevGoq-mahdy4-nupmuc@db.wobknzuvbszgjebpsnyl.supabase.co:5432/postgres"
        
        try:
            _storage_instance = SQLAlchemyStorage(db_url)
        except Exception as e:
            print(f"[Storage] ERROR initializing SQLAlchemyStorage: {e}", flush=True)
            _storage_instance = RecordStorage()
    return _storage_instance
