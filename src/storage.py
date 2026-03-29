import os
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class DBRecord(Base):
    __tablename__ = "receipt_records"
    doc_id = Column(String, primary_key=True)
    session_id = Column(String, index=True)
    filename = Column(String)
    timestamp = Column(String)
    vendor = Column(String)
    total_amount = Column(Float)
    date = Column(String)
    raw_text = Column(Text)
    method = Column(String)
    image_url = Column(String, nullable=True)

class SQLStorage:
    def __init__(self, db_url: str):
        # Fix for some SQLAlchemy versions needing postgresql://
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        
        self.engine = create_engine(db_url, pool_pre_ping=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self._otps = {}

    def save_otp(self, email: str, otp_hash: str, expires_at: datetime):
        self._otps[email] = {"otp_hash": otp_hash, "expires_at": expires_at, "attempts": 0}

    def get_otp(self, email: str): return self._otps.get(email)
    def increment_otp_attempts(self, email: str):
        if email in self._otps: self._otps[email]["attempts"] += 1

    def save_record(self, extracted: Dict, doc_id: str, filename: Optional[str] = None, session_id: Optional[str] = None) -> Dict:
        session = self.Session()
        try:
            record = DBRecord(
                doc_id=doc_id,
                session_id=session_id or "default_session",
                filename=filename or f"receipt_{doc_id}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                vendor=extracted.get("vendor", "") or extracted.get("vendor_name", ""),
                total_amount=extracted.get("total_amount", 0.0),
                date=extracted.get("date", ""),
                raw_text=extracted.get("raw_text", ""),
                method=extracted.get("method", "unknown"),
                image_url=extracted.get("image_url")
            )
            session.merge(record) # Update if exists
            session.commit()
            return self._to_dict(record)
        finally:
            session.close()

    def get_record(self, doc_id: str) -> Optional[Dict]:
        session = self.Session()
        try:
            r = session.query(DBRecord).filter(DBRecord.doc_id == doc_id).first()
            return self._to_dict(r) if r else None
        finally:
            session.close()

    def list_all(self, limit: int = 50, session_id: Optional[str] = None) -> List[Dict]:
        session = self.Session()
        try:
            query = session.query(DBRecord)
            if session_id: query = query.filter(DBRecord.session_id == session_id)
            records = query.order_by(DBRecord.timestamp.desc()).limit(limit).all()
            return [self._to_dict(r) for r in records]
        finally:
            session.close()

    def delete_record(self, doc_id: str) -> bool:
        session = self.Session()
        try:
            r = session.query(DBRecord).filter(DBRecord.doc_id == doc_id).first()
            if r:
                session.delete(r)
                session.commit()
                return True
            return False
        finally:
            session.close()

    def _to_dict(self, r: DBRecord) -> Dict:
        return {
            "doc_id": r.doc_id,
            "session_id": r.session_id,
            "filename": r.filename,
            "timestamp": r.timestamp,
            "extracted": {
                "vendor": r.vendor,
                "total_amount": r.total_amount,
                "date": r.date,
                "image_url": r.image_url
            },
            "raw_text": r.raw_text,
            "method": r.method
        }

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

def get_storage():
    global _storage_instance
    if _storage_instance is None:
        db_url = os.environ.get("DATABASE_URL")
        if db_url and "supabase" in db_url:
            print("[Storage] Connecting to Supabase (PostgreSQL)...")
            try:
                _storage_instance = SQLStorage(db_url)
            except Exception as e:
                print(f"[Storage] DB connection error: {e}. Falling back to JSONL.")
                _storage_instance = RecordStorage()
        else:
            print("[Storage] Using local JSONL file storage.")
            _storage_instance = RecordStorage()
    return _storage_instance
