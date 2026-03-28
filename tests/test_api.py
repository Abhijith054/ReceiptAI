"""
tests/test_api.py – API endpoint integration tests using httpx TestClient.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a test FastAPI client with temporary data paths."""
    monkeypatch.setenv("GROQ_API_KEY", "")

    import src.storage as storage_mod
    import src.qa_engine as qa_mod
    import src.extractor as extractor_mod

    temp_records = str(tmp_path / "records.jsonl")

    # Reset singletons before each test
    storage_mod._storage_instance = None
    qa_mod._qa_engine = None
    extractor_mod._extractor_instance = None

    # Create fresh storage pointing at temp path
    storage_mod._storage_instance = storage_mod.RecordStorage(temp_records)

    from app.main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    yield client

    # Teardown: reset singletons after each test
    storage_mod._storage_instance = None
    qa_mod._qa_engine = None
    extractor_mod._extractor_instance = None


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "extraction_method" in data


class TestExtractText:
    RECEIPT_TEXT = """
    SUNSHINE CAFE
    Date: 2024-01-15
    Receipt No: RC-001
    Cappuccino x2    $8.00
    TOTAL: $25.50
    """

    def test_extract_returns_doc_id(self, client):
        r = client.post(
            "/extract/text",
            json={"text": self.RECEIPT_TEXT},
        )
        assert r.status_code == 200
        data = r.json()
        assert "doc_id" in data
        assert "extracted" in data
        assert len(data["doc_id"]) > 0

    def test_extract_finds_total(self, client):
        r = client.post("/extract/text", json={"text": self.RECEIPT_TEXT})
        assert r.status_code == 200
        ex = r.json()["extracted"]
        assert ex.get("total_amount") is not None

    def test_extract_finds_date(self, client):
        r = client.post("/extract/text", json={"text": self.RECEIPT_TEXT})
        assert r.status_code == 200
        ex = r.json()["extracted"]
        assert ex.get("date") is not None

    def test_extract_empty_text_fails(self, client):
        r = client.post("/extract/text", json={"text": "   "})
        assert r.status_code == 400


class TestDocuments:
    def test_list_empty(self, client):
        r = client.get("/documents")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["documents"] == []

    def test_list_after_extract(self, client):
        client.post("/extract/text", json={"text": "TOTAL: 10.00 Date: 2024-01-01"})
        r = client.get("/documents")
        assert r.status_code == 200
        assert r.json()["total"] >= 1

    def test_get_specific_doc(self, client):
        r1 = client.post("/extract/text", json={"text": "Total: $25.50"})
        doc_id = r1.json()["doc_id"]
        r2 = client.get(f"/documents/{doc_id}")
        assert r2.status_code == 200
        assert r2.json()["doc_id"] == doc_id

    def test_get_nonexistent_doc(self, client):
        r = client.get("/documents/NONEXISTENT")
        assert r.status_code == 404

    def test_delete_doc(self, client):
        r1 = client.post("/extract/text", json={"text": "Total: $10.00"})
        doc_id = r1.json()["doc_id"]
        r2 = client.delete(f"/documents/{doc_id}")
        assert r2.status_code == 200
        r3 = client.get(f"/documents/{doc_id}")
        assert r3.status_code == 404


class TestQuery:
    def test_query_no_docs(self, client):
        r = client.post("/query", json={"question": "What is the total?"})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "No documents" in data["answer"] or data["answer"]

    def test_query_with_doc(self, client):
        r1 = client.post("/extract/text", json={"text": "TOTAL: $99.99 Date: 2024-06-01"})
        doc_id = r1.json()["doc_id"]

        r2 = client.post("/query", json={"question": "What is the total amount?", "doc_id": doc_id})
        assert r2.status_code == 200
        assert "answer" in r2.json()

    def test_query_empty_question(self, client):
        r = client.post("/query", json={"question": "  "})
        assert r.status_code == 400
