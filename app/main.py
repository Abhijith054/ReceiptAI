"""
FastAPI application for the Document Information Extraction & QA System.

Endpoints:
  GET  /health                  – Health check
  POST /extract                 – Extract fields from uploaded image or text
  POST /query                   – Answer a natural language question
  GET  /documents               – List all extracted records
  GET  /documents/{doc_id}      – Get a specific record
  DELETE /documents/{doc_id}    – Delete a record
  GET  /                        – Serve the frontend UI
"""

import io
import os
import sys
import json
import uuid
from pathlib import Path
from typing import Optional

import shutil
# Setup Tesseract with hybrid discovery (PATH search + common absolute paths)
TESS_CMD = os.environ.get("TESSERACT_CMD")
if not TESS_CMD:
    TESS_CMD = shutil.which("tesseract")

if not TESS_CMD or not os.path.isfile(TESS_CMD):
    # Search common fallbacks if PATH lookup failed
    for p in ["/opt/homebrew/bin/tesseract", "/usr/bin/tesseract", "/usr/local/bin/tesseract"]:
        if os.path.isfile(p):
            TESS_CMD = p
            break


try:
    import pytesseract
    if TESS_CMD and os.path.isfile(TESS_CMD):
        pytesseract.pytesseract.tesseract_cmd = TESS_CMD
    else:
        print("[System] Tesseract not found. Image OCR will be disabled.")
except ImportError:
    print("[System] Pytesseract library not installed.")


# Trigger reloader
import uvicorn

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Ensure project root is on sys.path so `src.*` imports work ──
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extractor import get_extractor
from src.storage import get_storage
from src.qa_engine import get_qa_engine

app = FastAPI(
    title="Receipt IE & QA System",
    description="Extract key fields from receipts and answer natural language questions.",
    version="1.0.0",
)

# Allow all origins for development (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Detect Vercel / Read-only environments
IS_VERCEL = os.environ.get("VERCEL") == "1"

# Prepare data and uploads directories (pivot to /tmp if ready-only)
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"

if IS_VERCEL:
    UPLOADS_DIR = Path("/tmp/uploads")
    STORAGE_FILE = Path("/tmp/extracted_records.jsonl")
    # Redirect storage provider to use tmp file
    os.environ["STORAGE_FILE"] = str(STORAGE_FILE)
else:
    STORAGE_FILE = DATA_DIR / "extracted_records.jsonl"

try:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"[System] Warning: Could not initialize local disk paths ({e}). Environment may be restricted.")



# Wait to mount frontend at the end after API routes


# ─── Pydantic models ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None


class TextExtractRequest(BaseModel):
    text: str
    doc_id: Optional[str] = None
    filename: Optional[str] = None


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_ui():
    """Serve the frontend HTML interface."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(
        content="<h1>Receipt IE & QA System</h1>"
                "<p>Frontend not found. Place frontend files in the <code>frontend/</code> directory.</p>"
                "<p>API docs: <a href='/docs'>/docs</a></p>"
    )


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint."""
    extractor = get_extractor()
    return {
        "status": "ok",
        "model_loaded": extractor._model_loaded,
        "extraction_method": "model" if extractor._model_loaded else "regex",
        "qa_mode": "llm" if get_qa_engine().use_llm else "rule-based",
    }



    # ...



@app.post("/extract", tags=["Extraction"])
async def extract_from_upload(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    doc_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """
    Extract structured information from a receipt.

    Accepts either:
    - An image file (JPEG/PNG) – OCR will be attempted
    - Raw text (OCR text pasted directly)

    Returns structured JSON with total_amount, date, vendor_name, receipt_id.
    """
    extractor = get_extractor()
    storage = get_storage()
    filename = None

    # ── Determine input source ──
    if doc_id is None:
        doc_id = str(uuid.uuid4())[:8].upper()


    if file is not None:
        filename = file.filename
        content = await file.read()
        
        # Save image to disk
        ext = Path(filename).suffix if filename else ".jpg"
        file_path = UPLOADS_DIR / f"{doc_id}{ext}"
        with open(file_path, "wb") as f:
            f.write(content)


        # Attempt OCR via pytesseract if it's an image
        if file.content_type and file.content_type.startswith("image/"):
            try:
                import pytesseract
                from PIL import Image, ImageEnhance, ImageFilter
                import shutil
                
                # Check for tesseract and set its command path if on Mac/Homebrew
                tess_cmd = shutil.which("tesseract") or "/opt/homebrew/bin/tesseract"
                pytesseract.pytesseract.tesseract_cmd = tess_cmd

                img = Image.open(io.BytesIO(content))
                
                # Preprocess for better OCR
                # Upscale by 1.5x to help Tesseract with smaller/blurry character fonts
                img = img.resize((int(img.width * 1.5), int(img.height * 1.5)), Image.Resampling.LANCZOS)
                img = img.convert('L')  # Grayscale
                
                # Boost contrast slightly to distinguish letters from background noise
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.8)
                
                # psm 6 (Assume a uniform block of text) often performs better on complex layouts
                # than psm 4 if columns are tight or inconsistent.
                ocr_text = pytesseract.image_to_string(img, config='--psm 6')
            except Exception as e:
                print(f"[API] OCR failed or not configured: {e}")
                # Don't crash on Vercel. Instead, provide a helpful fallback message.
                ocr_text = f"--- [RECERPT_AI_SYSTEM_ERROR] ---\nCould not process image via OCR (Tesseract missing in this environment).\n\nNOTE: You can still paste the receipt text directly into the chat box or use the 'New Chat' feature to try manual entry."

        else:
            # .txt file or plain text upload
            try:
                ocr_text = content.decode("utf-8")
            except Exception:
                ocr_text = content.decode("latin-1", errors="ignore")
    elif text:
        ocr_text = text
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either a file upload or text in the request body.",
        )

    if not ocr_text.strip():
        # Tesseract actually ran but found absolutely zero text.
        raise HTTPException(
            status_code=422,
            detail="We couldn't detect any text in this image. Please ensure the receipt is well-lit and the text is clear.",
        )

    # ── Extract fields ──
    extracted = extractor.extract(ocr_text)

    # ── Persist ──
    record = storage.save_record(extracted, doc_id=doc_id, filename=filename, session_id=session_id)
    
    # Store file extension natively in record so frontend can find it
    if file is not None and "file_ext" not in record:
        ext = Path(filename).suffix if filename else ".jpg"
        record["file_ext"] = ext
        # hacky but works for the session
        record["has_image"] = True 

    return {
        "doc_id": record["doc_id"],
        "filename": record["filename"],
        "has_image": True if file is not None else False,
        "file_ext": Path(filename).suffix if (file and filename) else ".jpg",
        "timestamp": record["timestamp"],
        "extracted": record["extracted"],
        "method": record["method"],
        "message": "Extraction successful. Use /query to ask questions about this receipt.",
    }


@app.post("/extract/text", tags=["Extraction"])
async def extract_from_text(body: TextExtractRequest):
    """Extract from raw OCR text (JSON body). Useful for API clients."""
    extractor = get_extractor()
    storage = get_storage()

    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    extracted = extractor.extract(body.text)
    record = storage.save_record(
        extracted,
        doc_id=body.doc_id,
        filename=body.filename or "pasted_text",
    )

    return {
        "doc_id": record["doc_id"],
        "filename": record["filename"],
        "timestamp": record["timestamp"],
        "extracted": record["extracted"],
        "method": record["method"],
    }


@app.post("/query", tags=["QA"])
async def query(body: QueryRequest):
    """
    Answer a natural language question about extracted receipt data.

    If doc_id is provided, queries only that document.
    Otherwise queries across all stored documents.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    qa = get_qa_engine()
    result = qa.answer(body.question, doc_id=body.doc_id)
    return result


@app.get("/documents", tags=["Documents"])
async def list_documents(limit: int = 50, session_id: Optional[str] = None):
    """List all extracted receipt records, optionally filtered by session_id."""
    storage = get_storage()
    records = storage.list_all(limit=limit, session_id=session_id)
    return {
        "total": len(records),
        "documents": [
            {
                "doc_id": r["doc_id"],
                "session_id": r.get("session_id"),
                "filename": r.get("filename"),
                "timestamp": r.get("timestamp"),
                "extracted": r.get("extracted"),
                "method": r.get("method"),
            }
            for r in records
        ],
    }


@app.get("/documents/{doc_id}", tags=["Documents"])
async def get_document(doc_id: str):
    """Get a specific extracted record by document ID."""
    storage = get_storage()
    record = storage.get_record(doc_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return record


@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete a specific record."""
    storage = get_storage()
    if not storage.delete_record(doc_id):
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return {"message": f"Document '{doc_id}' deleted."}


# ── Static File Mounting (MUST BE LAST) ──────────────────────────────────────
FRONTEND_DIR = PROJECT_ROOT / "frontend"

if FRONTEND_DIR.exists():
    # Primary mount for standard asset calls (/app.js, /style.css)
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    
    # Secondary mount for explicit static calls (/static/app.js)
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

if not IS_VERCEL:
    try:
        app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
    except:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
