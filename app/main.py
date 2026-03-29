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
import hashlib
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv
import jwt

# Load .env file
load_dotenv()

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

# Auth Configuration
JWT_SECRET = os.environ.get("JWT_SECRET", "receipt-ai-default-secret-54321")
JWT_ALGORITHM = "HS256"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")

app = FastAPI(
    title="Receipt IE & QA System",
    description="Extract key fields from receipts and answer natural language questions.",
    version="1.0.0",
)

# Pydantic Models for Auth
class EmailRequest(BaseModel):
    email: str

class VerifyRequest(BaseModel):
    email: str
    otp: str

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
    print(f"[System] Uploads directory: {UPLOADS_DIR}")
except Exception as e:
    print(f"[System] Warning: Could not initialize local disk paths ({e}). Environment may be restricted.")



# Wait to mount frontend at the end after API routes


# ─── Pydantic models ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None # For Conversational Memory


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
    """Check system status."""
    return {
        "status": "online",
        "engine": "ReceiptAI Intelligence v1.0",
        "qa_mode": "llm" if get_qa_engine().use_llm else "rule-based",
    }

# ── Auth Endpoints ──────────────────────────────────────────────────────────

@app.post("/send-otp", tags=["Auth"])
async def send_otp(req: EmailRequest):
    storage = get_storage()
    email = req.email.lower().strip()
    
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email address")

    otp = f"{random.randint(100000, 999999)}"
    otp_hash = hashlib.sha256(otp.encode()).hexdigest()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    
    storage.save_otp(email, otp_hash, expires_at)
    
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        print(f"[AUTH] NO CREDENTIALS. OTP FOR {email}: {otp}")
        return {"message": "OTP sent (dev mode)", "dev": True, "otp": otp}

    try:
        # SMTP email sending
        msg = MIMEMultipart("alternative")
        msg['Subject'] = 'ReceiptAI Verification Code'
        msg['From'] = GMAIL_USER
        msg['To'] = email

        text_version = f"Your ReceiptAI verification code is: {otp}\nExpires in 5 minutes."
        html_version = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; background-color: #000000; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
    <table width="100%" border="0" cellspacing="0" cellpadding="0" style="background-color: #000000; padding: 40px 20px;">
        <tr>
            <td align="center">
                <table width="100%" border="0" cellspacing="0" cellpadding="0" style="max-width: 500px; background-color: #121212; border-radius: 24px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05);">
                    <!-- Header -->
                    <tr>
                        <td align="center" style="padding: 40px 40px 20px 40px; background: linear-gradient(180deg, #1a1a1a 0%, #121212 100%);">
                            <h1 style="margin: 0; font-size: 24px; font-weight: 900; color: #ffffff; letter-spacing: -0.02em;">Receipt<span style="color: #6366f1;">AI</span></h1>
                            <p style="margin: 8px 0 0 0; font-size: 10px; font-weight: 800; color: #4b5563; text-transform: uppercase; letter-spacing: 0.2em;">Email Verification</p>
                        </td>
                    </tr>
                    
                    <!-- Main Body -->
                    <tr>
                        <td align="center" style="padding: 20px 40px 40px 40px;">
                            <h2 style="margin: 0; font-size: 18px; font-weight: 700; color: #ffffff; letter-spacing: -0.01em;">Verification OPT</h2>
                            <p style="margin: 12px 0 32px 0; font-size: 13px; line-height: 1.6; color: #9ca3af;">Use the verification code below to complete your login.</p>
                            
                            <!-- OTP Box -->
                            <div style="background-color: #1e1e1e; border-radius: 16px; padding: 24px; border: 1px solid rgba(255,255,255,0.05); text-align: center;">
                                <div style="font-size: 36px; font-weight: 900; color: #ffffff; letter-spacing: 0.5em; margin-right: -0.5em;">{otp}</div>
                            </div>
                            
                            <p style="margin: 32px 0 0 0; font-size: 11px; font-weight: 600; color: #4b5563; text-transform: uppercase; letter-spacing: 0.05em;">This code will expire in <span style="color: #ef4444;">5 minutes</span></p>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td align="center" style="padding: 0 40px 40px 40px; border-top: 1px solid rgba(255,255,255,0.03);">
                            <p style="margin: 24px 0 0 0; font-size: 11px; line-height: 1.5; color: #374151;">If you didn't request this code, you can safely ignore this email.</p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""
        msg.attach(MIMEText(text_version, "plain"))
        msg.attach(MIMEText(html_version, "html"))
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.send_message(msg)
            
        return {"message": "OTP sent successfully"}
    except Exception as e:
        print(f"[AUTH] SMTP Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to send email")

@app.post("/verify-otp", tags=["Auth"])
async def verify_otp(req: VerifyRequest):
    storage = get_storage()
    email = req.email.lower().strip()
    otp = req.otp.strip()
    
    record = storage.get_otp(email)
    if not record:
        raise HTTPException(status_code=404, detail="No OTP requested for this email")
        
    if record["attempts"] >= 3:
        raise HTTPException(status_code=403, detail="Too many attempts. Request new OTP.")
        
    if datetime.now(timezone.utc) > record["expires_at"]:
        raise HTTPException(status_code=400, detail="OTP expired")
        
    otp_hash = hashlib.sha256(otp.encode()).hexdigest()
    if record["otp_hash"] != otp_hash:
        storage.increment_otp_attempts(email)
        raise HTTPException(status_code=401, detail="Invalid OTP")
        
    access_token = jwt.encode(
        {"sub": email, "exp": datetime.now(timezone.utc) + timedelta(days=7)},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM
    )
    
    return {"access_token": access_token, "email": email}



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
        
        # Save image to disk (for temporary local caching)
        ext = Path(filename).suffix if filename else ".jpg"
        file_path = UPLOADS_DIR / f"{doc_id}{ext}"
        with open(file_path, "wb") as f:
            f.write(content)

        # Prepare for DB storage (Base64)
        import base64
        image_b64 = base64.b64encode(content).decode("utf-8")
        file_type = file.content_type or "image/jpeg"
        image_data = f"data:{file_type};base64,{image_b64}"


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
    
    # Include image for storage & return
    if file is not None:
        extracted["image_data"] = image_data
        # Construct path-based URL for performance/caching
        ext = Path(filename).suffix if filename else ".jpg"
        extracted["image_url"] = f"/uploads/{doc_id}{ext}"

    # ── Persist ──
    record = storage.save_record(extracted, doc_id=doc_id, filename=filename, session_id=session_id)
    
    # Return the full record for consistency and to ensure image_data is top-level for JS
    return record


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
    result = qa.answer(body.question, doc_id=body.doc_id, history=body.history)
    return result


@app.get("/documents", tags=["Documents"])
async def list_documents(limit: int = 50, session_id: Optional[str] = None):
    """List all extracted receipt records, optionally filtered by session_id."""
    storage = get_storage()
    records = storage.list_all(limit=limit, session_id=session_id)
    
    docs = []
    for r in records:
        doc_id = r["doc_id"]
        filename = r.get("filename", "")
        # Fallback extension discovery
        ext = Path(filename).suffix if filename else ".png"
        
        # Ensure image_url is present even for legacy records
        image_url = r.get("extracted", {}).get("image_url")
        if not image_url:
            # Check if file exists on disk
            potential = UPLOADS_DIR / f"{doc_id}{ext}"
            if potential.exists():
                image_url = f"/uploads/{doc_id}{ext}"
            else:
                # If .png didn't work, try .jpg
                potential = UPLOADS_DIR / f"{doc_id}.jpg"
                if potential.exists():
                    image_url = f"/uploads/{doc_id}.jpg"
        
        docs.append({
            "doc_id": doc_id,
            "session_id": r.get("session_id"),
            "filename": filename,
            "timestamp": r.get("timestamp"),
            "extracted": r.get("extracted"),
            "image_url": image_url,
            "method": r.get("method"),
        })

    return {
        "total": len(docs),
        "documents": docs,
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


# ── Static File Mounting (ORDER MATTERS) ─────────────────────────────────────
FRONTEND_DIR = PROJECT_ROOT / "frontend"

if not IS_VERCEL:
    try:
        # Mount uploads first so it's checked before the / catch-all
        app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
    except Exception as e:
        print(f"[System] Could not mount uploads: {e}")

if FRONTEND_DIR.exists():
    # Mount specific static path
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    
    # Catch-all for the frontend app (MUST BE LAST)
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
