<<<<<<< HEAD

=======
# 🧾 ReceiptAI — Intelligent Document Analysis

ReceiptAI is a modern, high-performance platform for extracting structured information from receipts using localized NER (Named Entity Recognition) and advanced LLaMA-based fine-tuning.

---

## ⚡ Quick Start

### 1. Prerequisites
*   Python 3.10+
*   Tesseract OCR (`brew install tesseract`)

### 2. Setup
```bash
git clone ... && cd ReceiptAI
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch
```bash
# Start the FastAPI server
uvicorn app.main:app --reload --port 8000
```
Open **http://localhost:8000** to begin your analysis.

---

## 🚀 Key Features

*   **Session-Isolated Chats**: Every analysis is a unique, secure session. Use **"New Chat"** to start fresh without data bleed.
*   **Persistent Chat History**: Previous sessions are automatically saved in the sidebar for easy retrieval.
*   **Hybrid Extraction Engine**: Combines fine-tuned DistilBERT (NER) with deterministic regex fallbacks.
*   **Data Management**: View raw files or extracted JSON directly through the intuitive "Data Management" sidebar options.

---

## 🏗️ Architecture & ML Approach

For detailed technical documentation, please refer to:
👉 **[DOCUMENTATION.md](./DOCUMENTATION.md)**

Includes:
*   Core System Architecture
*   Model Training Strategies (DistilBERT & LLaMA 3 QLoRA)
*   Design Decisions & Performance Trade-offs

---

## 🧪 Running Tests
```bash
pytest tests/ -v
```

*ML Engineer Role Assessment — ReceiptAI focuses on precision, modularity, and premium user experience.*
>>>>>>> 945c64b (🚀 Initial Release: ReceiptAI Session-Based Document Analytics Platform)
# ee
