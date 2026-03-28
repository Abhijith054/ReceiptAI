# 🏗️ ReceiptAI Architecture & Design Documentation

This document provides a deep dive into the technical design, model approach, and trade-offs of the ReceiptAI platform.

## 1. System Architecture Overview

ReceiptAI is a modular, event-driven (via REST) document extraction platform. Each component is loosely coupled to allow for independent scaling or replacement (e.g., swapping storage providers without changing extraction logic).

### Core Components
*   **FastAPI Backend (`app/main.py`)**: The primary gateway handling image uploads, extraction orchestration, and session-based CRUD operations.
*   **NER Extraction Engine (`src/extractor.py`)**: A hybrid model that uses a fine-tuned DistilBERT Token Classifier for intelligent field recognition, with a deterministic Regex fallback.
*   **Session-Isolated Storage (`src/storage.py`)**: A flat-file JSON-Lines database that treats each chat session as a unique namespace, ensuring no data leakage between users.
*   **Modern Web UI (`frontend/`)**: A high-performance, single-page application built with vanilla JS and Tailwind CSS, featuring a 3-panel layout and real-time session explorers.

---

## 2. Model & Training Approach

We utilize a multi-layer strategy for extraction to optimize for both accuracy and edge-case coverage.

### Model 1: Fine-tuned DistilBERT (NER)
*   **Backbone**: `distilbert-base-uncased` (66M parameters).
*   **Task**: Token Classification (IOB-tagging) for `VENDOR`, `DATE`, `TOTAL`, and `ID`.
*   **Dataset**: **CORD-v2** (Consolidated Receipt Dataset for Post-OCR Parsing).
*   **Metrics**: Evaluated using **seqeval F1-score**, precision, and recall.
*   **Why DistilBERT?**: 40% smaller than BERT-base while retaining 97% of the performance. It runs comfortably on CPUs without GPU acceleration.

### Model 2: LLaMA 3 QLoRA (Advanced Fine-tuning)
For more complex spatial relationships (e.g., multi-column receipts), we implement a QLoRA adapter for **LLaMA 3 8B**:
*   **Approach**: Quantized Low-Rank Adaptation (4-bit quantization).
*   **Objective**: Zero-shot or few-shot extraction of deeply nested tables.
*   **File**: `scripts/train_llama3_qlora.py`.

---

## 3. Design Decisions & Trade-offs

| Decision | Rationale | Trade-off |
| :--- | :--- | :--- |
| **JSONL vs SQL** | JSON-Lines provides excellent write performance and easy human readability for logs. | Lacks ACID compliance and formal relational constraints. |
| **UUID8 Session IDs** | Short, 8-character hex IDs keep the UI compact and URLs clean. | Marginal increase in collision probability vs standard UUID4 (acceptable at human scale). |
| **OCR Fallback** | Tesseract PSM=6 is used when Model extraction fails or yields low confidence. | Slower than pure NER inference. |
| **Session Isolation** | Enforces a strict boundary for Chat History. | Requires constant tracking of `session_id` in the frontend state. |

---

## 4. Setup & Deployment

### Prerequisites
*   Python 3.10+
*   Tesseract OCR (`brew install tesseract`)

### Environment
1. Create venv: `python -m venv venv && source venv/bin/activate`
2. Install: `pip install -r requirements.txt`
3. Launch: `uvicorn app.main:app --port 8000`

### Fine-Tuning
Execute the one-command training pipeline to refresh the local NER model:
`bash scripts/run_training.sh`

---

## 5. Clean Code Principles
*   **Single Responsibility**: `src/storage.py` handles *only* I/O; `src/extractor.py` handles *only* parsing.
*   **Graceful Degradation**: If the ML model fails, the system transparently falls back to Regex-based heuristics.
*   **Self-Healing Storage**: The `_load_all()` logic automatically reconciles index corruption on startup.
