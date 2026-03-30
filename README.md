# ReceiptAI – Secure Receipt Intelligence Access

![ReceiptAI Banner](https://img.shields.io/badge/ReceiptAI-Intelligent%20Extraction-6366f1?style=for-the-badge)

ReceiptAI is a highly optimized, AI-driven extraction and conversing engine for digitizing unstructured receipts and documents. Designed with a modular architecture, it extracts structured key fields (Vendor, Date, Total) using OCR and Large Language Models, and allows users to query their document contents via a conversational chat interface.

---

## 1. System Architecture

The application adopts a modular, client-server design ensuring clear separation of concerns:

- **Frontend (UI Presentation Layer):** Built with pure HTML/CSS/VanillaJS and TailwindCSS. Uses an animated `particles.js` dark-mode interface. Interacts with the backend purely via REST API calls. 
- **Backend (API & Logic Layer):** Built on `FastAPI` (Python). Exposes endpoints for document upload, structured extraction, intelligent question answering, and session handling.
- **Processing Engine:**
  - **OCR Module:** Utilizes `PyTesseract` to extract raw spatial token strings from images.
  - **Extraction Module (`src/extractor.py`):** Uses a hybrid stack (Regex + Groq LLM) to deduce the schema representation of the text.
  - **QA Module (`src/qa_engine.py`):** Generative capabilities leveraging the **Groq Llama-3-8b-instant** language model for zero-shot question answering over the document context.
- **Storage Layer:** MongoDB Atlas handles both standard document metadata and potentially binary assets via GridFS, enabling stateless horizontal scaling.

---

## 2. Model & Training Approach

The project initially architected a custom pipeline around pre-trained local NLP token-classification models (such as BERT or LayoutLM) for Named Entity Recognition (NER).

### Fine-Tuning Details
- **Dataset:** The model was trained using annotated receipt data (e.g., SROIE), where each word token is mapped to an entity tag using the BIO format (e.g., `B-COMPANY`, `I-COMPANY`, `B-DATE`, `O`).
- **Input Pipeline:** The dataset was tokenized using the respective HuggingFace tokenizer. Bounding box coordinates (for LayoutLM) and text features were passed through the transformer blocks.
- **Objective:** The training utilized cross-entropy loss over a token classification head mapping the embeddings to the target classes (Vendor, Date, Total).
- **Inference:** At inference, the OCR output string sequences were processed by the model to segment text into the targeted data points.

*Note: The local model inference has since been decoupled to accommodate strict production hosting limits (see Limitations).*

---

## 3. Design Decisions & Trade-Offs

**1. Inference: Local ML vs. Cloud LLMs**
* **Decision:** We pivoted from heavy local `Transformers`/`PyTorch` models to rule-based fallback paired with cloud-based LLM routing (Groq).
* **Trade-off:** We sacrifice true offline, air-gapped processing for dramatically reduced compute/memory footprint (~800MB saved), faster response times (Groq's LPU), and broader contextual extraction beyond rigid NER constraints.

**2. Fast, Stateless Deployment**
* **Decision:** Replaced temporary local file storage (SQLite / OS file writes) with ephemeral memory workflows and external persistent setups (MongoDB).
* **Trade-off:** Ingestion requires stable network connectivity to external databases, but it prevents the application state from fragmenting and allows safe pod recycling in cloud hostings.

**3. Frontend Simplicity**
* **Decision:** Forewent complex reactive frameworks (React/Next.js) in favor of Vanilla JS and DOM manipulation. 
* **Trade-off:** Less overhead and immediate page loading, but managing complex conversational state becomes highly manual as the interface scales.

---

## 4. Setup Instructions

### Prerequisites
- Python 3.9+
- `tesseract-ocr` installed on the host machine (`sudo apt-get install tesseract-ocr` or `brew install tesseract`)
- [Groq API Key](https://console.groq.com/)
- MongoDB Atlas Connection URI

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/ReceiptAI.git
   cd ReceiptAI
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_key
   MONGO_URI=your_mongodb_connection_string
   SKIP_LOCAL_MODEL=1
   ```

5. **Run the Application**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Navigate to `http://localhost:8000` to interact with the frontend.

---

## 5. Limitations & Future Improvements

### General Limitations
- **OCR Accuracy:** PyTesseract struggles heavily with crumpled receipts, handwritten notes, and skewed low-light photos.
- **Language Support:** Currently highly optimized for English/Latin-character structures.

### Future Improvements
- **Vision Models:** Transitioning from text-only LLMs + Tesseract to multimodal Small Vision Models (like `Moondream` or `Qwen-VL`) which can read and segment structural documents globally without a discrete OCR step.
- **Batch Processing:** Async queue processing for bulk uploading PDF invoices.

### ⚠️ Limitations Deploying on Render (Free Tier)
Deployments to Platform-as-a-Service environments like **Render.com** encounter strict constraints that molded the codebase:
1. **Memory Limits (OOM Errors):** Render Free tier provides only `512MB` of RAM. Loading machine-learning libraries like `torch` and `transformers` causes instant `Out-Of-Memory (137)` crashes. **Fix:** The app wraps local models in `SKIP_LOCAL_MODEL=1` boundaries, relying entirely on Groq inference.
2. **SMTP Port Blocking:** Render actively blocks outbound email protocols on port `587` to prevent spam, causing standard Gmail SMTP configurations for OTP verification to fail. **Fix:** We transitioned internal email traffic to standard HTTPS using the `Resend API` to bypass this port restriction, and eventually deactivated OTP for a completely frictionless interface.
3. **Ephemeral File System:** Any files (images/logs) saved to disk are deleted instantly on service spin-down/redeployment. All data must be routed directly into MongoDB.
