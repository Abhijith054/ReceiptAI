FROM python:3.12-slim

# Prevent interactive prompts (like tzdata) from completely freezing the build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies (Tesseract + Poppler for PDF fallback)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Hugging Face requires running as a non-root user with UID 1000
RUN useradd -m -u 1000 user

# Set local workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code with correct permissions
COPY --chown=user:user . /app

# Switch to the non-root user
USER user

# Hugging Face requires exposing port 7860
EXPOSE 7860

# Run the FastAPI app on port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
