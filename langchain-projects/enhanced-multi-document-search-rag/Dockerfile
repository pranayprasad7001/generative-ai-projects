# 1. Use an official lightweight Python image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# 4. Install system dependencies (for PDF parsing, OCR, and builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy application code
COPY . .

# 7. Expose default port
EXPOSE 8000

# 8. Start FastAPI backend with Uvicorn (uses Render's dynamic $PORT if set)
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
