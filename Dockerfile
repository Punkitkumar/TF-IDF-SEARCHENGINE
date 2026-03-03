# Optimized Dockerfile for Render deployment
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model during build
# so it doesn't download at runtime (avoids timeout + saves memory)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy source code and data
COPY backend/ backend/
COPY data/processed/ data/processed/

ENV PYTHONPATH=/app

# Render sets PORT env variable; default to 8000 for local dev
ENV PORT=8000
EXPOSE ${PORT}

# Start FastAPI - use $PORT so Render can control the port
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT
