# Multi-stage build for a smaller image size
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and modules
COPY backend/ backend/
COPY scrapers/ scrapers/
COPY scripts/ scripts/
COPY data/ data/

ENV PYTHONPATH=/app

# Expose the API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
