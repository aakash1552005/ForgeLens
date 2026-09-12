# ForgeLens-X — Milestone 8 Production Container Dockerfile
# Optimized for high-throughput forensic document authenticity screening & REST API

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

# Install native system dependencies for OpenCV, Tesseract, and GL rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first for layer caching
COPY requirements.txt /app/

# Install python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and configurations
COPY api/ /app/api/
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY models/ /app/models/
COPY dashboard/ /app/dashboard/

# Create data directories
RUN mkdir -p /app/data/temp_uploads /app/data/reports /app/data/splits

# Expose REST API and Dashboard ports
EXPOSE 8000 8501

# Healthcheck probe against FastAPI liveness endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command: launch production FastAPI REST microservice
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
