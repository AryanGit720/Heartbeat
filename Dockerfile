# Base: Python 3.11 slim (CPU-only)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    RUNTIME_DIR=/tmp/hsc

# System deps for audio and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use lean production requirements (no Kaggle/PyTorch)
COPY requirements-prod.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements.txt

# Copy application code + model + metadata into image
COPY src /app/src
COPY models /app/models
COPY data/metadata /app/data/metadata

# Healthcheck against /ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/ready || exit 1

EXPOSE 8000

# Run gunicorn directly via bash (expands env vars like PORT)
CMD ["/bin/bash", "-lc", "exec gunicorn -k uvicorn.workers.UvicornWorker -w ${WEB_CONCURRENCY:-1} -b 0.0.0.0:${PORT:-8000} --timeout ${TIMEOUT:-120} --graceful-timeout ${GRACEFUL_TIMEOUT:-120} --keep-alive ${KEEPALIVE:-5} --log-level ${LOG_LEVEL:-warning} src.app.main:app"]