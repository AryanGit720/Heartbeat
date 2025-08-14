# Base: Python 3.11 slim (CPU-only)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

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

# App code
COPY src /app/src
COPY start.sh /app/start.sh

# Prepare dirs; usually mounted via volumes
RUN mkdir -p /app/data /app/models

# Healthcheck against /ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/ready || exit 1

EXPOSE 8000
CMD ["/app/start.sh"]