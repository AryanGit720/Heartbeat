# Base: Python 3.11 slim (CPU-only)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    RUNTIME_DIR=/tmp/hsc \
    HOME=/tmp/hsc \
    MPLCONFIGDIR=/tmp/mplconfig \
    XDG_CACHE_HOME=/tmp/xdg-cache

# System deps for audio, fonts, healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    bash \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Writable caches
RUN mkdir -p /tmp/hsc /tmp/mplconfig /tmp/xdg-cache && chmod -R 777 /tmp/hsc /tmp/mplconfig /tmp/xdg-cache || true

WORKDIR /app

# Lean production requirements (no Kaggle/PyTorch)
COPY requirements-prod.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements.txt

# Copy application code + model + metadata
COPY src /app/src
COPY models /app/models
COPY data/metadata /app/data/metadata

# Expose the Space port (7860)
EXPOSE 7860

# Lighter startup: single-process Uvicorn (good for free CPU)
# Bind to ${PORT} if set (HF sets it to 7860), otherwise default to 7860
CMD ["/bin/bash", "-lc", "exec python -m uvicorn src.app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]