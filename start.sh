#!/usr/bin/env bash
set -e

# Defaults; override with env vars
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${WEB_CONCURRENCY:=2}"
: "${TIMEOUT:=120}"
: "${KEEPALIVE:=5}"
: "${GRACEFUL_TIMEOUT:=120}"
: "${LOG_LEVEL:=info}"

echo "Starting gunicorn on ${HOST}:${PORT} with ${WEB_CONCURRENCY} workers"
exec gunicorn \
  -k uvicorn.workers.UvicornWorker \
  -w "${WEB_CONCURRENCY}" \
  -b "${HOST}:${PORT}" \
  --timeout "${TIMEOUT}" \
  --graceful-timeout "${GRACEFUL_TIMEOUT}" \
  --keep-alive "${KEEPALIVE}" \
  --log-level "${LOG_LEVEL}" \
  src.app.main:app