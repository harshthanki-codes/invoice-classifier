# Stage 1: install dependencies
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: lean runtime image
FROM python:3.11-slim
LABEL maintainer="your-email@example.com"
LABEL description="Invoice Expense Classifier API v2.0.0"

# Non-root user — required for Render, Fly.io, and security-conscious environments
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app
COPY --from=builder /install /usr/local
COPY app/ ./app/
COPY data/ ./data/
COPY scripts/ ./scripts/

# Pre-train at build time — container starts with model ready, zero cold-start ML cost
RUN python scripts/train.py

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
