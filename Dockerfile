# ── Email Triage Hub – OpenEnv Environment ────────────────────────────────────
# Compatible with HuggingFace Spaces (runs as non-root user on port 7860)
# and standard Docker deployments (default port 8000).

FROM python:3.11-slim

# Metadata
LABEL maintainer="email-triage-hub-team"
LABEL description="OpenEnv Email Triage Hub environment"
LABEL version="1.0.0"

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY . .

# ── HuggingFace Spaces runs as user 1000; ensure writable dirs ────────────────
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Expose port (HF Spaces expects 7860; standard OpenEnv expects 8000) ───────
# HF_SPACE=1 env var switches port automatically via PORT env var
EXPOSE 7860
EXPOSE 8000

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8000}/health" || exit 1

# ── Launch ─────────────────────────────────────────────────────────────────────
# PORT env var is set by HuggingFace Spaces to 7860; defaults to 8000 locally.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info"]
