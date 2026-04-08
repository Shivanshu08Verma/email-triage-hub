# Email Triage Hub – OpenEnv Environment
# Compatible with HuggingFace Spaces (port 7860) and local Docker (port 8000)

FROM python:3.11-slim

LABEL maintainer="email-triage-hub-team"
LABEL description="OpenEnv Email Triage Hub environment"
LABEL version="1.0.0"

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# HuggingFace Spaces runs as user 1000
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8000}/health" || exit 1

# Use server.app:main entry point (matching pyproject.toml [project.scripts])
CMD ["sh", "-c", "python -m uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info"]
