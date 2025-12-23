# syntax=docker/dockerfile:1
# RepoRank RAG Agent - Production Dockerfile
# Uses uv for fast dependency management

FROM python:3.11-slim

LABEL maintainer="RepoRank Team"
LABEL description="RAG-based GitHub repository classifier with x402 payment gating"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Copy dependency files first for layer caching
COPY pyproject.toml ./
COPY uv.lock* ./

# Install Python dependencies with uv
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

# Copy application code
COPY src/ ./src/

# Create data directory and volume mount point
RUN mkdir -p /app/data/chroma_db

# Copy data files if they exist (taxonomy.json)
COPY data/ ./data/ 2>/dev/null || true

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uv
CMD ["uv", "run", "reporank"]
