# Base Image: Use a more recent Python version for better performance
FROM python:3.11-slim

# Add metadata labels
LABEL maintainer="Repository Classifier"
LABEL description="FastAPI service for classifying GitHub repositories using zero-shot ML"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Copy requirements first (to cache dependencies if you rebuild)
COPY requirements.txt .

# Install system dependencies and Python packages
# We add --no-cache-dir to keep the image smaller
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and configuration
COPY app.py .
COPY categories.json .

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
