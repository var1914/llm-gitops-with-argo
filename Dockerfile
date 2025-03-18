# Multi-stage build for LLM service
# Stage 1: Build dependencies
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Runtime image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MODEL_PATH="/app/model" \
    TRANSFORMERS_CACHE="/app/.cache/huggingface" \
    # Can be overridden at runtime with: -e MODEL_NAME=your-model-name
    MODEL_NAME="prajjwal1/bert-tiny" \
    # Set to 'cpu' or 'cuda' 
    DEVICE="cpu" \
    # Number of workers for Gunicorn
    WORKERS=1

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /app/wheels /app/wheels
RUN pip install --no-cache-dir --no-index --find-links=/app/wheels/ /app/wheels/* && \
    pip install --no-cache-dir gunicorn==20.1.0 uvicorn && \
    rm -rf /app/wheels

# Copy application code
COPY src/ /app/src/

# Create directories
RUN mkdir -p ${MODEL_PATH} ${TRANSFORMERS_CACHE}

# Create entrypoint script
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the server
ENTRYPOINT ["/app/entrypoint.sh"]