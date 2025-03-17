# Multi-stage build for optimized LLM service container

# ---- Base Python image ----
    FROM python:3.9-slim AS base

    # Set environment variables
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PYTHONFAULTHANDLER=1 \
        PIP_NO_CACHE_DIR=off \
        PIP_DISABLE_PIP_VERSION_CHECK=on
    
    WORKDIR /app
    
    # Install system dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*
    
    # ---- Dependencies stage ----
    FROM base AS dependencies
    
    # Copy requirements file
    COPY requirements.txt .
    
    # Install Python dependencies
    RUN pip install --no-cache-dir -r requirements.txt
    
    # ---- Builder stage for model training (optional) ----
    FROM dependencies AS model-builder
    COPY . .
    
    # Default training arguments - can be overridden at runtime
    ENV MODEL_NAME="bert-base-uncased" \
        DATASET_NAME="glue" \
        DATASET_CONFIG="sst2" \
        OUTPUT_DIR="/app/model/artifacts" \
        NUM_TRAIN_EPOCHS=3 \
        BATCH_SIZE=16 \
        SEED=42 \
        LR=2e-5
    
    # Create model artifacts directory
    RUN mkdir -p /app/model/artifacts
    
    # Script to train the model if needed
    RUN echo '#!/bin/bash \n\
    python src/train.py \
        --model_name ${MODEL_NAME} \
        --dataset_name ${DATASET_NAME} \
        --dataset_config ${DATASET_CONFIG} \
        --output_dir ${OUTPUT_DIR} \
        --num_train_epochs ${NUM_TRAIN_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --seed ${SEED} \
        --lr ${LR} \
        "$@"' > /app/train.sh && chmod +x /app/train.sh
    
    # ---- Final production image ----
    FROM dependencies AS production
    
    # Install additional production-specific dependencies
    RUN pip install --no-cache-dir \
        gunicorn==20.1.0 \
        prometheus-client==0.16.0 \
        uvloop==0.17.0 \
        httptools==0.6.0
    
    # Copy application code and utilities
    COPY src/ /app/src/
    
    # Create model directory
    RUN mkdir -p /app/model/artifacts
    
    # Copy model artifacts from builder image if using trained model
    # COPY --from=model-builder /app/model/artifacts /app/model/artifacts
    
    # Set environment variables
    ENV MODEL_PATH="/app/model/artifacts" \
        PORT=8080 \
        MAX_WORKERS=1 \
        TIMEOUT=120 \
        LOG_LEVEL=info
    
    # Copy entrypoint script
    RUN echo '#!/bin/bash \n\
    # Set up cache directory with proper permissions \n\
    export TRANSFORMERS_CACHE="/app/.cache/huggingface" \n\
    mkdir -p ${TRANSFORMERS_CACHE} \n\
    \n\
    # Check if we need to download model from some external source \n\
    if [ -n "${MODEL_URL}" ]; then \n\
      echo "Downloading model from ${MODEL_URL}" \n\
      mkdir -p ${MODEL_PATH} \n\
      curl -L ${MODEL_URL} | tar xz -C ${MODEL_PATH} \n\
    fi \n\
    \n\
    # If no model exists, download a small default model \n\
    if [ ! -f ${MODEL_PATH}/config.json ]; then \n\
      echo "No model found in ${MODEL_PATH}, downloading a default model..." \n\
      mkdir -p ${MODEL_PATH} \n\
      python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; model = AutoModelForSequenceClassification.from_pretrained(\"prajjwal1/bert-tiny\", num_labels=2); tokenizer = AutoTokenizer.from_pretrained(\"prajjwal1/bert-tiny\"); model.save_pretrained(\"${MODEL_PATH}\"); tokenizer.save_pretrained(\"${MODEL_PATH}\")" \n\
    fi \n\
    \n\
    # Start the inference server \n\
    exec gunicorn src.inference:app \
      --bind 0.0.0.0:${PORT} \
      --workers ${MAX_WORKERS} \
      --worker-class uvicorn.workers.UvicornWorker \
      --timeout ${TIMEOUT} \
      --log-level ${LOG_LEVEL} \
      --forwarded-allow-ips="*" \
      "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh
    
    # Create a non-root user and change ownership
    RUN groupadd -r model && useradd -r -g model model \
        && mkdir -p /app/.cache/huggingface \
        && chown -R model:model /app
    
    # Switch to non-root user
    USER model
    
    # Expose the port
    EXPOSE ${PORT}
    
    # Health check
    HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
      CMD curl -f http://localhost:${PORT}/health || exit 1
    
    # Run the server
    ENTRYPOINT ["/app/entrypoint.sh"]