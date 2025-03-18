#!/bin/bash
set -e

# Print environment for debugging
echo "Starting LLM service with:"
echo "  - MODEL_PATH: ${MODEL_PATH}"
echo "  - MODEL_NAME: ${MODEL_NAME}"
echo "  - DEVICE: ${DEVICE}"
echo "  - PORT: ${PORT}"
echo "  - WORKERS: ${WORKERS}"

# Create cache directory
mkdir -p ${TRANSFORMERS_CACHE}

# Check if we have a model already in the model path
if [ ! -f ${MODEL_PATH}/config.json ]; then
  echo "No model found at ${MODEL_PATH}"
  
  # Check if we should download a pre-trained model
  if [ -n "${MODEL_NAME}" ]; then
    echo "Downloading model: ${MODEL_NAME}..."
    python -c "
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# Determine number of labels - default to 2 for binary classification
num_labels = int(os.environ.get('NUM_LABELS', '2'))
print(f'Using {num_labels} labels for classification')

# Download and save the model
model = AutoModelForSequenceClassification.from_pretrained('${MODEL_NAME}', num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained('${MODEL_NAME}')

# Save to the specified directory
model.save_pretrained('${MODEL_PATH}')
tokenizer.save_pretrained('${MODEL_PATH}')

print(f'Model and tokenizer saved to ${MODEL_PATH}')
"
  else
    echo "ERROR: No model at ${MODEL_PATH} and no MODEL_NAME specified."
    echo "Please either:"
    echo "  1. Mount a volume with a saved model at ${MODEL_PATH}, or"
    echo "  2. Specify a MODEL_NAME environment variable to download a pre-trained model"
    exit 1
  fi
else
  echo "Found existing model at ${MODEL_PATH}"
fi

# Check if using GPU
if [ "${DEVICE}" = "cuda" ]; then
  if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "CUDA is available and enabled"
  else
    echo "WARNING: DEVICE=cuda but CUDA is not available. Falling back to CPU."
    export DEVICE="cpu"
  fi
fi

# Start the inference server
exec gunicorn src.inference:app \
  --bind 0.0.0.0:${PORT} \
  --workers ${WORKERS} \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 300 \
  --log-level info \
  "$@"