# LLM Container Quick Start Guide

This guide explains how to containerize and deploy a pre-trained or fine-tuned language model for inference using Docker.

## Getting Started

### Prerequisites

- Docker and Docker Compose installed
- (Optional) NVIDIA GPU with CUDA support for faster inference
- (Optional) Pre-trained or fine-tuned model

### Basic Usage

1. Build and start the container:

```bash
docker-compose up --build
```

2. The service will automatically download a small default model (`prajjwal1/bert-tiny`) and start the inference API on port 8080.

3. Test the API:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["This is a great movie!", "I did not enjoy this film at all."]}'
```

## Using Your Own Model

### Option 1: Specify a HuggingFace Model

You can use any model from the HuggingFace Hub by setting the `MODEL_NAME` environment variable:

```yaml
environment:
  - MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
```

### Option 2: Mount a Local Model

If you have a fine-tuned model saved locally:

1. Uncomment and modify the volume mount in `docker-compose.yml`:

```yaml
volumes:
  - ./path/to/my-model:/app/model
```

2. Remove or comment out the `MODEL_NAME` environment variable

## Configuration Options

You can customize the container behavior through environment variables in `docker-compose.yml`:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | HuggingFace model to download | `prajjwal1/bert-tiny` |
| `DEVICE` | Device to use for inference (`cpu` or `cuda`) | `cpu` |
| `PORT` | Port to expose the API | `8080` |
| `WORKERS` | Number of Gunicorn workers | `1` |
| `NUM_LABELS` | Number of classification labels | `2` |

## API Endpoints

The following endpoints are available:

- `GET /health` - Check if the service is healthy
- `GET /model-info` - Get information about the loaded model
- `POST /predict` - Make predictions on text inputs
- `POST /batch-predict` - Make predictions on a batch of text inputs with IDs
- `POST /reload-model` - Reload the model (e.g., after updating the mounted model)

## Examples

### Simple Prediction Request

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["This is a great movie!"],
    "raw_scores": false
  }'
```

### Batch Prediction with IDs

```bash
curl -X POST http://localhost:8080/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"id": "text1", "text": "This is a great movie!"},
      {"id": "text2", "text": "I did not enjoy this film at all."}
    ],
    "raw_scores": true
  }'
```

## Performance Tuning

For production deployments, consider:

1. Using GPU acceleration by setting `DEVICE=cuda`
2. Increasing the number of workers based on your CPU cores
3. Adjusting resource limits in the Docker Compose file
4. Uncommenting the Prometheus and Grafana services for monitoring

## Troubleshooting

- **Container fails to start**: Check the logs with `docker-compose logs`
- **Model download fails**: Ensure internet connectivity or use a local model
- **Out of memory errors**: Reduce the number of workers or increase memory limits
