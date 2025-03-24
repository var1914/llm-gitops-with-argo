# LLM GitOps with ArgoCD Quick Start Guide

This guide explains how to deploy pre-trained or fine-tuned language models for inference using GitOps practices with ArgoCD and a custom LLM Operator.

## Architecture Overview

This repository implements GitOps-based deployment of LLM inference services using:
- **ArgoCD**: For declarative, Git-based delivery of Kubernetes manifests
- **Custom LLM Operator**: From [llm-operator](https://github.com/var1914/llm-operator) repository
- **Kubernetes**: For container orchestration and scaling

## Getting Started

### Prerequisites

- Kubernetes cluster (local or cloud)
- ArgoCD installed on your cluster
- Access to the LLM Operator ([github.com/var1914/llm-operator](https://github.com/var1914/llm-operator))
- (Optional) NVIDIA GPU with CUDA support for faster inference

### Installation Steps

1. **Install the LLM Operator**

   The LLM Operator must be installed first from the separate repository: https://github.com/var1914/llm-operator/blob/main/README.md
   
2. **Deploy using ArgoCD**

   Create an ArgoCD application pointing to this repository:

   ```bash
   kubectl apply -f dev-llm-app.yaml
   ```

   This will automatically deploy:
   - The `llm-dev` namespace
   - The BERT model configuration
   - The LLM inference service

3. **Verify the deployment**

   ```bash
   kubectl get applications -n argocd
   kubectl get llmmodel,llmdeployment -n llm-dev
   ```

## Configuration Options

The deployment is configured through Kubernetes manifests:

### Model Configuration (`models/bert-inference-dev.yaml`)

```yaml
apiVersion: llm.example.com/v1alpha1
kind: LLMModel
metadata:
  name: bert-inference
  namespace: llm-dev
spec:
  modelName: "BERT Inference Service - Dev"
  image: "your-registry/llm-inference-service:latest"
  resources:
    cpu: "0.5"
    memory: "1Gi"
  environmentVariables:
    MODEL_NAME: "prajjwal1/bert-tiny"
    DEVICE: "cpu"
    WORKERS: "1"
    TRANSFORMERS_CACHE: "/app/.cache/huggingface"
```

### Deployment Configuration (`deployments/bert-inference-service-dev.yaml`)

```yaml
apiVersion: llm.example.com/v1alpha1
kind: LLMDeployment
metadata:
  name: bert-inference-service
  namespace: llm-dev
spec:
  modelRef: "bert-inference"
  replicas: 1
  port: 8080
  enablePersistence: true
  persistentVolumeSize: "2Gi"
```

## Using Your Own Model

### Option 1: Specify a HuggingFace Model

You can use any model from the HuggingFace Hub by updating the `MODEL_NAME` environment variable in your model configuration:

```yaml
environmentVariables:
  MODEL_NAME: "distilbert-base-uncased-finetuned-sst-2-english"
```

### Option 2: Use a Custom Container

For models requiring custom preprocessing or postprocessing, build a custom container and push it to your registry:

1. Update the `image` field in the model configuration
2. Update any required environment variables

## API Endpoints

Once deployed, the service exposes the following endpoints:

- `GET /health` - Check if the service is healthy
- `GET /model-info` - Get information about the loaded model
- `POST /predict` - Make predictions on text inputs
- `POST /batch-predict` - Make predictions on a batch of text inputs with IDs
- `POST /reload-model` - Reload the model (e.g., after model update)

## Examples

### Simple Prediction Request

```bash
# Get the service endpoint
SERVICE_URL=$(kubectl get svc -n llm-dev bert-inference-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Send a request
curl -X POST http://$SERVICE_URL:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["This is a great movie!"],
    "raw_scores": false
  }'
```

### Batch Prediction with IDs

```bash
curl -X POST http://$SERVICE_URL:8080/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"id": "text1", "text": "This is a great movie!"},
      {"id": "text2", "text": "I did not enjoy this film at all."}
    ],
    "raw_scores": true
  }'
```

## GitOps Workflow

The GitOps workflow for updating your LLM deployments:

1. Make changes to the YAML files in this repository
2. Commit and push to your Git repository
3. ArgoCD automatically detects changes and syncs the application
4. The LLM Operator reconciles the changes to update your deployments

## Performance Tuning

For production deployments, consider:

1. Using GPU acceleration by setting `DEVICE=cuda` in your model configuration
2. Increasing the number of replicas in the deployment configuration
3. Adjusting resource limits in the model configuration
4. Adding HPA (Horizontal Pod Autoscaler) for automatic scaling
5. Using service meshes for advanced traffic routing and blue/green deployments

## Monitoring

For monitoring, we recommend:

1. Deploying Prometheus and Grafana for metrics collection
2. Setting up alerts for service health and performance
3. Integrating with your existing logging solution

## Troubleshooting

- **Pods fail to start**: Check the logs with `kubectl logs -n llm-dev <pod-name>`
- **Model download fails**: Ensure internet connectivity or use a custom image with pre-loaded models
- **Out of memory errors**: Increase memory limits in the model configuration
- **ArgoCD sync fails**: Check ArgoCD logs and ensure Git repository is accessible
