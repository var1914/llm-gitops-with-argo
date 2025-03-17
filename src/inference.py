#!/usr/bin/env python
# inference.py - Inference API for the trained model

import os
import time
import logging
import json
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define request and response models
class PredictionRequest(BaseModel):
    texts: List[str]
    raw_scores: bool = False

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_version: str
    processing_time: float

# Initialize metrics
PREDICTION_COUNT = Counter('prediction_requests_total', 'Total number of prediction requests')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction request latency in seconds')

# Initialize FastAPI app
app = FastAPI(
    title="LLM Inference API",
    description="API for text classification using a fine-tuned BERT model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer globally
model = None
tokenizer = None
classifier = None
model_info = {}

def load_model():
    """
    Load the model and tokenizer from the specified path
    """
    global model, tokenizer, classifier, model_info
    
    # Define model path - default to environment variable or use a default path
    model_path = os.environ.get("MODEL_PATH", "./model/artifacts")
    
    try:
        logger.info(f"Loading model from {model_path}")
        
        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Create text classification pipeline
        classifier = pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer, 
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Load model information
        if os.path.exists(os.path.join(model_path, "training_args.json")):
            with open(os.path.join(model_path, "training_args.json"), "r") as f:
                model_info = json.load(f)
        
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """
    Initialize the model on startup
    """
    if not load_model():
        raise RuntimeError("Failed to load the model")
    
    # Start Prometheus metrics server
    start_http_server(8000)

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_info": model_info}

@app.get("/model-info")
async def get_model_info():
    """
    Return information about the model
    """
    return {
        "model_info": model_info,
        "model_size": sum(p.numel() for p in model.parameters()),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions on the input texts
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Increment request counter
        PREDICTION_COUNT.inc()
        
        # Make predictions
        with PREDICTION_LATENCY.time():
            results = classifier(request.texts, return_all_scores=request.raw_scores)
        
        # Format response
        predictions = []
        for i, result in enumerate(results):
            if request.raw_scores:
                predictions.append({
                    "text": request.texts[i],
                    "scores": {item["label"]: item["score"] for item in result}
                })
            else:
                predictions.append({
                    "text": request.texts[i],
                    "label": result["label"],
                    "score": result["score"]
                })
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            predictions=predictions,
            model_version=model_info.get("model_name", "unknown"),
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-model")
async def reload_model(background_tasks: BackgroundTasks):
    """
    Reload the model - useful for model updates without restarting the service
    """
    background_tasks.add_task(load_model)
    return {"status": "Model reload initiated"}

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8080))
    
    # Run the server
    uvicorn.run("inference:app", host="0.0.0.0", port=port, reload=False)