#!/usr/bin/env python
# utils.py - Utility functions for model training and inference

import os
import random
import json
import logging
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from typing import Dict, List, Any, Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def preprocess_data(dataset: DatasetDict, tokenizer, max_length: int = 128):
    """
    Tokenize and preprocess dataset
    
    Args:
        dataset: HuggingFace dataset
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Tokenized dataset
    """
    def tokenize_function(examples):
        # For single sentence classification
        if "sentence" in examples:
            return tokenizer(
                examples["sentence"],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
        # For two-sentence classification (e.g., MNLI)
        elif "premise" in examples and "hypothesis" in examples:
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
        # For general text classification, adjust based on your dataset
        else:
            text_column = next(col for col in examples.keys() if col not in ["label", "idx"])
            return tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
    
    # Apply tokenization to each split
    tokenized_dataset = {}
    for split in dataset.keys():
        tokenized_dataset[split] = dataset[split].map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in dataset[split].column_names if col != "label"]
        )
        tokenized_dataset[split] = tokenized_dataset[split].with_format("torch")
    
    return DatasetDict(tokenized_dataset)

def save_model_artifacts(model, tokenizer, metrics: Dict[str, float], output_dir: str):
    """
    Save model, tokenizer, and evaluation metrics
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Model artifacts saved to {output_dir}")

def load_model_artifacts(model_dir: str):
    """
    Load model artifacts (model, tokenizer, and metrics)
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Dictionary containing model, tokenizer, and metrics
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load metrics if available
    metrics = {}
    metrics_path = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "metrics": metrics
    }

def batch_predict(texts: List[str], model, tokenizer, device=None):
    """
    Make batch predictions
    
    Args:
        texts: List of input texts
        model: Model for prediction
        tokenizer: Tokenizer
        device: Device to use for inference
        
    Returns:
        Predictions with labels and probabilities
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Tokenize inputs
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get predicted labels and their probabilities
    predictions = []
    for i, text in enumerate(texts):
        label_id = torch.argmax(probs[i]).item()
        label = model.config.id2label[label_id]
        confidence = probs[i][label_id].item()
        
        # Add all class probabilities
        class_probs = {model.config.id2label[j]: probs[i][j].item() 
                      for j in range(len(model.config.id2label))}
        
        predictions.append({
            "text": text,
            "label": label,
            "confidence": confidence,
            "probabilities": class_probs
        })
    
    return predictions

def prepare_custom_dataset(texts: List[str], labels: List[int]):
    """
    Create a custom dataset from texts and labels
    
    Args:
        texts: List of input texts
        labels: List of labels
        
    Returns:
        Dataset object
    """
    return Dataset.from_dict({"text": texts, "label": labels})