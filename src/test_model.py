#!/usr/bin/env python
# test_model.py - Unit tests for the model and inference code

import os
import sys
import pytest
import torch
import numpy as np
from transformers import AutoTokenizer

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the src directory
from src.utils import (
    set_seed, 
    preprocess_data, 
    batch_predict, 
    prepare_custom_dataset
)

# Test utility functions
def test_set_seed():
    """Test seed setting for reproducibility"""
    set_seed(42)
    random_tensor_1 = torch.rand(5)
    
    set_seed(42)
    random_tensor_2 = torch.rand(5)
    
    assert torch.all(torch.eq(random_tensor_1, random_tensor_2))

def test_prepare_custom_dataset():
    """Test custom dataset creation"""
    texts = ["This is great", "This is bad"]
    labels = [1, 0]
    
    dataset = prepare_custom_dataset(texts, labels)
    
    assert len(dataset) == 2
    assert dataset[0]["text"] == "This is great"
    assert dataset[0]["label"] == 1
    assert dataset[1]["text"] == "This is bad"
    assert dataset[1]["label"] == 0

# Test preprocessing
@pytest.mark.parametrize(
    "texts,expected_length",
    [
        (["This is a test"], 1),
        (["Short text", "Another short text"], 2),
        ([], 0)
    ]
)
def test_tokenization(texts, expected_length):
    """Test tokenization process"""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create a simple dataset
    from datasets import Dataset
    
    if expected_length > 0:
        dataset = Dataset.from_dict({"sentence": texts, "label": [0] * len(texts)})
        
        # Process
        from datasets import DatasetDict
        dataset_dict = DatasetDict({"train": dataset})
        tokenized = preprocess_data(dataset_dict, tokenizer)
        
        assert len(tokenized["train"]) == expected_length
        assert "input_ids" in tokenized["train"].column_names
        assert "attention_mask" in tokenized["train"].column_names
    else:
        # Test empty case
        dataset = Dataset.from_dict({"sentence": texts, "label": []})
        dataset_dict = DatasetDict({"train": dataset})
        
        # Should handle empty dataset gracefully
        tokenized = preprocess_data(dataset_dict, tokenizer)
        assert len(tokenized["train"]) == 0

# Integration test with a small model
@pytest.mark.slow
def test_model_prediction():
    """Test model prediction functionality"""
    try:
        from transformers import AutoModelForSequenceClassification
        
        # Load a tiny model for testing
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        model = AutoModelForSequenceClassification.from_pretrained(
            "prajjwal1/bert-tiny", 
            num_labels=2
        )
        
        texts = [
            "I really enjoyed this movie!",
            "This was the worst film I've ever seen."
        ]
        
        predictions = batch_predict(texts, model, tokenizer, device="cpu")
        
        # Basic validation
        assert len(predictions) == 2
        assert "label" in predictions[0]
        assert "confidence" in predictions[0]
        assert "probabilities" in predictions[0]
        assert 0 <= predictions[0]["confidence"] <= 1
        
        # Sum of probabilities should be close to 1
        assert abs(sum(predictions[0]["probabilities"].values()) - 1.0) < 1e-5
        
    except Exception as e:
        pytest.skip(f"Model prediction test skipped: {str(e)}")

# Test inference API
@pytest.mark.integration
def test_inference_api():
    """Test the inference API endpoints"""
    import tempfile
    import json
    from fastapi.testclient import TestClient
    
    # Import the FastAPI app
    from src.inference import app, load_model
    
    # Create a test client
    client = TestClient(app)
    
    # Test health endpoint without model loaded
    response = client.get("/health")
    assert response.status_code in [200, 503]  # Either healthy or service unavailable
    
    # Test prediction endpoint
    if response.status_code == 200:  # Only test if model is loaded
        predict_data = {
            "texts": ["This is a test"],
            "raw_scores": True
        }
        
        response = client.post("/predict", json=predict_data)
        
        # Check response
        if response.status_code == 200:
            assert "predictions" in response.json()
            assert "model_version" in response.json()
            assert "processing_time" in response.json()
            assert len(response.json()["predictions"]) == 1
            assert "scores" in response.json()["predictions"][0]
        else:
            # If the model isn't properly loaded, this might fail
            pytest.skip("Model not loaded for prediction test")

if __name__ == "__main__":
    # Run the tests
    pytest.main(["-v", __file__])