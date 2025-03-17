#!/usr/bin/env python
# train.py - Model training script for BERT-based model

import os
import argparse
import logging
import json
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
from utils import preprocess_data, set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a BERT model for text classification")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", 
                        help="Name of the pre-trained model to use")
    parser.add_argument("--dataset_name", type=str, default="glue", 
                        help="Name of the dataset to use")
    parser.add_argument("--dataset_config", type=str, default="sst2", 
                        help="Configuration of the dataset to use")
    parser.add_argument("--output_dir", type=str, default="./model/artifacts", 
                        help="Directory to save the model and results")
    parser.add_argument("--num_train_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training and evaluation")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Whether to use Weights & Biases for tracking")
    parser.add_argument("--wandb_project", type=str, default="llm-enterprise", 
                        help="W&B project name")
    parser.add_argument("--fp16", action="store_true", 
                        help="Whether to use mixed precision training")
    return parser.parse_args()

def compute_metrics(pred):
    """
    Compute evaluation metrics for model predictions
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}/{args.dataset_config}")
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    
    # Load tokenizer and preprocess data
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Preprocess the dataset
    tokenized_dataset = preprocess_data(dataset, tokenizer)
    
    # Determine number of labels
    if args.dataset_config == "sst2":
        num_labels = 2
    else:
        # Get from dataset info
        num_labels = len(set(tokenized_dataset["train"]["label"]))
    
    # Load pre-trained model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=num_labels
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=args.fp16,
        report_to="wandb" if args.use_wandb else "none",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate the model
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    
    # Save evaluation results
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
    
    # Save model and tokenizer
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save model config and training arguments
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    logger.info("Training completed!")
    
    # End W&B run if used
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()