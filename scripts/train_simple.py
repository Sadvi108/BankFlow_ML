#!/usr/bin/env python3
"""
Simple training script for enhanced ML models
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data():
    """Load processed receipt data from annotations.jsonl."""
    data = []
    annotations_file = Path('data/dataset/annotations.jsonl')
    
    if annotations_file.exists():
        with open(annotations_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    
    return data

def create_weak_labels(text, tokens):
    """Create weak supervision labels using regex patterns."""
    # Enhanced regex patterns
    patterns = [
        r'\b(?:ref|reference|ref\.|refno|refno\.|payment ref|trans ref|bank ref)\s*[:#-]?\s*([A-Za-z0-9._\-/#]{5,})',
        r'\b(?:transaction|txn|trans|tx)\s*(?:id|no|number|ref)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{5,})',
        r'\b(?:invoice|inv|bill)\s*(?:ref|no|number)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{4,})',
        r'\bduitnow\s*(?:ref|reference|id)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{6,})',
        r'\b(?:customer|bank|cust)\s*ref\s*[:#-]?\s*([A-Za-z0-9._\-/#]{5,})',
        r'\b(?:payment|pymt|pay)\s*(?:ref|reference|id)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{5,})',
        r'\b(?:transfer|xfer|trf)\s*(?:ref|id|no)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{5,})',
        r'\b(?:receipt|rcpt|rcp)\s*(?:no|number|id)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{4,})',
        r'\b(?:id|no|number|code)\s*[:#-]?\s*([A-Za-z0-9._\-/#]{6,})',
    ]
    
    import re
    labels = [0] * len(tokens)  # 0 = 'O'
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            match_text = match.group(1)
            match_start = match.start(1)
            match_end = match.end(1)
            
            # Find which tokens overlap with this match
            for i, token in enumerate(tokens):
                token_text = token['text']
                token_start = token.get('start', 0)
                token_end = token.get('end', len(token_text))
                
                # Check if token overlaps with match
                if (token_start >= match_start and token_start < match_end) or \
                   (token_end > match_start and token_end <= match_end):
                    
                    if token_text.strip() and token_text.strip() in match_text:
                        if labels[i] == 0:  # First token of entity
                            labels[i] = 1  # B-TRANSACTION_ID
                        else:
                            labels[i] = 2  # I-TRANSACTION_ID
    
    return labels

def prepare_training_data(data):
    """Prepare training data with weak supervision."""
    training_samples = []
    
    for receipt in data:
        text = receipt.get('ocr_text', '')
        
        # Create simple tokens from text (split by whitespace)
        words = text.split()
        tokens = []
        current_pos = 0
        
        for word in words:
            word_start = text.find(word, current_pos)
            word_end = word_start + len(word)
            
            tokens.append({
                'text': word,
                'start': word_start,
                'end': word_end,
                'bbox': [word_start, 0, word_end, 20]  # Simple bbox
            })
            current_pos = word_end
        
        if not text or not tokens:
            continue
        
        # Create weak labels
        labels = create_weak_labels(text, tokens)
        
        # Extract bounding boxes
        bboxes = []
        for token in tokens:
            bbox = token.get('bbox', [0, 0, 100, 100])
            # Normalize to 0-1000 range
            normalized_bbox = [
                max(0, min(1000, int(bbox[0] * 0.1))),
                max(0, min(1000, int(bbox[1] * 10))),
                max(0, min(1000, int(bbox[2] * 0.1))),
                max(0, min(1000, int(bbox[3] * 10)))
            ]
            bboxes.append(normalized_bbox)
        
        training_samples.append({
            'text': text,
            'labels': labels,
            'bboxes': bboxes
        })
    
    return training_samples

def train_simple_ner_model():
    """Train a simple NER model for transaction extraction."""
    logger.info("Loading processed data...")
    data = load_processed_data()
    
    if not data:
        logger.error("No processed data found!")
        return None
    
    logger.info(f"Found {len(data)} receipt samples")
    
    # Prepare training data
    logger.info("Preparing training data with weak supervision...")
    training_samples = prepare_training_data(data)
    
    if len(training_samples) < 5:
        logger.error("Not enough training samples!")
        return None
    
    logger.info(f"Prepared {len(training_samples)} training samples")
    
    # Initialize tokenizer and model
    tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased', use_fast=False)
    model = LayoutLMForTokenClassification.from_pretrained(
        'microsoft/layoutlm-base-uncased',
        num_labels=3  # O, B-TRANSACTION_ID, I-TRANSACTION_ID
    )
    
    # Simple training loop - process one sample at a time
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    num_epochs = 5
    
    logger.info(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for sample in training_samples:
            text = sample['text']
            labels = sample['labels']
            bboxes = sample['bboxes']
            
            # Tokenize
            encoding = tokenizer(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Simple alignment: truncate or pad labels to match sequence length
            seq_len = encoding['input_ids'].shape[1]
            
            # Convert labels to tensor, pad/truncate to sequence length
            if len(labels) >= seq_len:
                aligned_labels = labels[:seq_len-2] + [0, 0]  # Reserve space for special tokens
            else:
                aligned_labels = labels + [0] * (seq_len - len(labels))
            
            # Handle special tokens (first and last)
            aligned_labels[0] = -100  # [CLS]
            aligned_labels[-1] = -100  # [SEP]
            
            # Convert bboxes to tensor
            if len(bboxes) >= seq_len:
                aligned_bboxes = bboxes[:seq_len]
            else:
                aligned_bboxes = bboxes + [[0, 0, 1000, 1000]] * (seq_len - len(bboxes))
            
            # Forward pass
            outputs = model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                bbox=torch.tensor([aligned_bboxes]),
                labels=torch.tensor([aligned_labels])
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save model
    model_dir = Path('app/models')
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / 'transaction_extractor_enhanced.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_name': 'microsoft/layoutlm-base-uncased'
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    train_simple_ner_model()