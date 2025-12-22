#!/usr/bin/env python3
"""
Improved training script for high-accuracy NER model
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
import json
import logging
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from collections import Counter

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

def create_better_weak_labels(text, tokens):
    """
    Create cleaner weak supervision labels by:
    1. Preferring longer, more complex matches
    2. Avoiding matches that are just headers/labels
    3. Using bank-specific patterns
    """
    # Enhanced patterns with stricter validation
    patterns = [
        # Bank-specific transaction ID formats
        (r'\bMaybank[\s:]*([A-Z0-9]{8,})', 'Maybank'),
        (r'\bCIMB[\s:]*([A-Z0-9]{6,})', 'CIMB'),
        (r'\bPublic Bank[\s:]*([A-Z0-9]{8,})', 'Public Bank'),
        (r'\bRHB[\s:]*([A-Z0-9]{6,})', 'RHB'),
        (r'\bHSBC[\s:]*([A-Z0-9]{8,})', 'HSBC'),
        (r'\bUOB[\s:]*([A-Z0-9]{6,})', 'UOB'),
        (r'\bStandard Chartered[\s:]*([A-Z0-9]{8,})', 'Standard Chartered'),
        (r'\bAmbank[\s:]*([A-Z0-9]{6,})', 'Ambank'),
        (r'\bAffin Bank[\s:]*([A-Z0-9]{6,})', 'Affin Bank'),
        (r'\bHong Leong Bank[\s:]*([A-Z0-9]{8,})', 'Hong Leong Bank'),
        (r'\bCitibank[\s:]*([A-Z0-9]{8,})', 'Citibank'),
        
        # DuitNow specific patterns
        (r'DuitNow[\s:]*([A-Z0-9]{10,})', 'DuitNow'),
        (r'DuitNow Reference[\s:]*([A-Z0-9]{8,})', 'DuitNow'),
        
        # General transaction patterns (only if they look like real IDs)
        (r'\b(?:ref|reference|ref\.)(?!.*(?:number|no|id))[\s:]*([A-Z0-9]{6,})', 'General'),
        (r'\b(?:transaction|txn|trans)(?!.*(?:number|no|id))[\s:]*([A-Z0-9]{6,})', 'General'),
        (r'\b(?:payment|transfer)(?!.*(?:ref|id))[\s:]*([A-Z0-9]{6,})', 'General'),
        
        # Invoice numbers that look like transaction IDs
        (r'\b(?:invoice|inv)[\s:]*([A-Z0-9]{8,})', 'Invoice'),
    ]
    
    labels = [0] * len(tokens)  # 0 = 'O'
    
    # Find all matches first
    all_matches = []
    for pattern, source in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            match_text = match.group(1)
            
            # Validate the match
            if not is_valid_transaction_id(match_text):
                continue
                
            match_start = match.start(1)
            match_end = match.end(1)
            all_matches.append({
                'text': match_text,
                'start': match_start,
                'end': match_end,
                'source': source,
                'length': len(match_text),
                'complexity': calculate_complexity(match_text)
            })
    
    # Sort by complexity and length (prefer longer, more complex matches)
    all_matches.sort(key=lambda x: (x['complexity'], x['length']), reverse=True)
    
    # Apply labels for the best matches, avoiding overlaps
    used_ranges = set()
    
    for match in all_matches:
        match_range = (match['start'], match['end'])
        
        # Skip if this range overlaps with already used ranges
        if any(start < match['end'] and end > match['start'] for start, end in used_ranges):
            continue
            
        # Find tokens that overlap with this match
        first_token_idx = None
        last_token_idx = None
        
        for i, token in enumerate(tokens):
            token_start = token.get('start', 0)
            token_end = token.get('end', len(token['text']))
            
            # Check if token overlaps with match
            if (token_start >= match['start'] and token_start < match['end']) or \
               (token_end > match['start'] and token_end <= match['end']):
                
                if first_token_idx is None:
                    first_token_idx = i
                last_token_idx = i
        
        # Label the tokens
        if first_token_idx is not None and last_token_idx is not None:
            for i in range(first_token_idx, last_token_idx + 1):
                if i == first_token_idx:
                    labels[i] = 1  # B-TRANSACTION_ID
                else:
                    labels[i] = 2  # I-TRANSACTION_ID
            
            used_ranges.add(match_range)
    
    return labels

def is_valid_transaction_id(text):
    """Validate if text looks like a real transaction ID."""
    if len(text) < 6 or len(text) > 30:
        return False
    
    # Must contain at least 2 digits
    if sum(c.isdigit() for c in text) < 2:
        return False
    
    # Must be mostly alphanumeric
    alphanumeric_count = sum(c.isalnum() for c in text)
    if alphanumeric_count < len(text) * 0.8:
        return False
    
    # Avoid common false positives
    false_positives = {'date', 'time', 'amount', 'total', 'balance', 'account', 'name', 'address'}
    if text.lower() in false_positives:
        return False
    
    # Must have reasonable character distribution
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    
    # Avoid strings that are mostly letters or mostly digits
    if letters > len(text) * 0.9 or digits > len(text) * 0.9:
        return False
    
    return True

def calculate_complexity(text):
    """Calculate complexity score for a transaction ID."""
    score = 0
    
    # Length score
    score += len(text) * 0.1
    
    # Mixed case score
    if any(c.isupper() for c in text) and any(c.islower() for c in text):
        score += 2
    
    # Digit score
    digit_count = sum(c.isdigit() for c in text)
    score += digit_count * 0.5
    
    # Special character score (limited)
    special_count = sum(not c.isalnum() for c in text)
    score += min(special_count, 3) * 0.3
    
    # Pattern score (alternating letters and digits)
    alternations = 0
    for i in range(1, len(text)):
        if (text[i-1].isalpha() and text[i].isdigit()) or (text[i-1].isdigit() and text[i].isalpha()):
            alternations += 1
    score += alternations * 0.2
    
    return score

def prepare_training_data(data):
    """Prepare training data with improved weak supervision."""
    training_samples = []
    
    for receipt in data:
        text = receipt.get('ocr_text', '')
        
        if not text or len(text.strip()) < 10:
            continue
        
        # Use existing OCR tokens if available, otherwise create simple tokens
        tokens = receipt.get('ocr_tokens', [])
        if not tokens:
            # Create tokens from text with basic positions
            words = text.split()
            tokens = []
            current_pos = 0
            
            for word in words:
                word_start = text.find(word, current_pos)
                if word_start == -1:
                    continue
                word_end = word_start + len(word)
                
                tokens.append({
                    'text': word,
                    'start': word_start,
                    'end': word_end,
                    'bbox': [word_start, 0, word_end, 20]  # Simple bbox
                })
                current_pos = word_end
        
        if not tokens:
            continue
        
        # Create improved weak labels
        labels = create_better_weak_labels(text, tokens)
        
        # Count positive labels
        positive_count = sum(1 for label in labels if label != 0)
        logger.info(f"Sample has {positive_count} positive labels out of {len(labels)} tokens")
        
        # Keep samples even with few positive labels for better training
        if positive_count == 0:
            logger.warning(f"No positive labels found for sample, skipping")
            continue
        
        # Extract bounding boxes
        bboxes = []
        for token in tokens:
            bbox = token.get('bbox', [0, 0, 100, 100])
            # Normalize to 0-1000 range for LayoutLM
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
            'bboxes': bboxes,
            'tokens': tokens,
            'id': receipt.get('id', str(hash(text)))
        })
    
    return training_samples

def train_improved_model():
    """Train an improved NER model with better supervision."""
    logger.info("Loading processed data...")
    data = load_processed_data()
    
    if not data:
        logger.error("No processed data found!")
        return None
    
    logger.info(f"Found {len(data)} receipt samples")
    
    # Prepare training data with improved supervision
    logger.info("Preparing training data with improved weak supervision...")
    training_samples = prepare_training_data(data)
    
    if len(training_samples) < 5:
        logger.error("Not enough training samples with positive labels!")
        return None
    
    logger.info(f"Prepared {len(training_samples)} training samples with positive labels")
    
    # Analyze label distribution
    all_labels = []
    for sample in training_samples:
        all_labels.extend(sample['labels'])
    
    label_counts = Counter(all_labels)
    logger.info(f"Label distribution: {dict(label_counts)}")
    
    # Calculate class weights for imbalanced data
    total_labels = len(all_labels)
    class_weights = {}
    for label, count in label_counts.items():
        if count > 0:
            class_weights[label] = total_labels / (len(label_counts) * count)
        else:
            class_weights[label] = 1.0
    
    logger.info(f"Class weights: {class_weights}")
    
    # Initialize tokenizer and model
    tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased', use_fast=False)
    model = LayoutLMForTokenClassification.from_pretrained(
        'microsoft/layoutlm-base-uncased',
        num_labels=3  # O, B-TRANSACTION_ID, I-TRANSACTION_ID
    )
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    # Loss function with class weights
    class_weights_tensor = torch.tensor([
        class_weights.get(0, 1.0),  # O
        class_weights.get(1, 1.0),  # B-TRANSACTION_ID
        class_weights.get(2, 1.0)   # I-TRANSACTION_ID
    ], dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, ignore_index=-100)
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)  # Lower LR, add weight decay
    
    # Training parameters
    num_epochs = 10  # More epochs
    batch_size = 4   # Small batch size for stability
    gradient_accumulation_steps = 2  # Simulate larger batch
    max_grad_norm = 1.0  # Gradient clipping
    
    logger.info(f"Training for {num_epochs} epochs with batch size {batch_size}")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        # Shuffle training samples
        import random
        random.shuffle(training_samples)
        
        # Process in mini-batches
        for batch_start in range(0, len(training_samples), batch_size):
            batch_end = min(batch_start + batch_size, len(training_samples))
            batch_samples = training_samples[batch_start:batch_end]
            
            # Prepare batch
            batch_input_ids = []
            batch_attention_masks = []
            batch_bboxes = []
            batch_labels = []
            
            for sample in batch_samples:
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
                
                seq_len = encoding['input_ids'].shape[1]
                
                # Align labels to tokenized sequence
                aligned_labels = align_labels_with_tokens(labels, encoding, tokenizer)
                
                # Align bboxes
                aligned_bboxes = align_bboxes_with_tokens(bboxes, encoding, seq_len)
                
                batch_input_ids.append(encoding['input_ids'])
                batch_attention_masks.append(encoding['attention_mask'])
                batch_bboxes.append(torch.tensor([aligned_bboxes]))
                batch_labels.append(torch.tensor([aligned_labels]))
            
            # Stack batch tensors
            batch_input_ids = torch.cat(batch_input_ids, dim=0).to(device)
            batch_attention_masks = torch.cat(batch_attention_masks, dim=0).to(device)
            batch_bboxes = torch.cat(batch_bboxes, dim=0).to(device)
            batch_labels = torch.cat(batch_labels, dim=0).to(device)
            
            # Forward pass
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                bbox=batch_bboxes,
                labels=batch_labels
            )
            
            loss = outputs.loss / gradient_accumulation_steps
            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (num_batches + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Early stopping check
        if avg_loss < 0.1 and epoch > 5:
            logger.info("Early stopping triggered")
            break
    
    # Save model
    model_dir = Path('app/models')
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / 'transaction_extractor_improved.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_name': 'microsoft/layoutlm-base-uncased',
        'training_config': {
            'num_labels': 3,
            'class_weights': class_weights,
            'label_distribution': dict(label_counts)
        }
    }, model_path)
    
    logger.info(f"Improved model saved to {model_path}")
    
    return model

def align_labels_with_tokens(labels, encoding, tokenizer):
    """Align original labels with tokenized sequence."""
    # Get word IDs (mapping from tokens to original words)
    word_ids = encoding.word_ids()
    
    # Create aligned labels
    aligned_labels = []
    previous_word_id = None
    
    for word_id in word_ids:
        if word_id is None:
            # Special tokens
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            # First token of a word
            if word_id < len(labels):
                aligned_labels.append(labels[word_id])
            else:
                aligned_labels.append(0)  # O label
        else:
            # Subsequent tokens of the same word
            if word_id < len(labels) and labels[word_id] != 0:
                aligned_labels.append(2)  # I-TRANSACTION_ID
            else:
                aligned_labels.append(0)  # O label
        
        previous_word_id = word_id
    
    return aligned_labels

def align_bboxes_with_tokens(bboxes, encoding, seq_len):
    """Align bounding boxes with tokenized sequence."""
    word_ids = encoding.word_ids()
    aligned_bboxes = []
    
    for word_id in word_ids:
        if word_id is None or word_id >= len(bboxes):
            # Special tokens or out of range
            aligned_bboxes.append([0, 0, 1000, 1000])
        else:
            aligned_bboxes.append(bboxes[word_id])
    
    return aligned_bboxes

if __name__ == "__main__":
    train_improved_model()