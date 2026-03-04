#!/usr/bin/env python3
"""
Training script using successful extractions as positive examples
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
import json
import logging
from pathlib import Path
import re
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_data_from_results():
    """Load training data from successful extractions in test results."""
    results_file = Path('enhanced_test_results.json')
    
    if not results_file.exists():
        logger.error("No test results file found!")
        return []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    training_samples = []
    
    for result in results.get('detailed_results', []):
        if result['status'] == 'SUCCESS' and result.get('transaction_id'):
            # Extract the transaction ID and filename
            transaction_id = result['transaction_id']
            if isinstance(transaction_id, dict):
                transaction_id = transaction_id.get('best_match')
            
            if not transaction_id or len(str(transaction_id)) < 6:
                continue
            
            filename = result['filename']
            
            # Find the original receipt file
            receipt_file = Path('Receipts') / filename
            if not receipt_file.exists():
                logger.warning(f"Receipt file not found: {receipt_file}")
                continue
            
            # Load the processed data for this receipt
            full_result = result.get('full_result', {})
            ocr_text = full_result.get('fields', {}).get('meta', {}).get('ocr_text', '')
            ocr_tokens = full_result.get('fields', {}).get('meta', {}).get('ocr_tokens', [])
            
            if not ocr_text:
                # Try to get OCR text from the processed file
                processed_path = full_result.get('meta', {}).get('processed_path')
                if processed_path and Path(processed_path).exists():
                    # Try to load OCR data from processed file
                    ocr_file = Path(processed_path).with_suffix('.json')
                    if ocr_file.exists():
                        with open(ocr_file, 'r') as f:
                            ocr_data = json.load(f)
                            ocr_text = ocr_data.get('text', '')
                            ocr_tokens = ocr_data.get('tokens', [])
            
            if not ocr_text:
                logger.warning(f"No OCR text found for {filename}")
                continue
            
            # Create training sample
            sample = create_training_sample_from_success(
                ocr_text, ocr_tokens, transaction_id, filename
            )
            
            if sample:
                training_samples.append(sample)
                logger.info(f"Created training sample from {filename}: {transaction_id}")
    
    return training_samples

def create_training_sample_from_success(ocr_text, ocr_tokens, transaction_id, filename):
    """Create a training sample from a successful extraction."""
    if not ocr_text or not transaction_id:
        return None
    
    # If no tokens, create simple tokens from text
    if not ocr_tokens:
        words = ocr_text.split()
        ocr_tokens = []
        current_pos = 0
        
        for word in words:
            word_start = ocr_text.find(word, current_pos)
            if word_start == -1:
                continue
            word_end = word_start + len(word)
            
            ocr_tokens.append({
                'text': word,
                'start': word_start,
                'end': word_end,
                'bbox': [word_start, 0, word_end, 20]  # Simple bbox
            })
            current_pos = word_end
    
    if not ocr_tokens:
        return None
    
    # Create labels based on the successful transaction ID
    labels = create_labels_from_success(ocr_text, ocr_tokens, transaction_id)
    
    # Count positive labels
    positive_count = sum(1 for label in labels if label != 0)
    if positive_count == 0:
        logger.warning(f"No positive labels created for transaction ID: {transaction_id}")
        return None
    
    # Extract bounding boxes
    bboxes = []
    for token in ocr_tokens:
        bbox = token.get('bbox', [0, 0, 100, 100])
        # Normalize to 0-1000 range for LayoutLM
        normalized_bbox = [
            max(0, min(1000, int(bbox[0] * 0.1))),
            max(0, min(1000, int(bbox[1] * 10))),
            max(0, min(1000, int(bbox[2] * 0.1))),
            max(0, min(1000, int(bbox[3] * 10)))
        ]
        bboxes.append(normalized_bbox)
    
    return {
        'text': ocr_text,
        'labels': labels,
        'bboxes': bboxes,
        'tokens': ocr_tokens,
        'id': filename,
        'transaction_id': transaction_id
    }

def create_labels_from_success(text, tokens, transaction_id):
    """Create labels from a successful transaction ID extraction."""
    labels = [0] * len(tokens)  # 0 = 'O'
    
    # Find the transaction ID in the text
    transaction_id_str = str(transaction_id).strip()
    if not transaction_id_str:
        return labels
    
    # Try to find exact match first
    start_pos = text.find(transaction_id_str)
    if start_pos != -1:
        end_pos = start_pos + len(transaction_id_str)
        
        # Label tokens that overlap with this range
        first_token_idx = None
        last_token_idx = None
        
        for i, token in enumerate(tokens):
            token_start = token.get('start', 0)
            token_end = token.get('end', len(token['text']))
            
            # Check if token overlaps with transaction ID
            if (token_start >= start_pos and token_start < end_pos) or \
               (token_end > start_pos and token_end <= end_pos):
                
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
            
            return labels
    
    # If exact match not found, try to find partial matches
    # Look for the core part of the transaction ID (remove prefixes/suffixes)
    core_id = re.sub(r'^[A-Z]{2,4}[\-_]', '', transaction_id_str)  # Remove prefixes like "MYCN-"
    core_id = re.sub(r'[\-_][A-Z]{2,4}$', '', core_id)  # Remove suffixes
    
    if core_id and core_id != transaction_id_str:
        start_pos = text.find(core_id)
        if start_pos != -1:
            end_pos = start_pos + len(core_id)
            
            # Label tokens that overlap with this range
            first_token_idx = None
            last_token_idx = None
            
            for i, token in enumerate(tokens):
                token_start = token.get('start', 0)
                token_end = token.get('end', len(token['text']))
                
                # Check if token overlaps with core ID
                if (token_start >= start_pos and token_start < end_pos) or \
                   (token_end > start_pos and token_end <= end_pos):
                    
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
                
                return labels
    
    # If still no match, try to find similar patterns
    # Look for alphanumeric sequences that might be the transaction ID
    alphanumeric_pattern = r'[A-Z0-9]{6,25}'
    matches = re.finditer(alphanumeric_pattern, text, re.IGNORECASE)
    
    best_match = None
    best_similarity = 0
    
    for match in matches:
        match_text = match.group()
        # Calculate similarity with transaction ID
        similarity = calculate_similarity(match_text, transaction_id_str)
        
        if similarity > best_similarity and similarity > 0.7:  # 70% similarity threshold
            best_similarity = similarity
            best_match = match
    
    if best_match:
        start_pos = best_match.start()
        end_pos = best_match.end()
        
        # Label tokens that overlap with this range
        first_token_idx = None
        last_token_idx = None
        
        for i, token in enumerate(tokens):
            token_start = token.get('start', 0)
            token_end = token.get('end', len(token['text']))
            
            # Check if token overlaps with best match
            if (token_start >= start_pos and token_start < end_pos) or \
               (token_end > start_pos and token_end <= end_pos):
                
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
            
            logger.info(f"Found similar match: {best_match.group()} (similarity: {best_similarity:.2f})")
            return labels
    
    logger.warning(f"Could not find transaction ID '{transaction_id_str}' in text")
    return labels

def calculate_similarity(text1, text2):
    """Calculate similarity between two texts."""
    if not text1 or not text2:
        return 0
    
    # Convert to uppercase and remove non-alphanumeric
    clean1 = re.sub(r'[^A-Z0-9]', '', text1.upper())
    clean2 = re.sub(r'[^A-Z0-9]', '', text2.upper())
    
    if not clean1 or not clean2:
        return 0
    
    # Calculate character-level similarity
    common_chars = 0
    for char in clean1:
        if char in clean2:
            common_chars += 1
    
    # Calculate similarity score
    max_len = max(len(clean1), len(clean2))
    if max_len == 0:
        return 0
    
    similarity = common_chars / max_len
    
    # Bonus for length similarity
    len_diff = abs(len(clean1) - len(clean2))
    len_bonus = max(0, 1 - (len_diff / max_len)) * 0.3
    
    return similarity + len_bonus

def train_improved_model():
    """Train an improved NER model using successful extractions."""
    logger.info("Loading training data from successful extractions...")
    training_samples = load_training_data_from_results()
    
    if len(training_samples) < 2:
        logger.error("Not enough training samples from successful extractions!")
        return None
    
    logger.info(f"Prepared {len(training_samples)} training samples from successful extractions")
    
    # Show sample information
    for i, sample in enumerate(training_samples[:3]):  # Show first 3 samples
        logger.info(f"Sample {i+1}: {sample['id']} -> {sample['transaction_id']}")
        logger.info(f"  Text length: {len(sample['text'])}")
        logger.info(f"  Tokens: {len(sample['tokens'])}")
        logger.info(f"  Positive labels: {sum(1 for l in sample['labels'] if l != 0)}")
    
    # Analyze label distribution
    all_labels = []
    for sample in training_samples:
        all_labels.extend(sample['labels'])
    
    label_counts = Counter(all_labels)
    logger.info(f"Label distribution: {dict(label_counts)}")
    
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
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Training parameters
    num_epochs = 20  # More epochs for small dataset
    batch_size = 2   # Small batch size for stability
    gradient_accumulation_steps = 4  # Simulate larger batch
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
        if avg_loss < 0.02 and epoch > 10:
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
            'num_samples': len(training_samples),
            'label_distribution': dict(label_counts),
            'training_method': 'successful_extractions'
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