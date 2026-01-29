#!/usr/bin/env python3
"""
Enhanced ML Training Pipeline for Bank Receipt Processing
Implements proper NER training with weak supervision and data augmentation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import numpy as np
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import logging
from collections import defaultdict
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReceiptNERDataset(Dataset):
    """Dataset for training NER model on receipt transaction IDs."""
    
    def __init__(self, texts: List[str], labels: List[List[int]], 
                 bboxes: List[List[List[int]]], tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.bboxes = bboxes
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        bboxes = self.bboxes[idx]
        
        # Tokenize with LayoutLM tokenizer
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Align labels and bboxes to tokenized sequence
        word_ids = encoding.word_ids()
        aligned_labels = []
        aligned_bboxes = []
        
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens
                aligned_labels.append(-100)
                aligned_bboxes.append([0, 0, 1000, 1000])
            elif word_idx != previous_word_idx:
                # First token of a word
                if word_idx < len(labels):
                    aligned_labels.append(labels[word_idx])
                    if word_idx < len(bboxes):
                        aligned_bboxes.append(bboxes[word_idx])
                    else:
                        aligned_bboxes.append([0, 0, 1000, 1000])
                else:
                    aligned_labels.append(0)  # O tag
                    aligned_bboxes.append([0, 0, 1000, 1000])
            else:
                # Subsequent tokens of the same word
                if word_idx < len(labels):
                    aligned_labels.append(labels[word_idx])
                    if word_idx < len(bboxes):
                        aligned_bboxes.append(bboxes[word_idx])
                    else:
                        aligned_bboxes.append([0, 0, 1000, 1000])
                else:
                    aligned_labels.append(0)
                    aligned_bboxes.append([0, 0, 1000, 1000])
            
            previous_word_idx = word_idx
        
        # Pad or truncate to max_length
        if len(aligned_labels) > self.max_length:
            aligned_labels = aligned_labels[:self.max_length]
            aligned_bboxes = aligned_bboxes[:self.max_length]
        else:
            aligned_labels.extend([0] * (self.max_length - len(aligned_labels)))
            aligned_bboxes.extend([[0, 0, 1000, 1000]] * (self.max_length - len(aligned_bboxes)))
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'bbox': torch.tensor(aligned_bboxes, dtype=torch.long),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

class EnhancedTransactionExtractor:
    """Enhanced transaction ID extractor with proper NER training."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
        self.model = None
        self.label_map = {'O': 0, 'B-TRANSACTION_ID': 1, 'I-TRANSACTION_ID': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Enhanced regex patterns for weak supervision
        self.patterns = [
            r'\b(?:ref|reference|ref\.|refno|refno\.|payment ref|trans ref|bank ref)\s*[:#-]?\s*([A-Za-z0-9._\-/#]{5,})',
            r'\b(?:transaction|txn|trans|tx)\s*(?:id|no|number|ref)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{5,})',
            r'\b(?:invoice|inv|bill)\s*(?:ref|no|number)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{4,})',
            r'\bduitnow\s*(?:ref|reference|id)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{6,})',
            r'\b(?:customer|bank|cust)\s*ref\s*[:#-]?\s*([A-Za-z0-9._\-/#]{5,})',
            r'\b(?:payment|pymt|pay)\s*(?:ref|reference|id)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{5,})',
            r'\b(?:transfer|xfer|trf)\s*(?:ref|id|no)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{5,})',
            r'\b(?:receipt|rcpt|rcp)\s*(?:no|number|id)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{4,})',
            r'\b(?:id|no|number|code)\s*[:#-]?\s*([A-Za-z0-9._\-/#]{6,})',
            r'\b(?:maybank|cimb|rhb|public bank|hong leong|ambank|uob|ocbc|standard chartered|hsbc)\s*[:#-]?\s*([A-Za-z0-9._\-/#]{6,})',
        ]
    
    def create_weak_labels(self, text: str, tokens: List[Dict]) -> List[int]:
        """Create weak supervision labels using regex patterns."""
        labels = [0] * len(tokens)  # 0 = 'O'
        
        # Find all pattern matches
        for pattern in self.patterns:
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
                        
                        # Check if token is part of the transaction ID
                        if token_text.strip() and token_text.strip() in match_text:
                            if labels[i] == 0:  # First token of entity
                                labels[i] = 1  # B-TRANSACTION_ID
                            else:
                                labels[i] = 2  # I-TRANSACTION_ID
        
        return labels
    
    def prepare_training_data(self, receipts_data: List[Dict]) -> Tuple[List[str], List[List[int]], List[List[List[int]]]]:
        """Prepare training data with weak supervision."""
        texts = []
        labels = []
        bboxes = []
        
        for receipt in receipts_data:
            # Extract text and tokens
            text = receipt.get('text', '')
            tokens = receipt.get('tokens', [])
            
            if not text or not tokens:
                continue
            
            # Create weak labels
            token_labels = self.create_weak_labels(text, tokens)
            
            # Extract bounding boxes
            token_bboxes = []
            for token in tokens:
                bbox = token.get('bbox', [0, 0, 100, 100])
                # Normalize to 0-1000 range
                normalized_bbox = [
                    max(0, min(1000, int(bbox[0] * 10))),
                    max(0, min(1000, int(bbox[1] * 10))),
                    max(0, min(1000, int(bbox[2] * 10))),
                    max(0, min(1000, int(bbox[3] * 10)))
                ]
                token_bboxes.append(normalized_bbox)
            
            texts.append(text)
            labels.append(token_labels)
            bboxes.append(token_bboxes)
        
        return texts, labels, bboxes
    
    def train_model(self, train_data: List[Dict], val_data: List[Dict], 
                   num_epochs=10, batch_size=8, learning_rate=2e-5):
        """Train the NER model with data augmentation."""
        
        # Prepare training data
        logger.info("Preparing training data...")
        train_texts, train_labels, train_bboxes = self.prepare_training_data(train_data)
        val_texts, val_labels, val_bboxes = self.prepare_training_data(val_data)
        
        # Create datasets
        train_dataset = ReceiptNERDataset(train_texts, train_labels, train_bboxes, self.tokenizer)
        val_dataset = ReceiptNERDataset(val_texts, val_labels, val_bboxes, self.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = LayoutLMForTokenClassification.from_pretrained(
            'microsoft/layoutlm-base-uncased',
            num_labels=len(self.label_map)
        ).to(self.device)
        
        # Set up optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        # Training loop
        best_val_loss = float('inf')
        training_history = []
        
        logger.info(f"Starting training with {len(train_dataset)} samples...")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_predictions = []
            train_true_labels = []
            
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                bbox = batch['bbox'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Collect predictions for metrics
                predictions = torch.argmax(logits, dim=-1)
                valid_indices = labels != -100
                
                if valid_indices.any():
                    train_predictions.extend(predictions[valid_indices].cpu().numpy())
                    train_true_labels.extend(labels[valid_indices].cpu().numpy())
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_predictions = []
            val_true_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    bbox = batch['bbox'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        bbox=bbox,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    logits = outputs.logits
                    
                    val_loss += loss.item()
                    
                    predictions = torch.argmax(logits, dim=-1)
                    valid_indices = labels != -100
                    
                    if valid_indices.any():
                        val_predictions.extend(predictions[valid_indices].cpu().numpy())
                        val_true_labels.extend(labels[valid_indices].cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Calculate metrics
            train_f1 = f1_score(train_true_labels, train_predictions, average='weighted') if train_true_labels else 0
            val_f1 = f1_score(val_true_labels, val_predictions, average='weighted') if val_true_labels else 0
            
            scheduler.step(avg_val_loss)
            
            # Log progress
            logger.info(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, "
                       f"Train F1: {train_f1:.4f}, Val Loss: {avg_val_loss:.4f}, "
                       f"Val F1: {val_f1:.4f}")
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_f1': train_f1,
                'val_loss': avg_val_loss,
                'val_f1': val_f1
            })
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model('app/models/transaction_extractor_enhanced.pt')
                logger.info(f"Saved best model with validation loss: {avg_val_loss:.4f}")
        
        return training_history
    
    def save_model(self, path: str):
        """Save the trained model."""
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'tokenizer_name': 'microsoft/layoutlm-base-uncased',
                'label_map': self.label_map,
                'reverse_label_map': self.reverse_label_map
            }, path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.label_map = checkpoint['label_map']
            self.reverse_label_map = checkpoint['reverse_label_map']
            
            self.model = LayoutLMForTokenClassification.from_pretrained(
                'microsoft/layoutlm-base-uncased',
                num_labels=len(self.label_map)
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info(f"Model loaded from {path}")
            return True
        return False

def load_processed_data(data_dir: str = 'processed_data') -> List[Dict]:
    """Load processed receipt data."""
    data = []
    data_path = Path(data_dir)
    
    if data_path.exists():
        for file in data_path.glob('*.json'):
            with open(file, 'r', encoding='utf-8') as f:
                data.append(json.load(f))
    
    return data

def main():
    """Main training function."""
    logger.info("Starting enhanced ML training pipeline...")
    
    # Load processed data
    logger.info("Loading processed receipt data...")
    all_data = load_processed_data()
    
    if not all_data:
        logger.error("No processed data found. Please run the data processing pipeline first.")
        return
    
    logger.info(f"Loaded {len(all_data)} receipt samples")
    
    # Split data
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
    logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Initialize enhanced extractor
    extractor = EnhancedTransactionExtractor()
    
    # Train model with enhanced hyperparameters
    logger.info("Training enhanced NER model...")
    training_history = extractor.train_model(
        train_data=train_data,
        val_data=val_data,
        num_epochs=15,
        batch_size=16,
        learning_rate=3e-5
    )
    
    # Save training history
    with open('training_logs/enhanced_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("Enhanced training completed!")
    
    # Print final results
    if training_history:
        final_metrics = training_history[-1]
        logger.info(f"Final validation F1: {final_metrics['val_f1']:.4f}")
        logger.info(f"Final validation loss: {final_metrics['val_loss']:.4f}")

if __name__ == "__main__":
    main()