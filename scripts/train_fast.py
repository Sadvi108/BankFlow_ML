import torch
import logging
import random
from pathlib import Path
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastTransactionDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = sample['labels'] + [0] * (self.max_length - len(sample['labels']))
        labels = labels[:self.max_length]
        
        bboxes = sample['bboxes'] + [[0, 0, 1000, 1000]] * (self.max_length - len(sample['bboxes']))
        bboxes = bboxes[:self.max_length]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'bbox': torch.tensor(bboxes),
            'labels': torch.tensor(labels)
        }

def create_focused_training_data():
    """Create focused training data with real transaction ID patterns"""
    samples = []
    
    # Real transaction ID patterns from your receipts
    patterns = [
        # Maybank
        "MYCN251031853500", "MYCN251031291491", "MYCN251031842914",
        # CIMB  
        "B10-2510-625105", "B10-2510-240816", "B10-2510-409674",
        # Public Bank
        "PBB251031758202", "PBB251031580390", "PBB251031407081",
        # RHB
        "RHB251031180741", "RHB251031943944", "RHB251031798726",
        # HSBC
        "HSBC251031529613", "HSBC251031645651", "HSBC251031701530",
        # UOB
        "UOB251031200108", "UOB251031806076", "UOB251031499003",
        # Standard Chartered
        "SCB251031744606", "SCB251031911983", "SCB251031295446",
        # DuitNow
        "DN251031333313", "DN251031565451", "DN251031915330"
    ]
    
    for i, tid in enumerate(patterns):
        # Create realistic receipt text with transaction ID
        templates = [
            f"Transfer Reference: {tid} Status: Successful",
            f"Transaction ID: {tid} Amount: RM100.00",
            f"Payment Reference {tid} Completed",
            f"Receipt {tid} Transfer Done",
            f"Bank Transfer {tid} Processed"
        ]
        
        text = templates[i % len(templates)]
        words = text.split()
        
        # Create tokens with positions
        tokens = []
        pos = 0
        for word in words:
            start = text.find(word, pos)
            end = start + len(word)
            tokens.append({
                'text': word,
                'start': start,
                'end': end,
                'bbox': [start, 0, end, 20]
            })
            pos = end
        
        # Create labels - mark transaction ID tokens as 1
        labels = [0] * len(tokens)
        for j, token in enumerate(tokens):
            if token['text'] == tid:
                labels[j] = 1
        
        # Create bounding boxes
        bboxes = [[max(0, min(1000, int(t['bbox'][0] * 10))), 0, 1000, 1000] for t in tokens]
        
        samples.append({
            'text': text,
            'labels': labels,
            'bboxes': bboxes,
            'tokens': tokens,
            'transaction_id': tid
        })
    
    return samples

def train_fast():
    """Fast training focused on transaction ID recognition"""
    logger.info("Creating focused training data...")
    samples = create_focused_training_data()
    logger.info(f"Training with {len(samples)} focused samples")
    
    # Initialize tokenizer and model
    tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased', use_fast=False)
    model = LayoutLMForTokenClassification.from_pretrained('microsoft/layoutlm-base-uncased', num_labels=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    # Create dataset and dataloader
    dataset = FastTransactionDataset(samples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    # Training loop - fewer epochs, early stopping
    best_loss = float('inf')
    patience = 2
    no_improve = 0
    
    for epoch in range(5):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bbox = batch['bbox'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f'Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}')
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience or avg_loss < 0.005:
            logger.info(f'Early stopping at epoch {epoch+1}')
            break
    
    # Save model
    Path('app/models').mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_name': 'microsoft/layoutlm-base-uncased',
        'model_config': model.config.to_dict()
    }, 'app/models/transaction_extractor_fast.pt')
    
    logger.info("Fast model training completed!")
    return model, tokenizer

if __name__ == "__main__":
    train_fast()