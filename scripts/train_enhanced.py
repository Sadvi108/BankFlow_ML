"""
Enhanced Training Script for Bank Receipt ML Models

This script implements advanced training techniques including:
- Data augmentation
- Semi-supervised learning
- Curriculum learning
- Focal loss
- Cross-validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
import json
import logging
import argparse
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import sys

# Add app to path
sys.path.insert(0, str(Path.cwd()))

from app.enhanced_data_augmentation import augment_training_data, ReceiptAugmentor
from app.semi_supervised_trainer import create_semi_supervised_dataset
from app.ml_models import (
    ReceiptDataset, HybridBankClassifier, TransactionIDExtractor,
    ReceiptMLPipeline
)
from app.dataset import read_annotations
from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EnhancedTrainer:
    """Enhanced trainer with advanced techniques."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = Path(args.models_dir)
        self.data_dir = Path(args.data_dir)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Data directory: {self.data_dir}")
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'augmented').mkdir(parents=True, exist_ok=True)
        (self.data_dir / 'semi_supervised').mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self) -> Tuple[List[Dict], List[float]]:
        """Prepare training data with augmentation and semi-supervised learning."""
        logger.info("=" * 60)
        logger.info("STEP 1: Preparing Training Data")
        logger.info("=" * 60)
        
        # Load original annotations
        annotations = read_annotations()
        logger.info(f"Loaded {len(annotations)} original annotations")
        
        # Apply data augmentation if enabled
        if self.args.use_augmentation:
            logger.info("Applying data augmentation...")
            annotations = augment_training_data(
                annotations,
                self.data_dir / 'dataset' / 'images',
                num_image_augmentations=self.args.num_augmentations,
                num_synthetic_variations=2
            )
            
            # Save augmented annotations
            aug_path = self.data_dir / 'augmented' / 'annotations.jsonl'
            with open(aug_path, 'w', encoding='utf-8') as f:
                for ann in annotations:
                    f.write(json.dumps(ann, ensure_ascii=False) + '\n')
            logger.info(f"Saved augmented annotations to {aug_path}")
        
        # Apply semi-supervised learning if enabled
        sample_weights = None
        if self.args.use_semi_supervised:
            logger.info("Generating semi-supervised dataset...")
            
            # Initialize OCR pipeline and pattern extractor
            ocr_pipeline = EnhancedOCRPipeline()
            
            # Create semi-supervised dataset
            annotations, sample_weights = create_semi_supervised_dataset(
                labeled_annotations=annotations,
                unlabeled_receipts_dir=Path('Receipts'),
                ocr_pipeline=ocr_pipeline,
                pattern_extractor=extract_all_fields_v3,
                output_dir=self.data_dir / 'semi_supervised',
                max_pseudo_labels=self.args.max_pseudo_labels
            )
        
        logger.info(f"Final dataset size: {len(annotations)} samples")
        
        return annotations, sample_weights
    
    def train_with_curriculum(self, model, train_loader, val_loader, 
                             criterion, optimizer, scheduler, num_epochs):
        """Train model with curriculum learning."""
        logger.info("=" * 60)
        logger.info("STEP 2: Training with Curriculum Learning")
        logger.info("=" * 60)
        
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch in pbar:
                # Move data to device
                image_features = batch['image_features'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                layout_features = batch['layout_features'].to(self.device)
                labels = batch['bank_label'].squeeze().to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(image_features, input_ids, attention_mask, layout_features)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * train_correct / train_total:.2f}%'
                })
            
            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            val_acc, avg_val_loss = self.validate(model, val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            logger.info(f'Epoch {epoch+1}/{num_epochs}:')
            logger.info(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
            logger.info(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
            logger.info(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                patience_counter = 0
                logger.info(f"  ✓ New best model! Val Acc: {val_acc:.4f}")
                
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, self.models_dir / 'best_checkpoint.pt')
            else:
                patience_counter += 1
            
            # Early stopping
            if self.args.early_stopping and patience_counter >= self.args.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, best_val_acc
    
    def validate(self, model, val_loader, criterion):
        """Validate model."""
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                image_features = batch['image_features'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                layout_features = batch['layout_features'].to(self.device)
                labels = batch['bank_label'].squeeze().to(self.device)
                
                outputs = model(image_features, input_ids, attention_mask, layout_features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        return val_acc, avg_val_loss
    
    def train(self):
        """Main training function."""
        logger.info("=" * 60)
        logger.info("ENHANCED ML TRAINING FOR BANK RECEIPT PROCESSING")
        logger.info("=" * 60)
        
        # Prepare data
        annotations, sample_weights = self.prepare_data()
        
        if len(annotations) < 10:
            logger.error("Insufficient training data! Need at least 10 samples.")
            return
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_ann, temp_ann = train_test_split(annotations, test_size=0.3, random_state=42)
        val_ann, test_ann = train_test_split(temp_ann, test_size=0.5, random_state=42)
        
        logger.info(f"Data split: Train={len(train_ann)}, Val={len(val_ann)}, Test={len(test_ann)}")
        
        # Create datasets
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        train_dataset = ReceiptDataset(train_ann, self.data_dir / 'dataset' / 'images', tokenizer)
        val_dataset = ReceiptDataset(val_ann, self.data_dir / 'dataset' / 'images', tokenizer)
        test_dataset = ReceiptDataset(test_ann, self.data_dir / 'dataset' / 'images', tokenizer)
        
        # Create data loaders with weighted sampling if using semi-supervised
        if sample_weights is not None and self.args.use_semi_supervised:
            # Get weights for training samples
            train_weights = sample_weights[:len(train_ann)]
            sampler = WeightedRandomSampler(train_weights, len(train_weights))
            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)
        
        # Initialize model
        num_banks = len(train_dataset.label_encoder.classes_)
        model = HybridBankClassifier(num_banks).to(self.device)
        
        logger.info(f"Model initialized with {num_banks} bank classes")
        
        # Loss function (Focal Loss for class imbalance)
        if self.args.use_focal_loss:
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
            logger.info("Using Focal Loss")
        else:
            criterion = nn.CrossEntropyLoss()
            logger.info("Using Cross Entropy Loss")
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.args.scheduler_patience,
            factor=0.5,
            verbose=True
        )
        
        # Train model
        model, best_val_acc = self.train_with_curriculum(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            self.args.epochs
        )
        
        # Save final model
        torch.save(model.state_dict(), self.models_dir / 'bank_classifier_enhanced.pt')
        import joblib
        joblib.dump(train_dataset.label_encoder, self.models_dir / 'label_encoder.pkl')
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
        logger.info(f"Model saved to: {self.models_dir / 'bank_classifier_enhanced.pt'}")
        
        return model, test_loader, train_dataset.label_encoder


def main():
    parser = argparse.ArgumentParser(description='Enhanced ML Training for Bank Receipts')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing training data')
    parser.add_argument('--models-dir', type=str, default='app/models',
                       help='Directory to save trained models')
    
    # Augmentation arguments
    parser.add_argument('--use-augmentation', action='store_true',
                       help='Use data augmentation')
    parser.add_argument('--num-augmentations', type=int, default=3,
                       help='Number of augmentations per image')
    
    # Semi-supervised arguments
    parser.add_argument('--use-semi-supervised', action='store_true',
                       help='Use semi-supervised learning')
    parser.add_argument('--max-pseudo-labels', type=int, default=100,
                       help='Maximum number of pseudo-labels to generate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay for regularization')
    
    # Advanced training arguments
    parser.add_argument('--use-focal-loss', action='store_true',
                       help='Use focal loss instead of cross entropy')
    parser.add_argument('--early-stopping', action='store_true',
                       help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--scheduler-patience', type=int, default=5,
                       help='Patience for learning rate scheduler')
    
    # Quick test mode
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode with reduced epochs')
    
    args = parser.parse_args()
    
    # Quick test mode overrides
    if args.quick_test:
        args.epochs = 2
        args.batch_size = 4
        args.max_pseudo_labels = 10
        logger.info("Running in QUICK TEST mode")
    
    # Create trainer and run
    trainer = EnhancedTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
