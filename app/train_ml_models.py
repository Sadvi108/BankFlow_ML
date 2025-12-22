"""
Training Pipeline for Bank Receipt ML Models

This module handles training of:
1. Hybrid Bank Classifier (CNN + Text + Layout)
2. Transaction ID Extractor (NER-based)
3. Model evaluation and validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm
import time

from ml_models import (
    ReceiptDataset, HybridBankClassifier, TransactionIDExtractor,
    ReceiptMLPipeline, create_receipt_features
)
from dataset import read_annotations


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles training of receipt processing models."""
    
    def __init__(self, models_dir: Path, data_dir: Path):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(log_dir=models_dir / 'tensorboard')
        
        logger.info(f"Using device: {self.device}")
        
    def prepare_data(self, test_size: float = 0.2, val_size: float = 0.1) -> Dict:
        """Prepare training data with train/val/test splits."""
        # Load annotations
        annotations = read_annotations()
        
        if len(annotations) < 10:
            raise ValueError(f"Need at least 10 samples for training, got {len(annotations)}")
        
        logger.info(f"Loaded {len(annotations)} annotations")
        
        # Split data
        train_ann, temp_ann = train_test_split(annotations, test_size=test_size + val_size, random_state=42)
        val_ann, test_ann = train_test_split(temp_ann, test_size=test_size/(test_size + val_size), random_state=42)
        
        logger.info(f"Train: {len(train_ann)}, Val: {len(val_ann)}, Test: {len(test_ann)}")
        
        return {
            'train': train_ann,
            'val': val_ann,
            'test': test_ann
        }
    
    def train_bank_classifier(self, data_splits: Dict, epochs: int = 50, batch_size: int = 16) -> Dict:
        """Train the hybrid bank classifier."""
        logger.info("Training bank classifier...")
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Create datasets
        train_dataset = ReceiptDataset(data_splits['train'], self.data_dir / 'dataset' / 'images', tokenizer)
        val_dataset = ReceiptDataset(data_splits['val'], self.data_dir / 'dataset' / 'images', tokenizer)
        test_dataset = ReceiptDataset(data_splits['test'], self.data_dir / 'dataset' / 'images', tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        num_banks = len(train_dataset.label_encoder.classes_)
        model = HybridBankClassifier(num_banks).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
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
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': loss.item(), 'acc': 100. * train_correct / train_total})
            
            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
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
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Logging
            self.writer.add_scalar('BankClassifier/Train/Loss', avg_train_loss, epoch)
            self.writer.add_scalar('BankClassifier/Train/Acc', train_acc, epoch)
            self.writer.add_scalar('BankClassifier/Val/Loss', avg_val_loss, epoch)
            self.writer.add_scalar('BankClassifier/Val/Acc', val_acc, epoch)
            
            logger.info(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                       f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                logger.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
        
        # Save final model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Save model and label encoder
        torch.save(model.state_dict(), self.models_dir / 'bank_classifier.pt')
        joblib.dump(train_dataset.label_encoder, self.models_dir / 'label_encoder.pkl')
        
        # Test evaluation
        test_results = self.evaluate_bank_classifier(model, test_loader, train_dataset.label_encoder)
        
        logger.info("Bank classifier training completed!")
        return {
            'best_val_acc': best_val_acc,
            'test_results': test_results,
            'model': model,
            'label_encoder': train_dataset.label_encoder
        }
    
    def evaluate_bank_classifier(self, model: HybridBankClassifier, test_loader: DataLoader, 
                                label_encoder, save_predictions: bool = True) -> Dict:
        """Evaluate bank classifier on test set."""
        logger.info("Evaluating bank classifier on test set...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in test_loader:
                image_features = batch['image_features'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                layout_features = batch['layout_features'].to(self.device)
                labels = batch['bank_label'].squeeze().to(self.device)
                
                outputs = model(image_features, input_ids, attention_mask, layout_features)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # Convert to bank names
        pred_names = label_encoder.inverse_transform(all_predictions)
        true_names = label_encoder.inverse_transform(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(true_names, pred_names)
        precision, recall, f1, support = precision_recall_fscore_support(true_names, pred_names, average='weighted')
        
        # Detailed classification report
        detailed_report = classification_report(true_names, pred_names, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'detailed_report': detailed_report,
            'predictions': pred_names.tolist(),
            'true_labels': true_names.tolist(),
            'confidences': all_confidences
        }
        
        # Save results
        if save_predictions:
            with open(self.models_dir / 'bank_classifier_test_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return results
    
    def train_transaction_extractor(self, data_splits: Dict, epochs: int = 30, batch_size: int = 8) -> Dict:
        """Train the transaction ID extractor."""
        logger.info("Training transaction ID extractor...")
        
        # For now, we'll use pattern-based extraction enhanced with ML confidence
        # Full NER training would require annotated entity labels
        
        # Create training data for pattern validation
        train_texts = []
        train_labels = []
        
        for ann in data_splits['train']:
            text = ann.get('ocr_text', '')
            transaction_id = ann.get('ground_truth', {}).get('transaction_id') or \
                              ann.get('fields', {}).get('transaction_number', '')
            
            if text and transaction_id:
                train_texts.append(text)
                train_labels.append(transaction_id)
        
        # Validate pattern extraction on test set
        test_results = self.evaluate_transaction_extraction(data_splits['test'])
        
        logger.info("Transaction extractor training completed!")
        return {
            'test_results': test_results,
            'training_samples': len(train_texts)
        }
    
    def evaluate_transaction_extraction(self, test_data: List[Dict]) -> Dict:
        """Evaluate transaction ID extraction on test set."""
        logger.info("Evaluating transaction ID extraction...")
        
        all_predictions = []
        all_ground_truth = []
        
        # Create ML pipeline for evaluation
        pipeline = ReceiptMLPipeline(self.models_dir)
        if not pipeline.load_models():
            logger.warning("Could not load ML models, using pattern-based extraction only")
        
        for ann in test_data:
            text = ann.get('ocr_text', '')
            tokens = ann.get('ocr_tokens', [])
            true_transaction_id = ann.get('ground_truth', {}).get('transaction_id') or \
                                   ann.get('fields', {}).get('transaction_number', '')
            
            if not text or not true_transaction_id:
                continue
            
            # Create features
            # For evaluation, we need the original image - use a dummy for now
            from PIL import Image
            dummy_image = Image.new('RGB', (100, 100), color='white')
            features = create_receipt_features(dummy_image, text, tokens)
            
            # Extract transaction ID
            extraction_result = pipeline.extract_transaction_id(features)
            predicted_id = extraction_result.get('best_match', '')
            
            all_predictions.append(predicted_id)
            all_ground_truth.append(true_transaction_id)
        
        # Calculate metrics
        exact_matches = sum(1 for pred, true in zip(all_predictions, all_ground_truth) 
                           if pred == true)
        total_valid = len([p for p in all_predictions if p is not None])
        
        accuracy = exact_matches / len(all_ground_truth) if all_ground_truth else 0
        extraction_rate = total_valid / len(all_predictions) if all_predictions else 0
        
        results = {
            'total_samples': len(all_ground_truth),
            'exact_matches': exact_matches,
            'accuracy': accuracy,
            'extraction_rate': extraction_rate,
            'predictions': all_predictions,
            'ground_truth': all_ground_truth
        }
        
        # Save results
        with open(self.models_dir / 'transaction_extractor_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Transaction Extraction Results - Accuracy: {accuracy:.4f}, "
                   f"Extraction Rate: {extraction_rate:.4f}")
        
        return results
    
    def train_all_models(self, epochs_bank: int = 50, epochs_transaction: int = 30) -> Dict:
        """Train all models in sequence."""
        logger.info("Starting complete model training pipeline...")
        
        # Prepare data
        data_splits = self.prepare_data()
        
        # Train bank classifier
        bank_results = self.train_bank_classifier(data_splits, epochs=epochs_bank)
        
        # Train transaction extractor
        transaction_results = self.train_transaction_extractor(data_splits, epochs=epochs_transaction)
        
        # Create training summary
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_stats': {
                'train_samples': len(data_splits['train']),
                'val_samples': len(data_splits['val']),
                'test_samples': len(data_splits['test'])
            },
            'bank_classifier': bank_results,
            'transaction_extractor': transaction_results
        }
        
        # Save training summary
        with open(self.models_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("All models trained successfully!")
        return summary


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train receipt processing ML models')
    parser.add_argument('--models-dir', type=str, default='app/models',
                       help='Directory to save trained models')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing training data')
    parser.add_argument('--epochs-bank', type=int, default=50,
                       help='Number of epochs for bank classifier')
    parser.add_argument('--epochs-transaction', type=int, default=30,
                       help='Number of epochs for transaction extractor')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ModelTrainer(Path(args.models_dir), Path(args.data_dir))
    
    # Train all models
    results = trainer.train_all_models(
        epochs_bank=args.epochs_bank,
        epochs_transaction=args.epochs_transaction
    )
    
    print("Training completed!")
    print(f"Bank classifier accuracy: {results['bank_classifier']['test_results']['accuracy']:.4f}")
    print(f"Transaction extraction accuracy: {results['transaction_extractor']['test_results']['accuracy']:.4f}")


if __name__ == "__main__":
    main()