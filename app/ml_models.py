"""
Advanced Machine Learning Models for Bank Receipt Processing

This module implements production-ready ML models for:
1. Bank Classification (Multi-class CNN + Text Hybrid)
2. Transaction ID Extraction (NER + OCR Post-processing)
3. Receipt Template Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import re
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    LayoutLMTokenizer, LayoutLMForTokenClassification, LayoutLMConfig
)
import cv2
from PIL import Image
import pytesseract
from sklearn.preprocessing import LabelEncoder
import joblib
import logging

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ReceiptFeatures:
    """Features extracted from a receipt for ML processing."""
    image_features: np.ndarray
    text_features: np.ndarray
    layout_features: np.ndarray
    ocr_tokens: List[Dict[str, Any]]
    bank_name: Optional[str] = None
    transaction_id: Optional[str] = None


class ReceiptDataset(Dataset):
    """Dataset for training receipt processing models."""
    
    def __init__(self, annotations: List[Dict], image_dir: Path, 
                 tokenizer: AutoTokenizer, max_length: int = 512):
        self.annotations = annotations
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        
        # Fit label encoder for bank names
        bank_names = [ann.get('ground_truth', {}).get('bank_name') or 
                     ann.get('bank', {}).get('name', 'unknown') 
                     for ann in annotations]
        self.label_encoder.fit(bank_names)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load and process image
        image_path = None
        if 'dataset_image_path' in ann.get('meta', {}):
            image_path = Path(ann['meta']['dataset_image_path'])
        elif 'processed_path' in ann.get('meta', {}):
            image_path = Path(ann['meta']['processed_path'])
        
        if image_path and image_path.exists():
            image = Image.open(image_path).convert('RGB')
            image_features = self._extract_image_features(image)
        else:
            image_features = np.zeros((3, 224, 224))  # Default empty features
        
        # Process text
        text = ann.get('ocr_text', '')
        text_encoding = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract layout features from OCR tokens
        tokens = ann.get('ocr_tokens', [])
        layout_features = self._extract_layout_features(tokens)
        
        # Get labels
        bank_label = ann.get('ground_truth', {}).get('bank_name') or \
                      ann.get('bank', {}).get('name', 'unknown')
        bank_label_encoded = self.label_encoder.transform([bank_label])[0]
        
        transaction_id = ann.get('ground_truth', {}).get('transaction_id') or \
                          ann.get('fields', {}).get('transaction_number', '')
        
        return {
            'image_features': torch.FloatTensor(image_features),
            'input_ids': text_encoding['input_ids'].squeeze(),
            'attention_mask': text_encoding['attention_mask'].squeeze(),
            'layout_features': torch.FloatTensor(layout_features),
            'bank_label': torch.LongTensor([bank_label_encoded]),
            'transaction_id': transaction_id,
            'text': text,
            'tokens': tokens
        }
    
    def _extract_image_features(self, image: Image.Image) -> np.ndarray:
        """Extract CNN features from receipt image."""
        # Resize to standard size
        image = image.resize((224, 224))
        # Convert to numpy and normalize
        img_array = np.array(image) / 255.0
        # Convert to CHW format for PyTorch
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array.astype(np.float32)
    
    def _extract_layout_features(self, tokens: List[Dict]) -> np.ndarray:
        """Extract layout features from OCR token positions."""
        if not tokens:
            return np.zeros(100)  # Default empty features
        
        # Extract bounding box statistics
        lefts = [t.get('left', 0) for t in tokens]
        tops = [t.get('top', 0) for t in tokens]
        widths = [t.get('width', 0) for t in tokens]
        heights = [t.get('height', 0) for t in tokens]
        
        # Create feature vector with statistics
        features = [
            np.mean(lefts), np.std(lefts), np.min(lefts), np.max(lefts),
            np.mean(tops), np.std(tops), np.min(tops), np.max(tops),
            np.mean(widths), np.std(widths), np.min(widths), np.max(widths),
            np.mean(heights), np.std(heights), np.min(heights), np.max(heights),
            len(tokens)
        ]
        
        # Pad or truncate to fixed size
        target_size = 100
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)


class HybridBankClassifier(nn.Module):
    """Hybrid CNN + Text classifier for bank receipt classification."""
    
    def __init__(self, num_banks: int, text_model_name: str = 'bert-base-uncased'):
        super().__init__()
        self.num_banks = num_banks
        
        # Image processing branch (CNN)
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.image_fc = nn.Linear(128 * 7 * 7, 256)
        
        # Text processing branch (Transformer)
        self.text_config = AutoConfig.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.text_fc = nn.Linear(self.text_config.hidden_size, 256)
        
        # Layout features branch
        self.layout_fc = nn.Linear(100, 64)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_banks)
        )
        
    def forward(self, image_features, input_ids, attention_mask, layout_features):
        # Image branch
        img_out = self.image_conv(image_features)
        img_out = img_out.view(img_out.size(0), -1)
        img_out = F.relu(self.image_fc(img_out))
        
        # Text branch
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_pooled = text_out.last_hidden_state[:, 0, :]  # CLS token
        text_out = F.relu(self.text_fc(text_pooled))
        
        # Layout branch
        layout_out = F.relu(self.layout_fc(layout_features))
        
        # Fusion
        combined = torch.cat([img_out, text_out, layout_out], dim=1)
        output = self.fusion(combined)
        
        return output


class TransactionIDExtractor(nn.Module):
    """NER-based model for extracting transaction IDs from receipts."""
    
    def __init__(self, num_labels: int = 3, model_name: str = 'bert-base-uncased'):
        super().__init__()
        self.num_labels = num_labels  # B-I-O tagging: Begin, Inside, Outside
        
        # Initialize LayoutLM model without remote downloads
        config = LayoutLMConfig(num_labels=num_labels)
        self.model = LayoutLMForTokenClassification(config)
        
        # Additional layers for transaction ID specific features
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels)
        )
        
    def forward(self, input_ids, attention_mask, bbox=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            labels=labels
        )
        return outputs


class ReceiptMLPipeline:
    """Main pipeline for receipt processing with ML models."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models (will be loaded when needed)
        self.bank_classifier = None
        self.transaction_extractor = None
        self.tokenizer = None
        self.label_encoder = None
        
        # Transaction ID patterns for post-processing
        self.transaction_patterns = [
            r'\b(?:ref|reference|ref\.|refno|refno\.|payment ref|trans ref|bank ref)\s*[:#-]?\s*([A-Za-z0-9._\-/#]{5,})',
            r'\b(?:transaction|txn|trans|tx)\s*(?:id|no|number|ref)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{5,})',
            r'\b(?:invoice|inv|bill)\s*(?:ref|no|number)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{4,})',
            r'\bduitnow\s*(?:ref|reference|id)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{6,})',
            r'\b(?:customer|bank|cust)\s*ref\s*[:#-]?\s*([A-Za-z0-9._\-/#]{5,})',
            # Payment patterns
            r'\b(?:payment|pymt|pay)\s*(?:ref|reference|id)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{5,})',
            # Transfer patterns
            r'\b(?:transfer|xfer|trf)\s*(?:ref|id|no)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{5,})',
            # Receipt patterns
            r'\b(?:receipt|rcpt|rcp)\s*(?:no|number|id)?\s*[:#-]?\s*([A-Za-z0-9._\-/#]{4,})',
            # General ID patterns
            r'\b(?:id|no|number|code)\s*[:#-]?\s*([A-Za-z0-9._\-/#]{6,})',
            # Bank-specific patterns
            r'\b(?:maybank|cimb|rhb|public bank|hong leong|ambank|uob|ocbc|standard chartered|hsbc)\s*[:#-]?\s*([A-Za-z0-9._\-/#]{6,})',
        ]
        
    def load_models(self):
        """Load trained models from disk."""
        classifier_loaded = False
        extractor_loaded = False

        # Load bank classifier
        try:
            classifier_path = self.models_dir / 'bank_classifier.pt'
            encoder_path = self.models_dir / 'label_encoder.pkl'

            if classifier_path.exists() and encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                num_banks = len(self.label_encoder.classes_)

                self.bank_classifier = HybridBankClassifier(num_banks)
                self.bank_classifier.load_state_dict(
                    torch.load(classifier_path, map_location=self.device)
                )
                self.bank_classifier.to(self.device)
                self.bank_classifier.eval()
                classifier_loaded = True
                logger.info("Bank classifier loaded")
        except Exception as e:
            logger.error(f"Failed to load bank classifier: {e}")

        # Load transaction extractor (prefer enhanced model if available)
        try:
            enhanced_path = self.models_dir / 'transaction_extractor_enhanced.pt'
            extractor_path = self.models_dir / 'transaction_extractor.pt'
            load_path = enhanced_path if enhanced_path.exists() else extractor_path

            if load_path.exists():
                loaded_obj = torch.load(load_path, map_location=self.device)

                if isinstance(loaded_obj, dict) and 'model_state_dict' in loaded_obj:
                    # Instantiate LayoutLM token classifier directly
                    lm_config = LayoutLMConfig(num_labels=3)
                    self.transaction_extractor = LayoutLMForTokenClassification(lm_config)
                    self.transaction_extractor.load_state_dict(loaded_obj['model_state_dict'])
                    # Override tokenizer name if provided
                    tok_name = loaded_obj.get('tokenizer_name', 'microsoft/layoutlm-base-uncased')
                    try:
                        self.tokenizer = LayoutLMTokenizer.from_pretrained(tok_name, use_fast=False)
                    except Exception:
                        self.tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased', use_fast=False)
                elif hasattr(loaded_obj, 'state_dict'):
                    # Full model object saved
                    self.transaction_extractor = loaded_obj
                else:
                    # Raw state dict
                    lm_config = LayoutLMConfig(num_labels=3)
                    self.transaction_extractor = LayoutLMForTokenClassification(lm_config)
                    self.transaction_extractor.load_state_dict(loaded_obj)

                self.transaction_extractor.to(self.device)
                self.transaction_extractor.eval()
                extractor_loaded = True
                logger.info(f"Transaction extractor loaded: {load_path}")
        except Exception as e:
            logger.error(f"Failed to load transaction extractor: {e}")

        # Initialize tokenizer (LayoutLM has no fast tokenizer)
        try:
            self.tokenizer = LayoutLMTokenizer.from_pretrained(
                'microsoft/layoutlm-base-uncased', use_fast=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None

        return classifier_loaded or extractor_loaded
    
    def predict_bank(self, receipt_features: ReceiptFeatures) -> Tuple[str, float]:
        """Predict bank name from receipt features."""
        if not self.bank_classifier or not self.tokenizer:
            return "unknown", 0.0
        
        try:
            # Prepare inputs
            image_tensor = torch.FloatTensor(receipt_features.image_features).unsqueeze(0).to(self.device)
            
            text_encoding = self.tokenizer(
                receipt_features.text_features,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = text_encoding['input_ids'].to(self.device)
            attention_mask = text_encoding['attention_mask'].to(self.device)
            
            layout_tensor = torch.FloatTensor(receipt_features.layout_features).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.bank_classifier(image_tensor, input_ids, attention_mask, layout_tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)
                
            # Decode prediction
            predicted_bank = self.label_encoder.inverse_transform([predicted_class.item()])[0]
            confidence_score = confidence.item()
            
            return predicted_bank, confidence_score
            
        except Exception as e:
            print(f"Error in bank classification: {e}")
            return "unknown", 0.0
    
    def extract_transaction_id(self, receipt_features: ReceiptFeatures) -> Dict[str, Any]:
        """Extract transaction ID using NER and pattern matching."""
        if not self.transaction_extractor or not self.tokenizer:
            return self._fallback_extraction(receipt_features)
        
        try:
            # Prepare inputs for NER model
            text = receipt_features.text_features
            tokens = receipt_features.ocr_tokens
            
            # Tokenize text
            encoding = self.tokenizer(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Prepare bounding boxes for LayoutLM
            bbox = self._prepare_bboxes(tokens, encoding)
            
            # Predict with NER model
            with torch.no_grad():
                outputs = self.transaction_extractor(
                    input_ids=encoding['input_ids'].to(self.device),
                    attention_mask=encoding['attention_mask'].to(self.device),
                    bbox=bbox.to(self.device) if bbox is not None else None
                )
                
                predictions = torch.argmax(outputs.logits, dim=2)
            
            # Extract entities from predictions
            extracted_ids = self._extract_entities_from_predictions(
                text, encoding, predictions
            )
            
            # Combine with pattern matching
            pattern_matches = self._pattern_based_extraction(text)
            
            return {
                'transaction_ids': extracted_ids,
                'pattern_matches': pattern_matches,
                'best_match': self._select_best_match(extracted_ids, pattern_matches),
                'method': 'ner+patterns'
            }
            
        except Exception as e:
            print(f"Error in transaction extraction: {e}")
            return self._fallback_extraction(receipt_features)
    
    def _fallback_extraction(self, receipt_features: ReceiptFeatures) -> Dict[str, Any]:
        """Fallback extraction using only regex patterns."""
        text = receipt_features.text_features
        pattern_matches = self._pattern_based_extraction(text)
        
        return {
            'transaction_ids': [],
            'pattern_matches': pattern_matches,
            'best_match': pattern_matches[0] if pattern_matches else None,
            'method': 'patterns_only'
        }
    
    def _prepare_bboxes(self, tokens: List[Dict], encoding) -> Optional[torch.Tensor]:
        """Prepare bounding boxes for LayoutLM."""
        if not tokens:
            return None
        
        # Simple bbox preparation - in practice, you'd map tokens to wordpiece tokens
        # For now, return a default bbox
        seq_len = encoding['input_ids'].shape[1]
        default_bbox = torch.zeros((1, seq_len, 4), dtype=torch.long)
        default_bbox[:, :, 2] = 1000  # width
        default_bbox[:, :, 3] = 1000  # height
        
        return default_bbox
    
    def _extract_entities_from_predictions(self, text: str, encoding, predictions) -> List[str]:
        """Extract transaction ID entities from NER predictions."""
        # Convert predictions to entities (simplified)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        entities = []
        current_entity = []
        
        for token, pred in zip(tokens, predictions[0]):
            if pred == 1:  # Beginning of entity
                if current_entity:
                    entities.append(''.join(current_entity))
                current_entity = [token.replace('##', '')]
            elif pred == 2:  # Inside entity
                current_entity.append(token.replace('##', ''))
            else:  # Outside entity
                if current_entity:
                    entities.append(''.join(current_entity))
                    current_entity = []
        
        if current_entity:
            entities.append(''.join(current_entity))
        
        # Filter entities that look like transaction IDs
        valid_ids = []
        for entity in entities:
            if self._looks_like_transaction_id(entity):
                valid_ids.append(entity)
        
        return valid_ids
    
    def _pattern_based_extraction(self, text: str) -> List[str]:
        """Extract transaction IDs using regex patterns."""
        matches = []
        for pattern in self.transaction_patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            matches.extend(found)
        
        # Clean and validate matches
        cleaned_matches = []
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            cleaned = match.strip()
            if self._looks_like_transaction_id(cleaned):
                cleaned_matches.append(cleaned)
        
        return list(set(cleaned_matches))  # Remove duplicates
    
    def _looks_like_transaction_id(self, text: str) -> bool:
        """Check if text looks like a transaction ID."""
        if len(text) < 4 or len(text) > 50:
            return False
        
        # Must contain at least one digit
        if not any(c.isdigit() for c in text):
            return False
        
        # Must be alphanumeric (allow some special chars)
        if not re.match(r'^[A-Za-z0-9\-/]+$', text):
            return False
        
        # Must have reasonable character distribution
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        
        return letters + digits >= len(text) * 0.7  # At least 70% alphanumeric
    
    def _select_best_match(self, ner_matches: List[str], pattern_matches: List[str]) -> Optional[str]:
        """Select the best transaction ID match from multiple sources."""
        all_matches = ner_matches + pattern_matches
        
        if not all_matches:
            return None
        
        # Prefer longer matches that look more like transaction IDs
        scored_matches = []
        for match in all_matches:
            score = 0
            score += len(match) * 0.1  # Longer is better
            score += sum(c.isdigit() for c in match) * 0.2  # More digits is better
            score += sum(c.isupper() for c in match) * 0.1  # Uppercase letters are good
            
            # Penalize common false positives
            if match.lower() in {'date', 'ref', 'no', 'id', 'invoice'}:
                score -= 10
            
            scored_matches.append((match, score))
        
        # Return the highest scoring match
        best_match = max(scored_matches, key=lambda x: x[1])
        return best_match[0] if best_match[1] > 0 else None


def create_receipt_features(image: Image.Image, text: str, tokens: List[Dict]) -> ReceiptFeatures:
    """Create ReceiptFeatures from raw inputs."""
    # Extract image features
    image_array = np.array(image.resize((224, 224))) / 255.0
    image_features = np.transpose(image_array, (2, 0, 1)).astype(np.float32)
    
    # Layout features from tokens
    if tokens:
        lefts = [t.get('left', 0) for t in tokens]
        tops = [t.get('top', 0) for t in tokens]
        widths = [t.get('width', 0) for t in tokens]
        heights = [t.get('height', 0) for t in tokens]
        
        layout_features = [
            np.mean(lefts), np.std(lefts), np.min(lefts), np.max(lefts),
            np.mean(tops), np.std(tops), np.min(tops), np.max(tops),
            np.mean(widths), np.std(widths), np.min(widths), np.max(widths),
            np.mean(heights), np.std(heights), np.min(heights), np.max(heights),
            len(tokens)
        ]
        
        # Pad to fixed size
        target_size = 100
        if len(layout_features) < target_size:
            layout_features.extend([0.0] * (target_size - len(layout_features)))
        else:
            layout_features = layout_features[:target_size]
    else:
        layout_features = np.zeros(100)
    
    return ReceiptFeatures(
        image_features=image_features,
        text_features=text,
        layout_features=np.array(layout_features, dtype=np.float32),
        ocr_tokens=tokens
    )