"""
Comprehensive Test Suite for Bank Receipt ML System

Tests cover:
1. OCR functionality
2. Bank classification (ML and keyword-based)
3. Transaction ID extraction
4. API endpoints
5. Model training pipeline
"""

import pytest
import json
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from unittest.mock import Mock, patch
import torch

from app import utils, ocr, classify, extract, dataset
from app.ml_models import (
    ReceiptFeatures, HybridBankClassifier, TransactionIDExtractor,
    ReceiptMLPipeline, create_receipt_features
)
from app.train_ml_models import ModelTrainer


class TestOCR:
    """Test OCR functionality."""
    
    def test_pdf_to_image(self):
        """Test PDF to image conversion."""
        # Create a simple test PDF (would need actual PDF file)
        # For now, test with a dummy image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img = Image.new('RGB', (100, 100), color='white')
            img.save(tmp.name)
            
            # Test image loading
            loaded_img = utils.load_image(Path(tmp.name))
            assert loaded_img.size == (100, 100)
            
            Path(tmp.name).unlink()
    
    def test_image_preprocessing(self):
        """Test image preprocessing pipeline."""
        # Create test image
        img = Image.new('RGB', (200, 100), color='white')
        
        # Test preprocessing
        processed_img = utils.preprocess_image(img)
        assert processed_img.size[0] > 0 and processed_img.size[1] > 0
    
    @pytest.mark.skipif(not ocr.TESSERACT_AVAILABLE, reason="Tesseract not available")
    def test_ocr_extraction(self):
        """Test OCR text extraction."""
        # Create image with text
        img = Image.new('RGB', (300, 100), color='white')
        
        # Test OCR (will return empty for blank image)
        text, tokens = ocr.image_to_text_and_tokens(img)
        assert isinstance(text, str)
        assert isinstance(tokens, list)


class TestBankClassification:
    """Test bank classification functionality."""
    
    def test_keyword_classification(self):
        """Test keyword-based bank classification."""
        # Test Maybank
        text = "This is a Maybank receipt with transaction details"
        bank, confidence = classify.bank_from_text(text)
        assert bank == "Maybank"
        assert confidence > 0
        
        # Test CIMB
        text = "CIMB Bank Berhad transaction confirmation"
        bank, confidence = classify.bank_from_text(text)
        assert bank == "CIMB"
        assert confidence > 0
    
    def test_unknown_bank(self):
        """Test classification with unknown bank."""
        text = "This is just some random text without bank names"
        bank, confidence = classify.bank_from_text(text)
        assert bank == "unknown"
        assert confidence == 0.0


class TestTransactionExtraction:
    """Test transaction ID extraction."""
    
    def test_pattern_extraction(self):
        """Test regex pattern-based extraction."""
        text = "Reference: ABC123456789 Transaction ID: TXN987654"
        fields = extract.extract_fields(text, [], None)
        
        assert fields["reference_number"] == "ABC123456789"
        assert fields["transaction_id"] == "TXN987654"
    
    def test_duitnow_extraction(self):
        """Test DuitNow reference extraction."""
        text = "DuitNow Reference Number: DUIT123456789"
        fields = extract.extract_fields(text, [], None)
        
        assert fields["duitnow_reference_number"] == "DUIT123456789"
    
    def test_amount_extraction(self):
        """Test amount extraction."""
        text = "Total amount: RM 1,234.56"
        fields = extract.extract_fields(text, [], None)
        
        assert fields["amount"] == "1,234.56"
    
    def test_date_extraction(self):
        """Test date extraction."""
        text = "Transaction date: 2024-01-15"
        fields = extract.fields(text, [], None)
        
        assert fields["date"] == "2024-01-15"


class TestMLModels:
    """Test ML model functionality."""
    
    def test_receipt_features_creation(self):
        """Test receipt features creation."""
        # Create test data
        img = Image.new('RGB', (224, 224), color='white')
        text = "Test receipt from Maybank Reference: ABC123"
        tokens = [
            {"text": "Test", "left": 10, "top": 10, "width": 50, "height": 20},
            {"text": "Maybank", "left": 60, "top": 10, "width": 80, "height": 20}
        ]
        
        features = create_receipt_features(img, text, tokens)
        
        assert isinstance(features.image_features, np.ndarray)
        assert features.image_features.shape == (3, 224, 224)
        assert features.text_features == text
        assert isinstance(features.layout_features, np.ndarray)
        assert len(features.ocr_tokens) == 2
    
    def test_hybrid_classifier_architecture(self):
        """Test hybrid classifier model architecture."""
        num_banks = 10
        model = HybridBankClassifier(num_banks)
        
        # Test forward pass with dummy data
        batch_size = 2
        image_features = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (batch_size, 512))
        attention_mask = torch.ones(batch_size, 512)
        layout_features = torch.randn(batch_size, 100)
        
        with torch.no_grad():
            outputs = model(image_features, input_ids, attention_mask, layout_features)
        
        assert outputs.shape == (batch_size, num_banks)
    
    def test_transaction_extractor_architecture(self):
        """Test transaction extractor model architecture."""
        model = TransactionIDExtractor(num_labels=3)
        
        # Test forward pass with dummy data
        batch_size = 2
        input_ids = torch.randint(0, 1000, (batch_size, 512))
        attention_mask = torch.ones(batch_size, 512)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        
        assert outputs.logits.shape == (batch_size, 512, 3)
    
    @patch('app.ml_models.AutoTokenizer')
    def test_ml_pipeline_initialization(self, mock_tokenizer):
        """Test ML pipeline initialization."""
        models_dir = Path(tempfile.mkdtemp())
        pipeline = ReceiptMLPipeline(models_dir)
        
        assert pipeline.models_dir == models_dir
        assert pipeline.device in [torch.device('cpu'), torch.device('cuda')]
        assert pipeline.bank_classifier is None
        assert pipeline.transaction_extractor is None


class TestDatasetManagement:
    """Test dataset management functionality."""
    
    def test_annotation_operations(self):
        """Test annotation CRUD operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock dataset directory
            original_data_dir = dataset.DATA_DIR
            dataset.DATA_DIR = Path(tmpdir)
            dataset.ANNOTATIONS_PATH = dataset.DATA_DIR / "dataset" / "annotations.jsonl"
            dataset.ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # Test adding annotation
                test_entry = {
                    "id": "test123",
                    "filename": "test.pdf",
                    "bank": {"name": "Maybank", "confidence": 0.95},
                    "fields": {"transaction_number": "ABC123"},
                    "ocr_text": "Test receipt",
                    "meta": {},
                    "ground_truth": {}
                }
                
                dataset.append_annotation(test_entry)
                
                # Test reading annotations
                annotations = dataset.read_annotations()
                assert len(annotations) == 1
                assert annotations[0]["id"] == "test123"
                
                # Test updating annotation
                updates = {"ground_truth": {"bank_name": "Maybank"}}
                success = dataset.update_annotation("test123", updates)
                assert success
                
                # Verify update
                annotations = dataset.read_annotations()
                assert annotations[0]["ground_truth"]["bank_name"] == "Maybank"
                
                # Test summary
                summary = dataset.summary()
                assert summary["total"] == 1
                assert "Maybank" in summary["per_bank"]
                
            finally:
                # Restore original paths
                dataset.DATA_DIR = original_data_dir
                dataset.ANNOTATIONS_PATH = original_data_dir / "dataset" / "annotations.jsonl"


class TestTrainingPipeline:
    """Test model training pipeline."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            data_dir = Path(tmpdir) / "data"
            
            trainer = ModelTrainer(models_dir, data_dir)
            
            assert trainer.models_dir == models_dir
            assert trainer.data_dir == data_dir
            assert trainer.device in [torch.device('cpu'), torch.device('cuda')]
    
    def test_data_preparation(self):
        """Test data preparation for training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock annotations
            annotations = [
                {
                    "id": f"test{i}",
                    "filename": f"receipt{i}.pdf",
                    "bank": {"name": "Maybank", "confidence": 0.9},
                    "fields": {"transaction_number": f"TXN{i:03d}"},
                    "ocr_text": f"Test receipt {i} from Maybank",
                    "meta": {},
                    "ground_truth": {"bank_name": "Maybank", "transaction_id": f"TXN{i:03d}"}
                }
                for i in range(20)
            ]
            
            # Mock dataset
            original_read_annotations = dataset.read_annotations
            dataset.read_annotations = lambda: annotations
            
            try:
                models_dir = Path(tmpdir) / "models"
                data_dir = Path(tmpdir) / "data"
                trainer = ModelTrainer(models_dir, data_dir)
                
                data_splits = trainer.prepare_data()
                
                assert "train" in data_splits
                assert "val" in data_splits
                assert "test" in data_splits
                assert len(data_splits["train"]) > 0
                assert len(data_splits["val"]) > 0
                assert len(data_splits["test"]) > 0
                
            finally:
                # Restore original function
                dataset.read_annotations = original_read_annotations


class TestTransactionValidation:
    """Test transaction ID validation logic."""
    
    def test_transaction_id_validation(self):
        """Test transaction ID validation rules."""
        from app.main_enhanced import validate_transaction_id
        
        # Valid transaction IDs
        assert validate_transaction_id("ABC123456", "Maybank") > 0.7
        assert validate_transaction_id("TXN987654", "CIMB") > 0.6
        assert validate_transaction_id("REF20240115", "Public Bank") > 0.6
        
        # Invalid transaction IDs
        assert validate_transaction_id("", "Maybank") == 0.0
        assert validate_transaction_id("ABC", "Maybank") < 0.5  # Too short
        assert validate_transaction_id("NOID", "Maybank") < 0.5  # No digits
    
    def test_transaction_looks_like_id(self):
        """Test transaction ID pattern matching."""
        from app.ml_models import ReceiptMLPipeline
        
        pipeline = ReceiptMLPipeline(Path("."))
        
        # Valid patterns
        assert pipeline._looks_like_transaction_id("ABC123456")
        assert pipeline._looks_like_transaction_id("TXN-2024-001")
        assert pipeline._looks_like_transaction_id("REF/2024/123")
        
        # Invalid patterns
        assert not pipeline._looks_like_transaction_id("")  # Empty
        assert not pipeline._looks_like_transaction_id("ABC")  # Too short
        assert not pipeline._looks_like_transaction_id("NOID")  # No digits
        assert not pipeline._looks_like_transaction_id("123456789012345678901234567890123456789012345678901")  # Too long


class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        from app.main_enhanced import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "ml_models_loaded" in data
        assert "ocr_available" in data
    
    def test_models_status_endpoint(self):
        """Test models status endpoint."""
        from app.main_enhanced import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        response = client.get("/models/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "models_loaded" in data
        assert "available_models" in data
    
    def test_dataset_summary_endpoint(self):
        """Test dataset summary endpoint."""
        from app.main_enhanced import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        response = client.get("/dataset/summary")
        
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "per_bank" in data


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.mark.integration
    def test_complete_receipt_processing(self):
        """Test complete receipt processing pipeline."""
        from app.main_enhanced import process_receipt
        
        # Create a test receipt image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img = Image.new('RGB', (800, 600), color='white')
            img.save(tmp.name)
            
            try:
                # Process the receipt
                result = process_receipt(Path(tmp.name), "test_receipt.png", "test123")
                
                # Verify result structure
                assert "bank" in result
                assert "transaction_id" in result
                assert "fields" in result
                assert "meta" in result
                
                assert "name" in result["bank"]
                assert "confidence" in result["bank"]
                assert "processing_time" in result["meta"]
                
            finally:
                Path(tmp.name).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])