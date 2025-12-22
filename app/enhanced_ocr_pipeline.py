import logging
from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available, PDF text extraction disabled")

from app.ocr_pipeline import OCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3


class EnhancedOCRPipeline:
    """Enhanced OCR pipeline that tries PDF text extraction before OCR"""
    
    def __init__(self):
        self.ocr_pipeline = OCRPipeline()
    
    def extract_text_with_confidence(self, image: np.ndarray, skip_rotation: bool = False) -> Dict[str, Any]:
        """Delegate to internal OCR pipeline"""
        return self.ocr_pipeline.extract_text_with_confidence(image, skip_rotation=skip_rotation)
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a file with hybrid approach:
        1. Try PDF text extraction first (if PDF)
        2. Fall back to OCR if extraction fails or yields poor results
        3. If OCR yields no IDs, try rotating the image
        """
        file_path = Path(file_path)
        
        # Try PDF text extraction for PDF files
        if file_path.suffix.lower() == '.pdf' and PDFPLUMBER_AVAILABLE:
            try:
                text = self._extract_pdf_text(file_path)
                if text and len(text) > 100:  # Meaningful amount of text
                    # Check if we can extract IDs from this text
                    extraction = extract_all_fields_v3(text)
                    if extraction.get('all_ids'):
                        logger.info(f"PDF text extraction successful: {len(text)} characters, IDs found")
                        return {
                            'text': text,
                            'confidence': 0.95,
                            'method': 'pdf_text_extraction'
                        }
                    else:
                        logger.info("PDF text extracted but no IDs found, falling back to OCR")
                else:
                    logger.info("PDF text extraction yielded insufficient text, falling back to OCR")
            except Exception as e:
                logger.warning(f"PDF text extraction failed: {e}, falling back to OCR")
        
        # Fall back to OCR with rotation check
        logger.info("Using OCR for text extraction")
        return self._process_with_rotation_fallback(str(file_path))

    def _process_with_rotation_fallback(self, file_path: str) -> Dict[str, Any]:
        """Run OCR and try rotations if no IDs are found"""
        # Initial run
        result = self.ocr_pipeline.process_file(file_path)
        text = result['text']
        
        # Check if we found IDs
        extraction = extract_all_fields_v3(text)
        initial_ids = extraction.get('all_ids')
        
        # If IDs found AND confidence is good, return immediately
        # We check confidence because garbage text sometimes produces fake "IDs"
        if initial_ids and result['confidence'] >= 70:
            result['method'] = 'ocr'
            return result
            
        # If no IDs or low confidence, try rotating
        logger.info(f"No IDs found or low confidence ({result['confidence']:.2f}%). Attempting rotation fallback...")
        
        try:
            # Get images to rotate
            images = []
            if Path(file_path).suffix.lower() == '.pdf':
                images = self.ocr_pipeline.process_pdf_to_images(file_path)
            else:
                img = cv2.imread(file_path)
                if img is not None:
                    images = [img]
            
            if not images:
                return result
                
            # Try 90, 180, 270 degrees
            best_result = result
            
            for angle in [90, 180, 270]:
                rotated_text_parts = []
                total_conf = 0
                
                for img in images:
                    # Rotate image
                    if angle == 90:
                        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    elif angle == 180:
                        rotated = cv2.rotate(img, cv2.ROTATE_180)
                    elif angle == 270:
                        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    # Run OCR on rotated image
                    ocr_res = self.ocr_pipeline.extract_text_with_confidence(rotated, skip_rotation=True)
                    rotated_text_parts.append(ocr_res['text'])
                    total_conf += ocr_res['confidence']
                
                rotated_text = "\n".join(rotated_text_parts)
                avg_conf = total_conf / len(images) if images else 0
                
                # Check for IDs
                rot_extraction = extract_all_fields_v3(rotated_text)
                ids = rot_extraction.get('all_ids', [])
                
                if ids:
                    logger.info(f"Found IDs with {angle}° rotation: {ids}")
                    return {
                        'text': rotated_text,
                        'confidence': avg_conf,
                        'method': f'ocr_rotated_{angle}'
                    }
            
            logger.info("Rotation fallback yielded no new IDs")
            return result
            
        except Exception as e:
            logger.error(f"Rotation fallback error: {e}")
            return result
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text directly from PDF"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n=== PAGE {page_num + 1} ===\n{page_text}"
        return text.strip()
