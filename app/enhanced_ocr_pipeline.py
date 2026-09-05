import logging
import re
from pathlib import Path
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available, PDF text extraction disabled")

from app.ocr_pipeline import OCRPipeline


def is_text_garbage(text: str) -> bool:
    """Detect if extracted PDF text is unusable (CID-encoded, mojibake, too short).

    CID font-subset PDFs (no toUnicode map) produce sequences like `(cid:0)(cid:1)`
    which pass length checks but contain zero semantic content. Treat them as
    "no embedded text" so the caller falls through to rasterize+OCR.
    """
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) < 15:
        return True
    if stripped.count("(cid:") >= 3:
        return True
    printable = sum(1 for c in stripped if c.isprintable() or c in "\n\r\t")
    if printable / max(len(stripped), 1) < 0.6:
        return True
    # A one-line receipt or cropped confirmation can be shorter than the old
    # arbitrary 50-character threshold and still contain everything needed.
    # Requiring both a receipt label and digits keeps random PDF furniture from
    # bypassing OCR while avoiding an expensive raster pass for valid short
    # text layers.
    if len(stripped) < 50:
        has_field = bool(re.search(
            r"\b(?:REF(?:ERENCE)?|TRANSACTION|PAYMENT|RECEIPT|ACCOUNT|"
            r"AMOUNT|TOTAL|DATE|BANK)\b", stripped, re.IGNORECASE))
        if not has_field or sum(ch.isdigit() for ch in stripped) < 3:
            return True
    return False


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

        Rotation and adaptive retries belong to ``OCRPipeline``. Keeping them
        here too previously multiplied a weak page into as many as twelve OCR
        runs. It also stopped after the first page with an ID, which lost valid
        references printed on later pages.
        """
        file_path = Path(file_path)
        
        # Try PDF text extraction for PDF files
        if file_path.suffix.lower() == '.pdf' and PDFPLUMBER_AVAILABLE:
            try:
                text = self._extract_pdf_text(file_path)
                if text and not is_text_garbage(text):
                    # Readable embedded text is authoritative whether or not a
                    # legacy ID heuristic fires. The label-driven extractor may
                    # legitimately find payer references or an intentionally
                    # absent bank reference later in the request.
                    logger.info("PDF text extraction successful: %s characters",
                                len(text))
                    return {
                        'text': text,
                        'tokens': [],
                        'confidence': 95.0,  # 0-100 scale
                        'method': 'pdf_text_extraction',
                        'processed_successfully': True,
                    }
                else:
                    logger.info("PDF text extraction yielded insufficient text, falling back to OCR")
            except Exception as e:
                logger.warning(f"PDF text extraction failed: {e}, falling back to OCR")

        # The core pipeline processes every page sequentially, caps image size,
        # checks orientation once, and performs adaptive retries.
        logger.info("Using OCR for text extraction")
        result = self.ocr_pipeline.process_file(str(file_path))
        result.setdefault('method', 'ocr')
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
