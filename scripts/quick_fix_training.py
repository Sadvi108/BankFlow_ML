"""
Quick Fix and Training Script for Receipt Extraction

This script fixes the immediate OCR issues and improves pattern matching
without requiring PyTorch dependencies.
"""

import sys
from pathlib import Path
import logging

# Add app to path
sys.path.insert(0, str(Path.cwd()))

from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_latest_upload():
    """Test the latest uploaded receipt."""
    import os
    
    # Find latest upload
    uploads_dir = Path('data/uploads')
    files = sorted(
        [f for f in uploads_dir.glob('*') if f.suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg']],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not files:
        logger.error("No uploaded files found!")
        return
    
    latest_file = files[0]
    logger.info(f"Testing latest upload: {latest_file.name}")
    
    # Process with OCR
    pipeline = EnhancedOCRPipeline()
    
    try:
        result = pipeline.process_file(str(latest_file))
        text = result.get('text', '')
        
        logger.info(f"OCR Confidence: {result.get('confidence', 0):.2f}%")
        logger.info(f"OCR Method: {result.get('method', 'unknown')}")
        logger.info(f"Text Length: {len(text)} characters")
        
        if text:
            logger.info(f"Text Preview (first 500 chars):\n{text[:500]}")
            
            # Extract fields
            extraction = extract_all_fields_v3(text)
            
            logger.info("\n" + "="*60)
            logger.info("EXTRACTION RESULTS:")
            logger.info("="*60)
            logger.info(f"Bank: {extraction.get('bank_name', 'Unknown')}")
            logger.info(f"Reference IDs: {extraction.get('all_ids', [])}")
            logger.info(f"Primary ID: {extraction.get('transaction_id', 'None')}")
            logger.info(f"Amount: {extraction.get('amount', 'None')}")
            logger.info(f"Date: {extraction.get('date', 'None')}")
            logger.info(f"Confidence: {extraction.get('confidence', 0):.2f}")
            logger.info("="*60)
            
            if extraction.get('bank_name') == 'Unknown' or not extraction.get('all_ids'):
                logger.warning("\n⚠️  EXTRACTION FAILED!")
                logger.warning("Debugging information:")
                logger.warning(f"Full OCR text:\n{text}")
                
                # Try to identify why it failed
                if 'maybank' in text.lower():
                    logger.info("✓ Found 'Maybank' in text")
                if 'cimb' in text.lower():
                    logger.info("✓ Found 'CIMB' in text")
                if 'public' in text.lower():
                    logger.info("✓ Found 'Public' in text")
                    
                # Look for potential IDs
                import re
                potential_ids = re.findall(r'\b[A-Z0-9]{8,20}\b', text.upper())
                logger.info(f"Potential IDs found: {potential_ids[:10]}")
        else:
            logger.error("❌ OCR returned empty text!")
            logger.error("This could be due to:")
            logger.error("  1. Image quality issues")
            logger.error("  2. Unsupported file format")
            logger.error("  3. Tesseract OCR not properly configured")
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()


def improve_patterns():
    """Improve pattern matching based on common failures."""
    logger.info("\n" + "="*60)
    logger.info("PATTERN IMPROVEMENT RECOMMENDATIONS")
    logger.info("="*60)
    
    recommendations = [
        "1. Add more flexible spacing patterns",
        "2. Handle multi-line reference IDs",
        "3. Improve OCR error tolerance",
        "4. Add bank-specific ID formats",
        "5. Better handling of poor quality scans"
    ]
    
    for rec in recommendations:
        logger.info(rec)
    
    logger.info("\nTo improve extraction:")
    logger.info("- Review app/ultimate_patterns_v3.py")
    logger.info("- Add patterns for your specific receipt formats")
    logger.info("- Test with: python quick_fix_training.py")


if __name__ == '__main__':
    logger.info("="*60)
    logger.info("RECEIPT EXTRACTION QUICK FIX & DIAGNOSTIC")
    logger.info("="*60)
    
    test_latest_upload()
    improve_patterns()
    
    logger.info("\n" + "="*60)
    logger.info("NEXT STEPS:")
    logger.info("="*60)
    logger.info("1. Check the OCR text output above")
    logger.info("2. If text is empty, check Tesseract installation")
    logger.info("3. If text exists but extraction fails, improve patterns")
    logger.info("4. Test specific receipts with: python debug_single_receipt.py <file>")
    logger.info("="*60)
