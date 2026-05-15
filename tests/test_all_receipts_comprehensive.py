#!/usr/bin/env python3
"""
Comprehensive test script to validate all 23 receipts
Tests each receipt individually to achieve 100% accuracy
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import logging

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Initialize OCR pipeline
ocr_pipeline = EnhancedOCRPipeline()

# Directory with receipts
RECEIPTS_DIR = Path(__file__).parent / "Receipts"

def test_single_receipt(receipt_path: Path) -> dict:
    """Test a single receipt and return detailed results"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing: {receipt_path.name}")
    logger.info(f"{'='*80}")
    
    result = {
        'filename': receipt_path.name,
        'success': False,
        'bank_detected': None,
        'transaction_ids': [],
        'reference_ids': [],
        'all_ids': [],
        'amount': None,
        'date': None,
        'confidence': 0.0,
        'ocr_method': None,
        'error': None,
        'extracted_text_preview': None
    }
    
    try:
        # Process the file
        ocr_result = ocr_pipeline.process_file(str(receipt_path))
        text = ocr_result.get('text', '')
        result['ocr_method'] = ocr_result.get('method', 'unknown')
        result['confidence'] = ocr_result.get('confidence', 0.0)
        
        # Store text preview (first 200 chars)
        result['extracted_text_preview'] = text[:200] if text else None
        
        # Extract fields
        extraction = extract_all_fields_v3(text)
        
        result['bank_detected'] = extraction.get('bank_name')
        result['transaction_ids'] = extraction.get('transaction_ids', [])
        result['reference_ids'] = extraction.get('reference_ids', [])
        result['all_ids'] = extraction.get('all_ids', [])
        result['amount'] = extraction.get('amount')
        result['date'] = extraction.get('date')
        
        # Determine success - REQUIRE ALL FIELDS
        has_ids = result['all_ids'] and len(result['all_ids']) > 0
        has_date = result['date'] is not None
        has_amount = result['amount'] is not None
        
        if has_ids and has_date and has_amount:
            result['success'] = True
            logger.info(f"✅ SUCCESS!")
            logger.info(f"Bank: {result['bank_detected']}")
            logger.info(f"All IDs: {result['all_ids']}")
            logger.info(f"Date: {result['date']}")
            logger.info(f"Amount: {result['amount']}")
            logger.info(f"Confidence: {result['confidence']:.2%}")
            logger.info(f"Method: {result['ocr_method']}")
        else:
            result['success'] = False
            missing = []
            if not has_ids: missing.append("IDs")
            if not has_date: missing.append("Date")
            if not has_amount: missing.append("Amount")
            
            logger.error(f"❌ FAILED - Missing: {', '.join(missing)}")
            logger.error(f"Bank detected: {result['bank_detected']}")
            logger.error(f"Text preview: {result['extracted_text_preview']}")
            
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"❌ ERROR: {e}")
    
    return result

def main():
    """Test all receipts in the Receipts directory"""
    
    # Find all receipt files
    receipt_files = sorted(RECEIPTS_DIR.glob("*.*"))
    receipt_files = [f for f in receipt_files if f.suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png']]
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE RECEIPT TESTING")
    print(f"{'='*80}")
    print(f"Total receipts found: {len(receipt_files)}")
    print(f"Directory: {RECEIPTS_DIR}")
    print(f"{'='*80}\n")
    
    results = []
    success_count = 0
    failure_count = 0
    
    # Test each receipt
    for receipt_path in receipt_files:
        result = test_single_receipt(receipt_path)
        results.append(result)
        
        if result['success']:
            success_count += 1
        else:
            failure_count += 1
    
    # Summary
    print(f"\n\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total Receipts: {len(receipt_files)}")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failures: {failure_count}")
    print(f"Accuracy: {(success_count/len(receipt_files)*100):.2f}%")
    print(f"{'='*80}\n")
    
    # List failures
    if failure_count > 0:
        print(f"\nFAILED RECEIPTS:")
        print(f"{'='*80}")
        for result in results:
            if not result['success']:
                print(f"❌ {result['filename']}")
                if result['error']:
                    print(f"   Error: {result['error']}")
                else:
                    print(f"   Bank detected: {result.get('bank_detected', 'Unknown')}")
                    print(f"   IDs found: {result.get('all_ids', [])}")
                    print(f"   Text preview: {result.get('extracted_text_preview', 'N/A')[:100]}...")
                print()
    
    # Save detailed results to JSON
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_receipts': len(receipt_files),
            'success_count': success_count,
            'failure_count': failure_count,
            'accuracy': success_count/len(receipt_files)*100,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Return non-zero exit code if not 100% accuracy
    if failure_count > 0:
        print(f"\n⚠️  NOT AT 100% ACCURACY - Need to fix {failure_count} receipts")
        sys.exit(1)
    else:
        print(f"\n🎉 100% ACCURACY ACHIEVED!")
        sys.exit(0)

if __name__ == "__main__":
    main()
