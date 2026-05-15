"""
Automated Receipt Tester - Standalone Version
Tests all receipts without needing a running server
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime
import logging

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3

# Configuration
RECEIPTS_DIR = Path(__file__).parent / "Receipts"
RESULTS_DIR = Path(__file__).parent / "test_reports"
RESULTS_DIR.mkdir(exist_ok=True)

def test_single_receipt(receipt_path: Path, ocr_pipeline: EnhancedOCRPipeline) -> dict:
    """Test a single receipt file"""
    result = {
        'filename': receipt_path.name,
        'filepath': str(receipt_path),
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
        'extracted_text_preview': None,
        'full_text_length': 0
    }
    
    logger.info(f"\n{'='*100}")
    logger.info(f"Testing: {receipt_path.name}")
    logger.info(f"{'='*100}")
    
    try:
        # Process file with OCR
        ocr_result = ocr_pipeline.process_file(str(receipt_path))
        text = ocr_result.get('text', '')
        result['ocr_method'] = ocr_result.get('method', 'unknown')
        result['confidence'] = ocr_result.get('confidence', 0.0)
        result['full_text_length'] = len(text)
        
        # Store text preview
        result['extracted_text_preview'] = text[:300] if text else None
        
        logger.info(f"OCR Method: {result['ocr_method']}")
        logger.info(f"OCR Confidence: {result['confidence']:.2%}")
        logger.info(f"Text Length: {result['full_text_length']} characters")
        logger.info(f"Text Preview: {text[:200]}...")
        
        # Extract fields using pattern matching
        extraction = extract_all_fields_v3(text)
        
        result['bank_detected'] = extraction.get('bank_name')
        result['transaction_ids'] = extraction.get('transaction_ids', [])
        result['reference_ids'] = extraction.get('reference_ids', [])
        result['all_ids'] = extraction.get('all_ids', [])
        result['amount'] = extraction.get('amount')
        result['date'] = extraction.get('date')
        
        # Determine success - must have at least one ID
        if result['all_ids'] and len(result['all_ids']) > 0:
            result['success'] = True
            logger.info(f"✅ SUCCESS!")
            logger.info(f"  Bank: {result['bank_detected']}")
            logger.info(f"  All IDs: {result['all_ids']}")
            logger.info(f"  Transaction IDs: {result['transaction_ids']}")
            logger.info(f"  Reference IDs: {result['reference_ids']}")
            logger.info(f"  Amount: {result['amount']}")
            logger.info(f"  Date: {result['date']}")
        else:
            result['success'] = False
            logger.error(f"❌ FAILED - No transaction IDs extracted")
            logger.error(f"  Bank detected: {result['bank_detected']}")
            logger.error(f"  OCR text preview: {result['extracted_text_preview']}")
            
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        logger.error(f"❌ EXCEPTION: {e}", exc_info=True)
    
    return result

def main():
    """Main testing function"""
    print("\n" + "="*100)
    print("AUTOMATED RECEIPT TESTING - STANDALONE MODE")
    print("="*100)
    
    # Initialize OCR pipeline
    print("\nInitializing OCR pipeline...")
    ocr_pipeline = EnhancedOCRPipeline()
    print("✅ OCR pipeline initialized")
    
    # Find all receipt files
    receipt_files = sorted(RECEIPTS_DIR.glob("*.*"))
    receipt_files = [f for f in receipt_files if f.suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png']]
    
    print(f"\nFound {len(receipt_files)} receipts to test")
    print(f"Location: {RECEIPTS_DIR}")
    print("="*100)
    
    # Test each receipt
    all_results = []
    success_count = 0
    failure_count = 0
    
    for receipt_path in receipt_files:
        result = test_single_receipt(receipt_path, ocr_pipeline)
        all_results.append(result)
        
        if result['success']:
            success_count += 1
        else:
            failure_count += 1
    
    # Calculate accuracy
    total = len(receipt_files)
    accuracy = (success_count / total * 100) if total > 0 else 0
    
    # Print summary
    print("\n\n" + "="*100)
    print("TEST SUMMARY")
    print("="*100)
    print(f"Total Receipts Tested: {total}")
    print(f"✅ Successful Extractions: {success_count}")
    print(f"❌ Failed Extractions: {failure_count}")
    print(f"📊 Accuracy: {accuracy:.2f}%")
    print("="*100)
    
    # Print detailed failure information
    if failure_count > 0:
        print("\n" + "="*100)
        print("FAILED RECEIPTS DETAILS")
        print("="*100)
        
        for result in all_results:
            if not result['success']:
                print(f"\n❌ {result['filename']}")
                print(f"   Bank detected: {result.get('bank_detected', 'Unknown')}")
                print(f"   OCR Method: {result.get('ocr_method', 'Unknown')}")
                print(f"   OCR Confidence: {result.get('confidence', 0):.2%}")
                print(f"   Text Length: {result.get('full_text_length', 0)} chars")
                
                if result.get('error'):
                    print(f"   Error: {result['error']}")
                else:
                    print(f"   IDs Found: {result.get('all_ids', [])}")
                    preview = result.get('extracted_text_preview', '')
                    if preview:
                        print(f"   Text Preview: {preview[:150]}...")
    
    # Save detailed results to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f"automated_test_results_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_receipts': total,
            'success_count': success_count,
            'failure_count': failure_count,
            'accuracy_percent': accuracy,
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Print bank distribution
    print("\n" + "="*100)
    print("BANK DISTRIBUTION")
    print("="*100)
    bank_stats = {}
    for result in all_results:
        bank = result.get('bank_detected', 'Unknown')
        if bank not in bank_stats:
            bank_stats[bank] = {'total': 0, 'success': 0, 'failed': 0}
        bank_stats[bank]['total'] += 1
        if result['success']:
            bank_stats[bank]['success'] += 1
        else:
            bank_stats[bank]['failed'] += 1
    
    for bank, stats in sorted(bank_stats.items()):
        bank_acc = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{bank:20s}: {stats['success']:2d}/{stats['total']:2d} ({bank_acc:5.1f}%)")
    
    # Final message
    print("\n" + "="*100)
    if accuracy == 100:
        print("🎉 100% ACCURACY ACHIEVED! All receipts processed successfully!")
    else:
        print(f"⚠️  Current accuracy: {accuracy:.2f}%")
        print(f"   Need to fix {failure_count} failing receipts to reach 100%")
    print("="*100 + "\n")
    
    # Return exit code
    return 0 if accuracy == 100 else 1

if __name__ == "__main__":
    sys.exit(main())
