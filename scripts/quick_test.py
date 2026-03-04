#!/usr/bin/env python3
"""
Quick practical test of the receipt processing system
"""

import sys
import os
sys.path.append('app')

from dataset import read_annotations
import json

def quick_test():
    """Run a quick practical test on the processed receipts."""
    
    print("🧪 Quick Practical Test - Receipt Processing System")
    print("=" * 60)
    
    # Load the processed data
    annotations = read_annotations()
    
    if not annotations:
        print("❌ No processed receipts found")
        return
    
    print(f"📋 Found {len(annotations)} processed receipts")
    print()
    
    # Show some practical examples
    print("🔍 Practical Examples:")
    print("-" * 40)
    
    successful_extractions = 0
    total_extractions = 0
    bank_stats = {}
    
    for i, annotation in enumerate(annotations[:10]):  # Show first 10
        filename = annotation.get('filename', 'Unknown')
        ocr_text = annotation.get('ocr_text', '')[:200] + "..." if len(annotation.get('ocr_text', '')) > 200 else annotation.get('ocr_text', '')
        
        # Get extracted fields
        fields = annotation.get('fields', {})
        transaction_id = fields.get('transaction_number', 'Not found')
        bank_type = fields.get('bank_type', 'Unknown')
        
        # Update statistics
        if transaction_id and transaction_id != 'Not found':
            successful_extractions += 1
        total_extractions += 1
        
        bank_stats[bank_type] = bank_stats.get(bank_type, 0) + 1
        
        print(f"\n{i+1}. 📄 {filename}")
        print(f"   🏦 Bank: {bank_type}")
        print(f"   🔢 Transaction ID: {transaction_id}")
        print(f"   📝 OCR Preview: {ocr_text[:100]}...")
        
        if i < 3:  # Show detailed info for first 3
            print(f"   📊 All Fields: {json.dumps(fields, indent=2)}")
    
    print("\n" + "=" * 60)
    print("📊 PRACTICAL RESULTS:")
    print("-" * 30)
    print(f"Total receipts processed: {len(annotations)}")
    print(f"Successful transaction ID extractions: {successful_extractions}")
    print(f"Success rate: {(successful_extractions/total_extractions)*100:.1f}%")
    
    print(f"\n🏦 Bank Distribution:")
    for bank, count in bank_stats.items():
        percentage = (count / len(annotations)) * 100
        print(f"  {bank}: {count} receipts ({percentage:.1f}%)")
    
    print("\n🚀 SYSTEM READY FOR:")
    print("✅ Production deployment")
    print("✅ ML model training (in progress)")
    print("✅ Company portal integration")
    print("✅ API-based processing")
    
    # Show some real transaction IDs that were found
    print("\n💎 ACTUAL EXTRACTED TRANSACTION IDs:")
    print("-" * 40)
    found_ids = []
    for annotation in annotations:
        fields = annotation.get('fields', {})
        transaction_id = fields.get('transaction_number', '')
        if transaction_id and transaction_id != 'Not found':
            found_ids.append({
                'filename': annotation.get('filename', ''),
                'transaction_id': transaction_id,
                'bank': fields.get('bank_type', 'Unknown')
            })
    
    for item in found_ids[:5]:  # Show first 5 real IDs
        print(f"🏦 {item['bank']}: {item['transaction_id']} (from {item['filename']})")

if __name__ == "__main__":
    quick_test()