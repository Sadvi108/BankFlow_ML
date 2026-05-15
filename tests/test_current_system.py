#!/usr/bin/env python3
"""
Quick test script to demonstrate the current system capabilities
while ML training is in progress.
"""

import json
import requests
from pathlib import Path
import time

def test_current_system():
    """Test the current receipt processing system."""
    
    print("🧪 Testing Current Receipt Processing System")
    print("=" * 50)
    
    # Test the enhanced API
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ API Server is running")
        else:
            print("❌ API Server health check failed")
            return
    except Exception as e:
        print(f"❌ Could not connect to API: {e}")
        return
    
    # Get list of processed receipts
    try:
        response = requests.get(f"{base_url}/receipts")
        if response.status_code == 200:
            receipts = response.json()
            print(f"📋 Found {len(receipts)} processed receipts")
            
            # Show some examples
            print("\n📊 Sample Results:")
            print("-" * 30)
            
            for i, receipt in enumerate(receipts[:5]):  # Show first 5
                filename = receipt.get('filename', 'Unknown')
                fields = receipt.get('fields', {})
                transaction_id = fields.get('transaction_number', 'Not found')
                bank_type = fields.get('bank_type', 'Unknown')
                
                print(f"{i+1}. {filename}")
                print(f"   Bank: {bank_type}")
                print(f"   Transaction ID: {transaction_id}")
                print()
            
            # Show statistics
            print("📈 Overall Statistics:")
            print("-" * 20)
            
            total_receipts = len(receipts)
            receipts_with_transaction_id = 0
            bank_types = {}
            
            for receipt in receipts:
                fields = receipt.get('fields', {})
                if fields.get('transaction_number'):
                    receipts_with_transaction_id += 1
                
                bank = fields.get('bank_type', 'Unknown')
                bank_types[bank] = bank_types.get(bank, 0) + 1
            
            success_rate = (receipts_with_transaction_id / total_receipts) * 100 if total_receipts > 0 else 0
            
            print(f"Total receipts processed: {total_receipts}")
            print(f"Receipts with transaction IDs: {receipts_with_transaction_id}")
            print(f"Success rate: {success_rate:.1f}%")
            print()
            
            print("Bank distribution:")
            for bank, count in bank_types.items():
                percentage = (count / total_receipts) * 100
                print(f"  {bank}: {count} ({percentage:.1f}%)")
                
    except Exception as e:
        print(f"❌ Error getting receipts: {e}")
        return
    
    print("\n" + "=" * 50)
    print("🎯 System Status: Ready for ML Training")
    print("📊 Training is currently running to improve accuracy")
    print("🚀 Full deployment capabilities available")

if __name__ == "__main__":
    test_current_system()