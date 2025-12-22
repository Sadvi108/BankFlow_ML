#!/usr/bin/env python3
"""
Quick Start Script for Bank Receipt ML Training

This script helps you:
1. Ingest your 23 receipt samples
2. Review and annotate the data
3. Train ML models
4. Evaluate performance
"""

import sys
import time
from pathlib import Path
import json
import requests
from typing import Dict, List


def check_server_health(base_url: str = "http://localhost:8000") -> bool:
    """Check if the server is running."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def ingest_receipts(base_url: str = "http://localhost:8000") -> Dict:
    """Ingest receipts from the Receipts folder."""
    print("📁 Ingesting receipts from Receipts folder...")
    
    response = requests.post(f"{base_url}/ingest_receipts")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Successfully processed {result['processed']} receipts")
        if result['errors']:
            print(f"⚠️  {len(result['errors'])} errors occurred:")
            for error in result['errors']:
                print(f"   - {error['filename']}: {error['error']}")
        return result
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return {}


def get_dataset_summary(base_url: str = "http://localhost:8000") -> Dict:
    """Get current dataset summary."""
    response = requests.get(f"{base_url}/dataset/summary")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Error getting dataset summary: {response.status_code}")
        return {}


def display_dataset_stats(summary: Dict):
    """Display dataset statistics."""
    print("\n📊 Dataset Statistics:")
    print(f"   Total samples: {summary.get('total', 0)}")
    
    per_bank = summary.get('per_bank', {})
    if per_bank:
        print("   Samples per bank:")
        for bank, count in sorted(per_bank.items()):
            print(f"     - {bank}: {count}")
    else:
        print("   No bank distribution data available")


def train_models(base_url: str = "http://localhost:8000") -> bool:
    """Start ML model training."""
    print("\n🤖 Starting ML model training...")
    print("   This may take several minutes depending on your dataset size...")
    
    response = requests.post(f"{base_url}/train_models")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ {result['message']}")
        print(f"   Dataset size: {result['dataset_size']} samples")
        return True
    else:
        print(f"❌ Training failed: {response.status_code} - {response.text}")
        return False


def check_model_status(base_url: str = "http://localhost:8000") -> Dict:
    """Check model loading status."""
    response = requests.get(f"{base_url}/models/status")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Error checking model status: {response.status_code}")
        return {}


def test_single_receipt(base_url: str = "http://localhost:8000", receipt_path: str = None) -> bool:
    """Test processing a single receipt."""
    if not receipt_path:
        # Use first receipt from Receipts folder
        receipts_dir = Path("Receipts")
        if receipts_dir.exists():
            receipt_files = list(receipts_dir.glob("*.pdf")) + list(receipts_dir.glob("*.png")) + list(receipts_dir.glob("*.jpg"))
            if receipt_files:
                receipt_path = str(receipt_files[0])
            else:
                print("❌ No receipt files found in Receipts folder")
                return False
        else:
            print("❌ Receipts folder not found")
            return False
    
    print(f"\n🧪 Testing single receipt processing: {Path(receipt_path).name}")
    
    try:
        with open(receipt_path, 'rb') as f:
            files = {'file': (Path(receipt_path).name, f, 'application/pdf')}
            response = requests.post(f"{base_url}/extract", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Receipt processed successfully!")
            print(f"   Bank: {result['bank']['name']} (confidence: {result['bank']['confidence']:.2f})")
            print(f"   Transaction ID: {result.get('transaction_id', {}).get('transaction_id', 'Not found')}")
            print(f"   Processing time: {result['meta']['processing_time']:.2f}s")
            print(f"   ML models used: {result['meta']['ml_models_used']}")
            return True
        else:
            print(f"❌ Processing failed: {response.status_code} - {response.text}")
            return False
            
    except FileNotFoundError:
        print(f"❌ File not found: {receipt_path}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main training workflow."""
    print("🚀 Bank Receipt ML Training Quick Start")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Step 1: Check server health
    print("\n1️⃣  Checking server status...")
    if not check_server_health(base_url):
        print("❌ Server is not running!")
        print("   Please start the server first:")
        print("   uvicorn app.main_enhanced:app --reload --host 0.0.0.0 --port 8000")
        return 1
    print("✅ Server is running")
    
    # Step 2: Check current dataset
    print("\n2️⃣  Checking current dataset...")
    initial_summary = get_dataset_summary(base_url)
    if initial_summary:
        display_dataset_stats(initial_summary)
    else:
        print("⚠️  Could not retrieve dataset summary")
    
    # Step 3: Ingest receipts if needed
    if initial_summary.get('total', 0) < 10:
        print(f"\n3️⃣  Dataset has only {initial_summary.get('total', 0)} samples")
        print("   Ingesting receipts from Receipts folder...")
        ingest_result = ingest_receipts(base_url)
        
        # Check updated dataset
        time.sleep(2)  # Wait for processing
        updated_summary = get_dataset_summary(base_url)
        if updated_summary:
            print("\n📊 Updated dataset:")
            display_dataset_stats(updated_summary)
    else:
        print(f"\n3️⃣  Dataset already has {initial_summary.get('total', 0)} samples - skipping ingestion")
    
    # Step 4: Train models
    current_summary = get_dataset_summary(base_url)
    if current_summary.get('total', 0) >= 10:
        print(f"\n4️⃣  Training ML models with {current_summary.get('total', 0)} samples...")
        if train_models(base_url):
            print("   Training started successfully!")
            print("   ⏳ This will take several minutes...")
            
            # Wait a bit and check status
            print("\n   Waiting for training to complete...")
            for i in range(30):  # Wait up to 5 minutes
                time.sleep(10)
                model_status = check_model_status(base_url)
                if model_status.get('models_loaded'):
                    print("✅ Models loaded successfully!")
                    break
                else:
                    print(f"   Still training... ({i+1}/30)")
            else:
                print("⚠️  Training may still be in progress. Check logs for details.")
        else:
            print("❌ Training failed to start")
            return 1
    else:
        print(f"❌ Need at least 10 samples for training, got {current_summary.get('total', 0)}")
        return 1
    
    # Step 5: Test the system
    print("\n5️⃣  Testing the trained system...")
    test_single_receipt(base_url)
    
    # Final status
    print("\n✅ Quick start completed!")
    print("\nNext steps:")
    print("   1. Review and annotate your data at: http://localhost:8000/train")
    print("   2. Test with more receipts to validate performance")
    print("   3. Retrain models if needed with improved annotations")
    print("   4. Deploy to production using Docker")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())