import requests
import os
import glob
import json
from pathlib import Path

BASE_URL = "http://localhost:8081"
EXTRACT_URL = f"{BASE_URL}/extract"

def test_extraction():
    # Find all receipt images
    image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.pdf", "data/uploads/*.jpg", "data/uploads/*.pdf"]
    files = []
    for pattern in image_patterns:
        files.extend(glob.glob(pattern))
    
    # Filter out result images or system files if any
    files = [f for f in files if "result" not in f and "debug" not in f]
    
    print(f"Found {len(files)} files to test.")
    
    results = []
    
    # Test a sample of 20 files (prioritizing PDFs if available)
    pdf_files = [f for f in files if f.endswith(".pdf")]
    other_files = [f for f in files if not f.endswith(".pdf")]
    files_to_test = pdf_files[:15] + other_files[:5]
    
    for file_path in files_to_test:
        print(f"\nTesting {file_path}...")
        try:
            mime_type = "application/pdf" if file_path.endswith(".pdf") else "image/jpeg"
            with open(file_path, "rb") as f:
                response = requests.post(
                    EXTRACT_URL, 
                    files={"file": (os.path.basename(file_path), f, mime_type)},
                    timeout=60
                )
            
            if response.status_code == 200:
                data = response.json().get("data", {})
                print(f"  ✅ Success!")
                print(f"  Bank: {data.get('bank')}")
                print(f"  ID:   {data.get('transaction_id')}")
                print(f"  Amt:  {data.get('amount')}")
                print(f"  Date: {data.get('date')}")
                
                # Check for potential issues
                issues = []
                if not data.get('bank') or data.get('bank') == "Unknown":
                    issues.append("Unknown Bank")
                if not data.get('transaction_id'):
                    issues.append("Missing ID")
                if not data.get('amount'):
                    issues.append("Missing Amount")
                
                results.append({
                    "file": file_path,
                    "status": "Success",
                    "data": data,
                    "issues": issues
                })
            else:
                print(f"  ❌ Failed: {response.status_code} - {response.text}")
                results.append({
                    "file": file_path,
                    "status": "Failed",
                    "error": response.text
                })
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "file": file_path,
                "status": "Error",
                "error": str(e)
            })

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for res in results:
        if res['status'] == 'Success':
            issues_str = ", ".join(res['issues']) if res['issues'] else "OK"
            print(f"{os.path.basename(res['file']):<30} | {str(res['data'].get('bank')):<15} | {str(res['data'].get('transaction_id')):<20} | Issues: {issues_str}")
        else:
            print(f"{os.path.basename(res['file']):<30} | FAILED")

if __name__ == "__main__":
    test_extraction()
