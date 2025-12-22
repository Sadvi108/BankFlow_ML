
import requests
import json
import sys
import os

# Fix encoding
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def verify_api():
    url = "http://localhost:8081/extract"
    file_path = "Receipts/PUBLIC BANK - INVOICE.pdf"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Sending {file_path} to {url}...")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
            response = requests.post(url, files=files)
            
        if response.status_code == 200:
            result = response.json()
            print("\nAPI Response:")
            print(json.dumps(result, indent=2))
            
            ids = result.get('data', {}).get('all_ids', [])
            expected_id = "2510300265794708"
            
            if expected_id in ids:
                print(f"\n✅ SUCCESS: Found expected ID {expected_id}")
            else:
                print(f"\n❌ FAILURE: Expected ID {expected_id} not found. Got: {ids}")
        else:
            print(f"\n❌ Error: Status code {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")

if __name__ == "__main__":
    verify_api()
