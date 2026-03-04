
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
    # Testing the DnD receipt (DuitNow Outward)
    file_path = "Receipts/9d459d0f-467c-46ab-a8c7-d80a425d39c9_DnD CONTROL - EMC12511748202.pdf"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Sending {file_path} to {url}...")
    
    try:
        with open(file_path, 'rb') as f:
            # Send raw binary for simple server
            data = f.read()
            headers = {'Content-Type': 'application/pdf'}
            response = requests.post(url, data=data, headers=headers)
            
        if response.status_code == 200:
            result = response.json()
            print("\nAPI Response:")
            print(json.dumps(result, indent=2))
            
            # The API returns 'transaction_id' (single) and 'all_ids' (list)
            # We check if our target ID is in the list or is the main ID
            ids = result.get('transaction_ids', []) # Note: Structure might vary, checking dump
            if not ids:
                 ids = result.get('all_ids', [])
            
            # Additional fallback for different response structures
            if not ids and 'transaction_id' in result:
                ids = [result['transaction_id']]

            expected_id = "EMC12511748202"
            
            # Normalize for comparison
            ids_norm = [str(x).replace(' ', '').upper() for x in ids]
            
            if expected_id in ids_norm:
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
