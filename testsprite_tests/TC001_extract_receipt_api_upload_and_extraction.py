import requests

def test_tc001_extract_receipt_api_upload_and_extraction():
    base_url = "http://localhost:8081"
    endpoint = f"{base_url}/extract"
    timeout = 60  # Increased timeout for OCR + Database operations
    headers = {}
    # Use existing test image (from data/uploads)
    file_paths = ["data/uploads/e031113b-b29d-4733-a196-36b07bf005e3.jpg"]

    for file_path in file_paths:
        try:
            with open(file_path, "rb") as f:
                # Explicitly set mime type for jpg
                mime_type = "image/jpeg" if file_path.lower().endswith(('.jpg', '.jpeg')) else "application/pdf"
                files = {"file": (file_path, f, mime_type)}
                response = requests.post(endpoint, files=files, headers=headers, timeout=timeout)
            # Validate response status code
            assert response.status_code == 200, f"Expected status 200 but got {response.status_code}"

            json_resp = response.json()
            # Validate success field
            assert "success" in json_resp and isinstance(json_resp["success"], bool), "Missing or invalid 'success' field"
            assert json_resp["success"] is True, "'success' should be True"

            # Validate data field
            assert "data" in json_resp and isinstance(json_resp["data"], dict), "Missing or invalid 'data' field"

            data = json_resp["data"]
            print(f"DEBUG Response Data: {data}")
            # Validate required extraction fields presence and types
            for key in ["bank", "transaction_id", "amount", "date"]:
                assert key in data, f"Missing '{key}' in response data"
                assert isinstance(data[key], str), f"'{key}' should be of type string"
                # Additional non-empty check for extracted fields
                assert data[key].strip() != "", f"'{key}' should not be empty"

        except (requests.exceptions.RequestException, AssertionError) as e:
            raise AssertionError(f"Test failed for file '{file_path}': {str(e)}")

test_tc001_extract_receipt_api_upload_and_extraction()