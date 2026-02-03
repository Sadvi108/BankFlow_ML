import requests

BASE_URL = "http://localhost:8081/"
TIMEOUT = 30

def test_health_check_api_status():
    url = BASE_URL + "health"
    headers = {
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
    except requests.RequestException as e:
        assert False, f"Request failed: {e}"

    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"

    try:
        json_data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    assert "status" in json_data, "'status' field missing in response JSON"
    assert isinstance(json_data["status"], str), "'status' field is not a string"

    assert json_data["status"].lower() in ["healthy", "ok", "up"], f"Unexpected status value: {json_data['status']}"


test_health_check_api_status()