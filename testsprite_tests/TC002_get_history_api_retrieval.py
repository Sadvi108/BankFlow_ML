import requests

BASE_URL = "http://localhost:8081/"
TIMEOUT = 30
HEADERS = {
    "Accept": "application/json",
}

def test_get_history_api_retrieval():
    try:
        response = requests.get(f"{BASE_URL}history", headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        assert False, f"Request to /history failed: {e}"

    json_data = response.json()

    # Validate top-level keys
    assert isinstance(json_data, dict), "Response is not a JSON object"
    assert "success" in json_data, "'success' key missing in response"
    assert "history" in json_data, "'history' key missing in response"

    # Validate success is True
    assert json_data["success"] is True, "'success' key is not True"

    # Validate history is a list
    assert isinstance(json_data["history"], list), "'history' is not a list"

    # Optionally: validate the structure of items inside history if any
    for item in json_data["history"]:
        assert isinstance(item, dict), "Each item in history should be a dict"

test_get_history_api_retrieval()