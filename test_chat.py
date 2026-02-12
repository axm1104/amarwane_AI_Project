
import requests
import json

url = "http://localhost:8081/chat"
headers = {"Content-Type": "application/json"}
data = {
    "message": "Hello, are you there?",
    "user_id": "test_user",
    "session_id": "test_session"
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, headers=headers, json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
