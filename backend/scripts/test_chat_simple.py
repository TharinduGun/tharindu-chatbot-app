import requests
import json

url = "http://localhost:8000/chat/query"
payload = {
    "query": "What is the structure of the tooth?",
    "provider": "openai"
}

print(f"Testing URL: {url}")
print(f"Payload: {json.dumps(payload)}")

try:
    resp = requests.post(url, json=payload)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")
except Exception as e:
    print(f"Detailed Error: {e}")
