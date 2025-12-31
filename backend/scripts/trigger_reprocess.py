import requests

DOC_ID = "feb69ee7-e327-46f7-b70e-42096e69aa4d"
URL = f"http://localhost:8000/documents/{DOC_ID}/process-multimodal"

print(f"Triggering reprocessing for {DOC_ID}...")
try:
    resp = requests.post(URL)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")
except Exception as e:
    print(f"Error: {e}")
