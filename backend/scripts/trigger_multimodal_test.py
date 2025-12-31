import requests
import sys
import os
import time

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vector_store import MilvusService
from pymilvus import Collection

DOC_ID = "13640887-cbdd-49c7-a844-36a874c7b361" # Example doc
API_URL = f"http://localhost:8000/documents/{DOC_ID}/process-multimodal"

def get_count(collection_name):
    try:
        c = Collection(collection_name)
        c.flush()
        return c.num_entities
    except Exception as e:
        print(f"Error getting count for {collection_name}: {e}")
        return 0

def main():
    print(f"Triggering Multimodal Pipeline for {DOC_ID}...")
    
    # 1. Connect to Milvus
    service = MilvusService()
    service.connect()
    
    initial_text_count = get_count("text_chunks")
    initial_image_count = get_count("image_embeddings")
    
    print(f"Initial Counts -> Text: {initial_text_count}, Images: {initial_image_count}")
    
    # 2. Trigger API
    try:
        response = requests.post(API_URL)
        if response.status_code == 200:
            print("API Triggered Successfully.")
        else:
            print(f"API Failed: {response.text}")
            return
    except Exception as e:
        print(f"Failed to call API: {e}")
        print("Ensure uvicorn is running on port 8000")
        return

    # 3. Poll for changes
    print("Waiting for processing (timeout 120s)...")
    for i in range(24): # 24 * 5s = 120s
        time.sleep(5)
        current_text = get_count("text_chunks")
        current_image = get_count("image_embeddings")
        
        if current_text > initial_text_count or current_image > initial_image_count:
            print(f"SUCCESS: Counts increased! Text: {current_text}, Images: {current_image}")
            # Wait a bit more to let it finish
            time.sleep(10)
            final_text = get_count("text_chunks")
            final_image = get_count("image_embeddings")
            print(f"Final Counts -> Text: {final_text}, Images: {final_image}")
            return
        
        print(f"Waiting... ({i*5}s) Text: {current_text}, Images: {current_image}")

    print("Timed out waiting for Milvus updates.")

if __name__ == "__main__":
    main()
