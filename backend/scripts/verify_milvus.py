import sys
import os
import time

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vector_store import MilvusService

def verify_milvus():
    print("Attempting to connect to Milvus...")
    service = MilvusService()
    
    # Retry logic since docker might just be starting up
    max_retries = 5
    for i in range(max_retries):
        try:
            service.connect()
            print("\nSUCCESS: Connected to Milvus!")
            break
        except Exception as e:
            print(f"Connection failed (attempt {i+1}/{max_retries}): {e}")
            if i < max_retries - 1:
                time.sleep(5)
            else:
                print("Could not connect to Milvus.")
                return

    # Verify collections
    from pymilvus import utility
    collections = utility.list_collections()
    print(f"Existing collections: {collections}")
    
    expected = ["text_chunks", "image_embeddings"]
    for c in expected:
        if c in collections:
            print(f"Verified collection exists: {c}")
        else:
            print(f"ERROR: Collection {c} missing!")

if __name__ == "__main__":
    verify_milvus()
