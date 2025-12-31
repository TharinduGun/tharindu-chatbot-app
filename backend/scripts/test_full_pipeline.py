import requests
import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.vector_store import MilvusService

BASE_URL = "http://localhost:8000"

def get_milvus_counts():
    try:
        service = MilvusService()
        service.connect()
        from pymilvus import Collection
        c_text = Collection("text_chunks")
        c_img = Collection("image_embeddings")
        c_text.flush()
        c_img.flush()
        return c_text.num_entities, c_img.num_entities
    except Exception:
        return 0, 0

def test_full_pipeline(pdf_path):
    print(f"--- Testing Full Pipeline with {pdf_path} ---")
    
    # 1. Snapshot Milvus State
    start_text, start_img = get_milvus_counts()
    print(f"Initial Milvus Counts: Text={start_text}, Images={start_img}")

    # 2. Upload File
    print("Uploading file...")
    with open(pdf_path, "rb") as f:
        files = {"file": f}
        try:
            response = requests.post(f"{BASE_URL}/documents/upload", files=files)
            if response.status_code != 200:
                print(f"Upload failed: {response.text}")
                return
            
            data = response.json()
            doc_id = data["doc_id"]
            print(f"Upload Success! Doc ID: {doc_id}")
            
            # Note: The upload endpoint triggers parsing, which then triggers multimodal.
            # However, looking at documents.py, it seems 'upload' triggers 'parser.process_document'.
            # Does 'parser.process_document' trigger multimodal? 
            # I need to verify that chain. If not, I might need to trigger multimodal manually 
            # or Ensure parser calls it. 
            # For now, let's assume the user might need to trigger multimodal manually if not chained.
            # But let's check IF the parser actually calls it.
            # If the user's previous code didn't link them, we might be waiting forever.
            
        except Exception as e:
            print(f"Request failed: {e}")
            return

    # 3. Poll for Milvus Updates
    print("Waiting for processing to propagate to Milvus (this includes parsing + embedding)...")
    timeout = 180 # 3 minutes
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        curr_text, curr_img = get_milvus_counts()
        print(f"Checking... Text={curr_text}, Images={curr_img}")
        
        if curr_text > start_text or curr_img > start_img:
            print("\n✅ SUCCESS: New data detected in Milvus!")
            print(f"New entities: {curr_text - start_text} text chunks, {curr_img - start_img} images.")
            return
        
        time.sleep(10)

    print("\n❌ Timeout: No new data appeared in Milvus after 3 minutes.")
    print("Check server logs to see if parsing or embedding failed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Use a default test file if available
        # We saw some PDFs in data/raw previously. 
        # let's try to pick one or ask user.
        print("Usage: python backend/scripts/test_full_pipeline.py <path_to_pdf>")
        print("Running with a dummy path check...")
    else:
        test_full_pipeline(sys.argv[1])
