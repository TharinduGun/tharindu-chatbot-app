import requests
import time
import os
import json

# Configuration
BASE_URL = "http://localhost:8000"
PDF_PATH = r"D:\Work\jwinfotech\Assignment 2\Assignment 2 - JW Infotech - AIML and Data Science Interns November 2025.pdf"

def main():
    print(f"Checking if PDF exists: {PDF_PATH}")
    if not os.path.exists(PDF_PATH):
        print("Error: Test PDF not found.")
        return

    print("Submitting PDF for processing...")
    try:
        with open(PDF_PATH, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{BASE_URL}/documents/upload", files=files)
            
        if response.status_code != 200:
            print(f"Error: Upload failed with {response.status_code}: {response.text}")
            return
            
        result = response.json()
        doc_id = result["doc_id"]
        print(f"Upload successful. Doc ID: {doc_id}")
        
    except Exception as e:
        print(f"Error during upload request: {e}")
        return

    print("Waiting for background processing (polling result file)...")
    processed_file = f"backend/data/processed/{doc_id}/metadata.json"
    error_file = f"backend/data/processed/{doc_id}/error.txt"
    waited = 0
    TIMEOUT = 300 # Increased to 5 minutes to allow for model download on first run
    
    while waited < TIMEOUT: 
        if os.path.exists(processed_file):
            print(f"Processing complete! File found at {processed_file}")
            break
        
        if os.path.exists(error_file):
            print(f"Processing FAILED! Error file found at {error_file}")
            with open(error_file, "r") as ef:
                print(f"Error details: {ef.read()}")
            return

        time.sleep(2)
        waited += 2
        if waited % 10 == 0:
            print(f"Waiting... ({waited}s)")
    
    if not os.path.exists(processed_file):
        print("Timed out waiting for processing.")
        return

    # Verify content
    print("\nVerifying content...")
    with open(processed_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Document: {data['record']['filename']}")
    print(f"Sections: {len(data['sections'])}")
    print(f"Blocks: {len(data['blocks'])}")
    print(f"Chunks: {len(data['chunks'])}")
    print(f"Images: {len(data['images'])}")
    
    # Check hierarchy
    if data['sections']:
        print("\nSample Section 0:")
        print(f"Title: {data['sections'][0]['title']}")
        print(f"Level: {data['sections'][0]['level']}")
        
    if data['chunks']:
        print("\nSample Chunk 0:")
        print(f"Content Preview: {data['chunks'][0]['content'][:50]}...")
        print(f"Linked Images: {len(data['chunks'][0]['image_ids'])}")
        
    if data['images']:
        print(f"\nSample Image 0 Path: {data['images'][0]['file_path']}")
    else:
        print("\nNo images extracted (might be expected if PDF has none or Docling skipped them).")

if __name__ == "__main__":
    main()
