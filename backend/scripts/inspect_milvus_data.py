import sys
import os
from pymilvus import connections, Collection

# Add parent directory to path to import app if needed (though we use pymilvus directly here)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def inspect_collection(name, expected_dim):
    print(f"\n--- Inspecting Collection: {name} ---")
    try:
        collection = Collection(name)
        collection.load() # Load into memory to search/query
        
        count = collection.num_entities
        print(f"Total Entities: {count}")
        
        if count == 0:
            print("Collection is empty.")
            return

        # Query first 3 items
        # We need to know output fields. We'll simply ask for specific useful ones + wildcard? 
        # Wildcard output_fields=["*"] is supported in newer Milvus.
        results = collection.query(
            expr="", 
            limit=3, 
            output_fields=["*"] 
        )
        
        for i, res in enumerate(results):
            print(f"\n[Item {i+1}]")
            # Check embedding dim
            emb = res.get("embedding", [])
            print(f"  ID: {res.get('id')}")
            print(f"  Doc ID: {res.get('doc_id')}")
            
            if "text" in res:
                print(f"  Text (Snippet): {res['text'][:100]}...")
            if "caption" in res:
                print(f"  Caption: {res['caption'][:100]}...")
            if "image_path" in res:
                print(f"  Image Path: {res['image_path']}")
            
            if "metadata" in res:
                print(f"  Metadata: {res['metadata']}")
                
            dim = len(emb)
            print(f"  Embedding Dim: {dim}")
            
            if dim == expected_dim:
                print("  ✅ Dimension Check Passed")
            else:
                print(f"  ❌ Dimension Mismatch! Expected {expected_dim}, got {dim}")

    except Exception as e:
        print(f"Error inspecting {name}: {e}")

def main():
    try:
        connections.connect("default", host="localhost", port="19530")
        print("Connected to Milvus.")
        
        inspect_collection("text_chunks", 1024)
        inspect_collection("image_embeddings", 1152)
        
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    main()
