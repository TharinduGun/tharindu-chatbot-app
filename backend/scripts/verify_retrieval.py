import sys
import os
import torch
from transformers import AutoTokenizer, AutoModel, SiglipProcessor, SiglipModel

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vector_store import MilvusService

def get_bge_embedding(text):
    print("Loading BGE Model...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings[0].tolist()

def get_siglip_text_embedding(text):
    print("Loading SigLIP Model...")
    processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384")
    
    inputs = processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features[0].tolist()

def main():
    query = "diagram of the tooth structure"
    print(f"\nTest Query: '{query}'")
    
    # 1. Generate Embeddings
    vec_text = get_bge_embedding(query)
    vec_image = get_siglip_text_embedding(query)
    
    print(f"Generated BGE embedding (size: {len(vec_text)})")
    print(f"Generated SigLIP embedding (size: {len(vec_image)})")
    
    # 2. Search Milvus
    service = MilvusService()
    service.connect()
    
    print("\n--- Searching Text Collection (BGE) ---")
    results_text = service.search_text([vec_text], top_k=2)
    for hits in results_text:
        for hit in hits:
            print(f"Score: {hit.score:.4f} | Text: {hit.entity.get('text')[:100]}...")

    print("\n--- Searching Image Collection (SigLIP) ---")
    results_image = service.search_images([vec_image], top_k=2)
    for hits in results_image:
        for hit in hits:
            print(f"Score: {hit.score:.4f} | Caption: {hit.entity.get('caption')[:100]}... | Path: {hit.entity.get('image_path')}")

if __name__ == "__main__":
    main()
