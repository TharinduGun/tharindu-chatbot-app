import logging
import os
import json
import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModel, SiglipProcessor, SiglipModel
import numpy as np
from app.services import storage

# Configure logger to write to file as well
logging.basicConfig(
    filename='debug_multimodal.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class MultimodalPipeline:
    # Class-level cache to prevent reloading on every request
    _siglip_processor = None
    _siglip_model = None
    _bge_tokenizer = None
    _bge_model = None
    _blip_processor = None
    _blip_model = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Models are now lazy-loaded on first use via _load_... methods

    def _load_siglip(self):
        if MultimodalPipeline._siglip_model is None:
            logger.info("Loading SigLIP...")
            MultimodalPipeline._siglip_processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")
            MultimodalPipeline._siglip_model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384").to(self.device)
    
    def _load_bge(self):
        if MultimodalPipeline._bge_model is None:
            logger.info("Loading BGE...")
            MultimodalPipeline._bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
            MultimodalPipeline._bge_model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to(self.device)

    def _load_blip(self):
        if MultimodalPipeline._blip_model is None:
            logger.info("Loading BLIP (Large) for caption generation...")
            try:
                # Switching to BLIP-Large (approx 1.8GB) instead of BLIP-2 (10GB+) to avoid OOM/Hang
                MultimodalPipeline._blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
                
                MultimodalPipeline._blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-large", 
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
            except Exception as e:
                logger.error(f"Failed to load BLIP: {e}")
                raise e

    def get_bge_embedding(self, text: str) -> List[float]:
        self._load_bge()
        inputs = MultimodalPipeline._bge_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = MultimodalPipeline._bge_model(**inputs)
            # CLS pooling for BGE usually, or specific instructions. 
            # BGE uses [CLS] token embedding for classification/similarity tasks usually.
            # Official BGE usage: model_output[0][:, 0]
            embeddings = outputs.last_hidden_state[:, 0]
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings[0].cpu().numpy().tolist()

    def get_siglip_image_embedding(self, image_path: str) -> List[float]:
        self._load_siglip()
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = MultimodalPipeline._siglip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = MultimodalPipeline._siglip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features[0].cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error embedding image {image_path}: {e}")
            return []

    def get_siglip_text_embedding(self, text: str) -> List[float]:
        self._load_siglip()
        inputs = MultimodalPipeline._siglip_processor(text=[text], return_tensors="pt", padding="max_length", truncation=True).to(self.device)
        with torch.no_grad():
            text_features = MultimodalPipeline._siglip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features[0].cpu().numpy().tolist()

    def generate_caption(self, image_path: str) -> str:
        self._load_blip()
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = MultimodalPipeline._blip_processor(image, return_tensors="pt").to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
            
            generated_ids = MultimodalPipeline._blip_model.generate(**inputs, max_new_tokens=40)
            caption = MultimodalPipeline._blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return caption
        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {e}")
            return ""

    def validate_caption(self, caption: str) -> bool:
        if not caption: return False
        if len(caption.split()) < 3: return False # Too short
        if "Figure" in caption and len(caption) < 15: return False # "Figure 1" generic
        return True

    def run(self, doc_id: str):
        logger.info(f"Starting Multimodal Phase for {doc_id}")
        doc_dir = storage.PROCESSED_DIR / doc_id
        metadata_path = doc_dir / "metadata.json"
        
        if not metadata_path.exists():
            logger.error(f"Metadata not found for {doc_id}")
            return

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Initialize Milvus
        try:
            from app.services.vector_store import MilvusService
            milvus_service = MilvusService()
            milvus_service.connect()
        except Exception as e:
            logger.error(f"Failed to initialize Milvus: {e}")
            milvus_service = None

        # 1. Process Images (Captions + Embeddings)
        images = data.get("images", [])
        images_to_insert = []
        
        for img in images:
            img_path = img["file_path"]
            print(f"[DEBUG] Checking Image: {img_path}")
            print(f"[DEBUG] Absolute Path: {os.path.abspath(img_path)}")
            
            if not os.path.exists(img_path):
                logger.warning(f"Image file missing: {img_path}")
                print(f"[DEBUG] MISSING: {img_path}")
                continue
            
            # Caption Logic
            caption_source = "pdf"
            caption_final = img.get("caption_raw", "")
            
            if not self.validate_caption(caption_final):
                logger.info(f"Generating fallback caption for {img['image_id']}")
                caption_final = self.generate_caption(img_path)
                caption_source = "blip2"
            
            img["caption_final"] = caption_final
            img["caption_source"] = caption_source
            
            # Image Embeddings (SigLIP)
            print(f"[DEBUG] Generatng Embedding for {img_path}...")
            emb = self.get_siglip_image_embedding(img_path)
            img["embedding_siglip_image"] = emb
            print(f"[DEBUG] Embedding Size: {len(emb)}")
            
            # SigLIP embedding for the caption itself (for image-text matching)
            if caption_final:
                img["embedding_siglip_caption"] = self.get_siglip_text_embedding(caption_final)
            else:
                img["embedding_siglip_caption"] = []
            
        # 2. Process Text Chunks (Embeddings only, insertion later)
        chunks = data.get("chunks", [])
        
        for chunk in chunks:
            # Contextual embedding: Section Title + Content
            # Find section title
            section = next((s for s in data["sections"] if s["section_id"] == chunk["section_id"]), None)
            context = f"{section['title']}: {chunk['content']}" if section else chunk['content']
            
            chunk["embedding_bge"] = self.get_bge_embedding(context)
            
            # Also compute SigLIP text embedding for this chunk to allow image-text matching
            chunk["embedding_siglip"] = self.get_siglip_text_embedding(chunk['content'])

        # 3. Image-Text Matching
        # For each image, find best chunks
        logger.info("Running Image-Text Matching...")
        
        for img in images:
            if "embedding_siglip_image" not in img or not img["embedding_siglip_image"]:
                continue
                
            image_vec = np.array(img["embedding_siglip_image"])
            caption_vec = np.array(img["embedding_siglip_caption"]) if "embedding_siglip_caption" in img else None
            
            candidates = []
            
            # Filter candidates: Same Page / Section
            # Metadata-first filtering
            relevant_chunks = [
                c for c in chunks 
                if c["page_no"] == img["page_no"] or c["section_id"] == next((b["section_id"] for b in data["blocks"] if img["image_id"] in b["image_ids"]), "")
            ]
            
            # Fallback: if no candidates on same page/section, search global (but maybe restricted to +/- 2 pages)
            if not relevant_chunks:
                 relevant_chunks = [c for c in chunks if abs(c["page_no"] - img["page_no"]) <= 1]
            
            best_score = -1.0
            best_chunk_id = None
            
            for chunk in relevant_chunks:
                if "embedding_siglip" not in chunk: continue
                
                text_vec = np.array(chunk["embedding_siglip"])
                
                # Similarity: Image <-> Text
                score_img = np.dot(image_vec, text_vec)
                
                # Similarity: Caption <-> Text (Optional boost)
                score_cap = 0
                if caption_vec is not None:
                    score_cap = np.dot(caption_vec, text_vec)
                
                # Hybrid Score
                final_score = 0.7 * score_img + 0.3 * score_cap if caption_vec is not None else score_img
                
                if final_score > best_score:
                    best_score = final_score
                    best_chunk_id = chunk["chunk_id"]
            
            if best_chunk_id and best_score > 0.25: # Threshold
                img["linked_chunk_id"] = best_chunk_id
                img["match_score"] = float(best_score)
                
                # Bi-directional link
                for c in chunks:
                    if c["chunk_id"] == best_chunk_id:
                        if "linked_image_ids" not in c: c["linked_image_ids"] = []
                        if img["image_id"] not in c["linked_image_ids"]:
                            c["linked_image_ids"].append(img["image_id"])
                        break

        # -------------------------------------------------------
        # MILVUS INSERTION (Moved to AFTER Matching)
        # -------------------------------------------------------
        if milvus_service:
            # 1. Insert Images
            images_to_insert = []
            for img in images:
                if img.get("embedding_siglip_image"):
                    metadata = {}
                    if "linked_chunk_id" in img:
                        metadata["linked_chunk_id"] = img["linked_chunk_id"]
                        metadata["match_score"] = img["match_score"]
                    
                    images_to_insert.append({
                        "id": img["image_id"],
                        "embedding": img["embedding_siglip_image"],
                        "doc_id": doc_id,
                        "image_path": img["file_path"],
                        "caption": img["caption_final"][:2000],
                        "metadata": metadata
                    })
            if images_to_insert:
                try:
                    print(f"============================================================")
                    print(f"🖼️  Inserting {len(images_to_insert)} images into Milvus...")
                    milvus_service.insert_images(images_to_insert)
                    print(f"✅  Successfully inserted images to Milvus.")
                except Exception as e:
                    logger.error(f"Failed to insert images to Milvus: {e}")
                    print(f"❌  Failed to insert images: {e}")

            # 2. Insert Text Chunks
            text_chunks_to_insert = []
            for chunk in chunks:
                if chunk.get("embedding_bge"):
                    metadata = {}
                    if "linked_image_ids" in chunk and chunk["linked_image_ids"]:
                        metadata["linked_image_ids"] = chunk["linked_image_ids"]
                    
                    text_chunks_to_insert.append({
                        "id": chunk["chunk_id"],
                        "embedding": chunk["embedding_bge"],
                        "doc_id": doc_id,
                        "text": chunk["content"][:60000],
                        "metadata": metadata
                    })
            if text_chunks_to_insert:
                try:
                    print(f"📄  Inserting {len(text_chunks_to_insert)} text chunks into Milvus...")
                    milvus_service.insert_text(text_chunks_to_insert)
                    print(f"✅  Successfully inserted text to Milvus.")
                except Exception as e:
                    logger.error(f"Failed to insert text to Milvus: {e}")
                    print(f"❌  Failed to insert text: {e}")
            
            print(f"============================================================")
        # -------------------------------------------------------

        # Save Updated Data
        data["chunks"] = chunks
        data["images"] = images
        
        # --- NEW: Save Readable Summary for User Inspection ---
        try:
            summary = []
            for img in images:
                linked_text = ""
                if "linked_chunk_id" in img:
                    # Find chunk content
                    chk = next((c for c in chunks if c["chunk_id"] == img["linked_chunk_id"]), None)
                    if chk:
                        linked_text = chk["content"]

                summary.append({
                    "image_id": img["image_id"],
                    "caption": img.get("caption_final", ""),
                    "caption_source": img.get("caption_source", ""),
                    "match_score": img.get("match_score", None),
                    "linked_chunk_id": img.get("linked_chunk_id", None),
                    "linked_text_snippet": linked_text[:200] + "..." if linked_text else None
                })
            
            summary_path = doc_dir / "multimodal_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"Saved multimodal summary to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save multimodal summary: {e}")
        # -------------------------------------------------------

        storage.save_processed_data(doc_id, data)
        logger.info(f"Multimodal processing complete for {doc_id}")
        
        print(f"")
        print(f"🎉🎉🎉 PIPELINE COMPLETED SUCCESSFULLY for {doc_id} 🎉🎉🎉")
        print(f"Summary saved to: {storage.PROCESSED_DIR / doc_id / 'multimodal_summary.json'}")
        print(f"============================================================")

if __name__ == "__main__":
    # Test run
    # import sys
    # if len(sys.argv) > 1:
    #     pipeline = MultimodalPipeline()
    #     pipeline.run(sys.argv[1])
    pass
