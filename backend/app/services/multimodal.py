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

logger = logging.getLogger(__name__)

class MultimodalPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Models - Lazy loading or initialized here. 
        # Since this is a dedicated pipeline step, we'll initialize them here to fail fast if missing.
        
        # 1. Image Embeddings: SigLIP
        logger.info("Loading SigLIP...")
        self.siglip_processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        self.siglip_model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384").to(self.device)
        
        # 2. Text Embeddings: BGE
        logger.info("Loading BGE...")
        self.bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        self.bge_model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to(self.device)
        
        # 3. Captioning: BLIP-2 (Loaded only if needed/fallback)
        self.blip_processor = None
        self.blip_model = None

    def _load_blip(self):
        if self.blip_model is None:
            logger.info("Loading BLIP (Large) for caption generation...")
            try:
                # Switching to BLIP-Large (approx 1.8GB) instead of BLIP-2 (10GB+) to avoid OOM/Hang
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
                
                self.blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-large", 
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
            except Exception as e:
                logger.error(f"Failed to load BLIP: {e}")
                raise e

    def get_bge_embedding(self, text: str) -> List[float]:
        inputs = self.bge_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.bge_model(**inputs)
            # CLS pooling for BGE usually, or specific instructions. 
            # BGE uses [CLS] token embedding for classification/similarity tasks usually.
            # Official BGE usage: model_output[0][:, 0]
            embeddings = outputs.last_hidden_state[:, 0]
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings[0].cpu().numpy().tolist()

    def get_siglip_image_embedding(self, image_path: str) -> List[float]:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.siglip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.siglip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            return image_features[0].cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error embedding image {image_path}: {e}")
            return []

    def get_siglip_text_embedding(self, text: str) -> List[float]:
        inputs = self.siglip_processor(text=[text], return_tensors="pt", padding="max_length", truncation=True).to(self.device)
        with torch.no_grad():
            text_features = self.siglip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features[0].cpu().numpy().tolist()

    def generate_caption(self, image_path: str) -> str:
        self._load_blip()
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
            
            generated_ids = self.blip_model.generate(**inputs, max_new_tokens=40)
            caption = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
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

        # 1. Process Images (Captions + Embeddings)
        images = data.get("images", [])
        images_map = {img["image_id"]: img for img in images}
        
        for img in images:
            img_path = img["file_path"]
            if not os.path.exists(img_path):
                logger.warning(f"Image file missing: {img_path}")
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
            img["embedding_siglip_image"] = self.get_siglip_image_embedding(img_path)
            
            # Caption Embedding (SigLIP Text for Matching)
            if caption_final:
                img["embedding_siglip_caption"] = self.get_siglip_text_embedding(caption_final)

        # 2. Process Text Chunks (BGE Embeddings)
        chunks = data.get("chunks", [])
        
        for chunk in chunks:
            # Contextual embedding: Section Title + Content
            # Find section title
            section = next((s for s in data["sections"] if s["section_id"] == chunk["section_id"]), None)
            context = f"{section['title']}: {chunk['content']}" if section else chunk['content']
            
            chunk["embedding_bge"] = self.get_bge_embedding(context)
            
            # Also compute SigLIP text embedding for this chunk to allow image-text matching
            # Note: We match Image (SigLIP) <-> Text (SigLIP). We cannot match Image (SigLIP) <-> Text (BGE).
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

if __name__ == "__main__":
    # Test run
    # import sys
    # if len(sys.argv) > 1:
    #     pipeline = MultimodalPipeline()
    #     pipeline.run(sys.argv[1])
    pass
