import logging
import json
from typing import List, Dict, Any, Optional
from app.services.multimodal import MultimodalPipeline
from app.services.vector_store import MilvusService
from app.services.llm import get_llm_provider, LLMProvider

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, llm_provider: str = "openai"):
        self.multimodal = MultimodalPipeline()
        self.milvus = MilvusService()
        self.milvus.connect()
        
        # Initialize LLM
        try:
            self.llm: LLMProvider = get_llm_provider(llm_provider)
            logger.info(f"Initialized RAG Pipeline with LLM: {llm_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise e

    def retrieve_hybrid(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Performs hybrid retrieval:
        1. Embed query -> Text Vector (BGE)
        2. Embed query -> Image Vector (SigLIP)
        3. Search Milvus Text & Image Collections
        4. Merge results and expand context using Metadata Links
        """
        logger.info(f"Retrieving for query: {query}")
        
        # 1. Embed Query
        # Text Search Embedding (BGE)
        query_vec_bge = self.multimodal.get_bge_embedding(query)
        
        # Image Search Embedding (SigLIP)
        query_vec_siglip = self.multimodal.get_siglip_text_embedding(query)
        
        # 2. Search Milvus
        # Search Text
        text_results = self.milvus.search_text([query_vec_bge], top_k=top_k)
        
        # Search Images
        image_results = self.milvus.search_images([query_vec_siglip], top_k=top_k)
        
        # 3. Process & Merge Results
        retrieved_texts = []
        retrieved_images = []
        
        # Process Text Matches
        if text_results:
            for hit in text_results[0]:
                meta = hit.entity.get("metadata") or {}
                linked_imgs = meta.get("linked_image_ids", [])
                
                retrieved_texts.append({
                    "score": hit.score,
                    "text": hit.entity.get("text"),
                    "doc_id": hit.entity.get("doc_id"),
                    "linked_image_ids": linked_imgs
                })

        # Process Image Matches & Context Expansion
        if image_results:
            for hit in image_results[0]:
                meta = hit.entity.get("metadata") or {}
                linked_chunk_id = meta.get("linked_chunk_id")
                
                img_path = hit.entity.get("image_path")
                caption = hit.entity.get("caption")
                
                img_data = {
                    "score": hit.score,
                    "image_path": img_path,
                    "caption": caption,
                    "doc_id": hit.entity.get("doc_id"),
                    "linked_text": None
                }
                
                # Context Expansion: If image has a linked text chunk, we should fetch it?
                # For now, we rely on the caption. But if we wanted to be super smart, 
                # we could query Milvus by ID to get the text chunk. 
                # (Milvus doesn't support Get by ID easily in the same search flow, 
                # so we might skip this for v1 or implement a 'get_by_ids' in vector_store).
                
                retrieved_images.append(img_data)
        
        return {
            "text_results": retrieved_texts,
            "image_results": retrieved_images
        }

    async def answer_query(self, query: str) -> Dict[str, Any]:
        """
        End-to-end RAG flow: Retrieve -> Prompt -> Generate
        """
        # 1. Retrieve
        retrieval = self.retrieve_hybrid(query)
        texts = retrieval["text_results"]
        images = retrieval["image_results"]
        
        # 2. Context Construction
        context_str = ""
        
        # Add Text Context
        if texts:
            context_str += "--- TEXT CONTEXT ---\n"
            for i, t in enumerate(texts):
                context_str += f"Snippet {i+1}: {t['text'][:1000]}\n\n"
        
        # Add Image Context
        if images:
            context_str += "--- IMAGE MESSAGES ---\n"
            for i, img in enumerate(images):
                context_str += f"Image {i+1}: Found a relevant image containing '{img['caption']}'.\n"
                # We could add "File: {img['image_path']}" if we want the LLM to verify it.
        
        if not context_str:
            return {
                "answer": "I could not find any relevant information in the uploaded documents.",
                "sources": []
            }

        # 3. System Prompt
        system_prompt = (
            "You are an expert file analysis assistant. "
            "You have access to text snippets and descriptions of images extracted from a document. "
            "Answer the user's question primarily based on the provided Context. "
            "If the context mentions an image that helps explain the answer, explicitly mention 'Relevant Image Found: [Caption]'. "
            "Do not hallucinate facts not present in the context."
        )
        
        final_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"
        
        # 4. Generate
        answer = await self.llm.generate(final_prompt, system_prompt=system_prompt)
        
        return {
            "answer": answer,
            "retrieval_stats": {
                "text_count": len(texts),
                "image_count": len(images)
            },
            "sources": [t['doc_id'] for t in texts] + [i['doc_id'] for i in images],
            "images": images
        }
