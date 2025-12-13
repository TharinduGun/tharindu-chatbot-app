from typing import List
from app.models.schema import SectionNode, ParagraphBlock, FineChunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(sections: List[SectionNode], blocks: List[ParagraphBlock]) -> List[FineChunk]:
    """
    Generates fine-grained chunks from the blocks.
    Strategy: Group blocks by section, concatenate text, then split.
    """
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    
    # Map blocks by ID for easy access
    blocks_map = {b.block_id: b for b in blocks}
    
    for section in sections:
        if not section.block_ids:
            continue
            
        # 1. Aggregation with Offset Tracking
        # We need to build the text but know which block contributed which part
        # so we can map chunks back to blocks (and thus images).
        
        section_text = ""
        # List of (start, end, block_id, image_ids)
        block_spans = [] 
        
        current_offset = 0
        
        for bid in section.block_ids:
            if bid in blocks_map:
                blk = blocks_map[bid]
                
                # If it's an image block, we might effectively be skipping adding text 
                # or adding a placeholder. 
                # User wants "linked images to text".
                # If we add a placeholder like [IMAGE], the chunker might include it.
                
                content_to_add = blk.content + "\n\n"
                start = current_offset
                end = current_offset + len(content_to_add)
                
                block_spans.append({
                    "start": start,
                    "end": end,
                    "block_id": bid,
                    "image_ids": blk.image_ids,
                    "page_no": blk.page_no
                })
                
                section_text += content_to_add
                current_offset = end
        
        if not section_text.strip():
            continue
            
        # 2. Split into fine chunks
        texts = text_splitter.create_documents([section_text])        
        # Note: create_documents returns Document objects, but we need correct offsets.
        # RecursiveCharacterTextSplitter doesn't return offsets easily unless we use other methods.
        # However, we can use `split_text` and try to match, OR use a more advanced approach.
        # For robustness in this phase, we will use a heuristic:
        # If a chunk is contained within the section text, we find its approx location.
        # Since we just built section_text, we can search.
        
        # Better approach: Iterate over the split texts and find them in section_text.
        # Since order is preserved...
        
        search_start_index = 0
        for doc_chunk in texts:
            chunk_content = doc_chunk.page_content
            
            # Find exact start position of this chunk in the section_text
            # We start searching from search_start_index to handle duplicate phrases correctly
            found_start = section_text.find(chunk_content, search_start_index)
            if found_start == -1:
                # Should not happen if splitter works on same text
                continue
                
            found_end = found_start + len(chunk_content)
            search_start_index = found_start + 1 # Advance slightly (or by length)
            
            # 3. Resolve Relations
            # Which blocks overlap with this [found_start, found_end]?
            
            chunk_block_ids = set()
            chunk_image_ids = set()
            chunk_pages = set()
            
            for span in block_spans:
                # Check overlap
                # Span: [s, e], Chunk: [cs, ce]
                # Overlap if s < ce and e > cs
                if span["start"] < found_end and span["end"] > found_start:
                    chunk_block_ids.add(span["block_id"])
                    if span["image_ids"]:
                        chunk_image_ids.update(span["image_ids"])
                    chunk_pages.add(span["page_no"])
            
            # Also, link any images that are on the SAME PAGE as this chunk (Proximity)
            # This requires scanning ALL blocks or keeping a page index.
            # For efficiency, we will stick to direct structural link + exact span overlap for now,
            # but user asked for "page proximity". 
            # Let's add all images from the MAIN page of this chunk.
            
            main_page = min(chunk_pages) if chunk_pages else section.page_start
            
            # Simple Page Proximity: Check all blocks in this SECTION that are on `main_page` and have images
            for span in block_spans:
                if span["page_no"] == main_page and span["image_ids"]:
                     chunk_image_ids.update(span["image_ids"])

            chunk = FineChunk(
                doc_id=section.doc_id,
                page_no=main_page,
                section_id=section.section_id,
                block_ids=list(chunk_block_ids),
                content=chunk_content,
                image_ids=list(chunk_image_ids)
            )
            chunks.append(chunk)
            
    return chunks
