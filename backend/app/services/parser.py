import logging
from pathlib import Path
from docling.document_converter import DocumentConverter
from app.models.schema import DocumentRecord, SectionNode, ParagraphBlock, ImageAsset
from app.services import storage, chunker
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_document(doc_id: str, file_path: str):
    """
    Background task to parse the PDF, extract structure/images,
    chunk the content, and save everything.
    """
    # Create directory immediately to signal start
    status_dir = storage.PROCESSED_DIR / doc_id
    status_dir.mkdir(parents=True, exist_ok=True)
    status_file = status_dir / "status.txt"
    
    def update_status(msg):
        try:
            with open(status_file, "a") as f:
                f.write(f"{datetime.now()}: {msg}\n")
        except:
            pass

    update_status("Task started. Initializing DocumentConverter...")
    logger.info(f"Starting processing for doc_id: {doc_id} file: {file_path}")
    
    try:
        # 1. Parse with Docling
        # Configure pipeline to ensure images are extracted (if applicable for specific format options)
        # For default PDF, Docling does extract common elements.
        converter = DocumentConverter()
        update_status("Converter initialized. Starting conversion (this may take time for model download)...")
        
        result = converter.convert(file_path)
        doc = result.document
        
        # 2. Extract Images & Build Assets
        # Docling stores images in doc.pictures or similar depending on version, 
        # but typically elements refer to images.
        # We will collect them during traversal or via specific attributes if available.
        # For this implementation we'll scan the elements.
        
        image_assets = []
        
        # 3. Build Hierarchy (Sections & Paragraphs)
        
        sections = []
        blocks = []
        
        # Create a root section
        # doc.pages is a dict in recent Docling versions
        num_pages = len(doc.pages) if doc.pages else 1
        
        root_section = SectionNode(
            doc_id=doc_id,
            title="Root",
            level=0,
            page_start=1,
            page_end=num_pages,
            block_ids=[]
        )
        sections.append(root_section)
        
        # Stack to keep track of current sections: [root, chapter, section, ...]
        section_stack = [root_section]
        
        # Helper to process an item
        def process_item(item, current_stack):
            nonlocal blocks, image_assets
            
            # Identify item type (Checking typically available attributes)
            # This is a general "best effort" mapping based on standard Docling outputs
            
            text_content = ""
            label = getattr(item, "label", "").lower()
            text = getattr(item, "text", "")
            
            if not text and not label == "picture":
                return # Skip empty
                
            # Handle Section Headers
            if "header" in label or "heading" in label:
                # Determine level (approximate)
                level = 1 
                if "sub" in label: level = 2
                
                # Pop stack until we find a parent with level < new_level
                while len(current_stack) > 1 and current_stack[-1].level >= level:
                    current_stack.pop()
                
                parent = current_stack[-1]
                new_section = SectionNode(
                    doc_id=doc_id,
                    title=text[:200], # Trucate
                    level=level,
                    page_start=item.prov[0].page_no if hasattr(item, "prov") else 1,
                    page_end=item.prov[0].page_no if hasattr(item, "prov") else 1,
                    parent_section_id=parent.section_id,
                    block_ids=[]
                )
                sections.append(new_section)
                parent.child_section_ids.append(new_section.section_id)
                current_stack.append(new_section)
                return

            # Handle Images/Figures
            if label == "picture" or label == "figure":
                image_id = str(uuid.uuid4())
                # In a real scenario, we would save the image bytes here.
                # Assuming item has a 'image' attribute which is a PIL Image
                saved_path = ""
                if hasattr(item, "image") and item.image:
                    img_filename = f"{image_id}.png"
                    img_dir = storage.get_images_dir(doc_id)
                    item.image.save(img_dir / img_filename)
                    saved_path = str(img_dir / img_filename)
                
                bbox = [] 
                if hasattr(item, "prov") and item.prov:
                     # prov usually is list of dicts or objects with bbox
                     # simplified extraction
                     pass

                img_asset = ImageAsset(
                    image_id=image_id,
                    doc_id=doc_id,
                    page_no=item.prov[0].page_no if hasattr(item, "prov") else 1,
                    file_path=saved_path,
                    bbox=bbox,
                    caption_raw=text # often caption is attached or separate
                )
                image_assets.append(img_asset)
                
                # Also create a block reference for it
                blk = ParagraphBlock(
                    doc_id=doc_id,
                    page_no=img_asset.page_no,
                    section_id=current_stack[-1].section_id,
                    element_type="image",
                    content=f"[IMAGE: {text}]",
                    image_ids=[image_id]
                )
                blocks.append(blk)
                current_stack[-1].block_ids.append(blk.block_id)
                return

            # Handle Paragraphs/Lists/Tables
            element_type = "paragraph"
            if "list" in label: element_type = "list"
            elif "table" in label: element_type = "table"
            
            blk = ParagraphBlock(
                doc_id=doc_id,
                page_no=item.prov[0].page_no if hasattr(item, "prov") else 1,
                section_id=current_stack[-1].section_id,
                element_type=element_type,
                content=text,
                image_ids=[]
            )
            blocks.append(blk)
            current_stack[-1].block_ids.append(blk.block_id)


        # Traverse the body
        update_status("Traversing document structure...")
        
        # DEBUG: Log document structure
        update_status(f"Doc keys/attrs: {dir(doc)}")
        
        has_children = hasattr(doc.body, "children") if hasattr(doc, "body") else False
        has_iterate = hasattr(doc, "iterate_items")
        update_status(f"Structure checks: body.children={has_children}, iterate_items={has_iterate}")
        
        if hasattr(doc, "iterate_items"): # Preferred flat iterator
            update_status("Using iterate_items()")
            for item, level in doc.iterate_items():
                process_item(item, section_stack) 
                
        elif hasattr(doc.body, "children") if hasattr(doc, "body") else False: # Tree structure fallback
             update_status("Using Tree Traversal")
             def traverse_tree(node, stack):
                 # Process node itself if it has content
                 # DEBUG: Log first few nodes
                 if len(blocks) < 5:
                     try:
                         lbl = getattr(node, "label", "N/A")
                         txt = getattr(node, "text", "")[:20]
                         update_status(f"Node: label={lbl} text={txt} type={type(node)}")
                     except: pass
                     
                 process_item(node, stack)
                 # Recurse
                 if hasattr(node, "children"):
                     for child in node.children:
                         traverse_tree(child, stack)
             
             traverse_tree(doc.body, section_stack)

        else:
            # Fallback for simple text iteration
            update_status("Using doc.texts() fallback")
            for item in doc.texts():
                 process_item(item, section_stack)
        
        # 4. Create Fine Chunks
        update_status("Structure built. Creating fine-grained chunks...")
        fine_chunks = chunker.create_chunks(sections, blocks)
        
        # 5. Save Data
        data = {
            "record": {
                "doc_id": doc_id,
                "filename": Path(file_path).name,
                "num_pages": len(doc.pages) if hasattr(doc, 'pages') else 0,
                "processed_at": datetime.now()
            },
            "sections": [s.dict() for s in sections],
            "blocks": [b.dict() for b in blocks],
            "chunks": [c.dict() for c in fine_chunks],
            "images": [i.dict() for i in image_assets]
        }
        
        storage.save_processed_data(doc_id, data)
        logger.info(f"Processing complete for {doc_id}")
        
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {e}", exc_info=True)
        # Write error to file for debugging
        error_file = storage.PROCESSED_DIR / doc_id / "error.txt"
        # Ensure dir exists in case it failed early
        if not error_file.parent.exists():
            error_file.parent.mkdir(parents=True, exist_ok=True)
        with open(error_file, "w") as f:
            f.write(str(e))
