from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from app.services import storage, multimodal
from app.services import parser  # We will implement this next
from app.models.schema import DocumentRecord
import uuid
from datetime import datetime
from typing import List, Dict, Any
import os
import json

router = APIRouter()

@router.post("/upload", response_model=DocumentRecord)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 1. Calculate Hash & Check Cache
    try:
        file_hash = storage.calculate_content_hash(file)
        cached_doc = storage.get_cached_document(file_hash)
        
        if cached_doc:
            try:
                # Cache HIT: Return existing record structure
                # Trigger Multimodal Pipeline (Phase 4)
                pipeline = multimodal.MultimodalPipeline()
                background_tasks.add_task(pipeline.run, cached_doc["doc_id"])

                import json
                meta_path = storage.PROCESSED_DIR / cached_doc["doc_id"] / "metadata.json"
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        record_data = data.get("record", {})
                        if "uploaded_at" not in record_data:
                            record_data["uploaded_at"] = datetime.now()
                        return DocumentRecord(**record_data)
            except Exception as e:
                print(f"Error loading cached metadata: {e}")
                pass 
                
    except Exception as e:
        # If hashing fails, log it but maybe precede (or fail). 
        # Let's assume we proceed as if new file.
        print(f"Hashing failed: {e}")

    # 2. Save the file (Cache MISS)
    try:
        file_path = storage.save_upload_file(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # 3. Create initial document record
    doc_id = str(uuid.uuid4())
    doc_record = DocumentRecord(
        doc_id=doc_id,
        filename=file.filename,
        num_pages=0, # Will be updated after parsing
        uploaded_at=datetime.now()
    )
    
    # 4. Register in Cache
    # We need the hash calculated earlier. If it failed, we might skip registry or recalc (but stream invalid).
    # Assuming hash calculation worked if we are here (or we'd have returned).
    # We should re-calculate if we didn't get it above? No, storage.calculate_content_hash resets cursor.
    if 'file_hash' in locals():
         storage.add_to_registry(doc_id, file_hash, file.filename)

    # 5. Trigger background processing
    background_tasks.add_task(parser.process_document, doc_id, file_path)

    return doc_record

@router.post("/{doc_id}/process-multimodal")
async def process_multimodal(doc_id: str, background_tasks: BackgroundTasks):
    """
    Triggers Phase 4 Multimodal Embedding Pipeline for an existing document.
    """
    # Verify doc exists
    if not (storage.PROCESSED_DIR / doc_id / "metadata.json").exists():
        raise HTTPException(status_code=404, detail="Document not found or not processed yet.")
    
    # Run in background (it is heavy)
    pipeline = multimodal.MultimodalPipeline()
    background_tasks.add_task(pipeline.run, doc_id)
    
    return {"message": "Multimodal processing started", "doc_id": doc_id}

@router.get("/list", response_model=List[Dict[str, Any]])
async def list_documents():
    """
    Lists all processed documents available in the system.
    """
    try:
        processed_dir = storage.PROCESSED_DIR
        if not processed_dir.exists():
            return []

        documents = []
        # Iterate over directories in processed data folder
        for doc_id in os.listdir(processed_dir):
            doc_path = processed_dir / doc_id
            if doc_path.is_dir():
                # Default info
                doc_info = {
                    "id": doc_id,
                    "name": doc_id 
                }
                
                # Check for metadata to get better name/date
                meta_path = doc_path / "metadata.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            # Try to find filename in record or metadata
                            record = data.get("record", {})
                            if "filename" in record:
                                doc_info["name"] = record["filename"]
                    except:
                        pass
                
                documents.append(doc_info)
        
        return documents

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
