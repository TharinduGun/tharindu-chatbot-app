from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from app.services import storage
from app.services import parser  # We will implement this next
from app.models.schema import DocumentRecord
import uuid
from datetime import datetime

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
            # Cache HIT: Return existing record structure
            # We need to construct a DocumentRecord from the cached info. 
            # Note: The registry only has basic info. Ideally we would load the full metadata.
            # But for this return type, we need at least what's in DocumentRecord.
            # Let's attempt to load the processed metadata to get accurate info like num_pages.
            try:
                # Reconstruct path to metadata
                import json
                from pathlib import Path
                meta_path = storage.PROCESSED_DIR / cached_doc["doc_id"] / "metadata.json"
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        record_data = data.get("record", {})
                        # Convert string times back to datetime if needed, or rely on pydantic coercion (usually needs obj)
                        # The Pydantic model expects a datetime object for uploaded_at
                        if "uploaded_at" not in record_data:
                             record_data["uploaded_at"] = datetime.now() # Fallback
                        
                        return DocumentRecord(**record_data)
            except Exception:
                # If loading metadata fails, fall through to re-process or just return basic info?
                # For safety, if metadata is missing, maybe we should re-process.
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
