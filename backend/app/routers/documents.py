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

    # Save the file
    try:
        file_path = storage.save_upload_file(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Create initial document record
    doc_id = str(uuid.uuid4())
    doc_record = DocumentRecord(
        doc_id=doc_id,
        filename=file.filename,
        num_pages=0, # Will be updated after parsing
        uploaded_at=datetime.now()
    )

    # Trigger background processing
    background_tasks.add_task(parser.process_document, doc_id, file_path)

    return doc_record
