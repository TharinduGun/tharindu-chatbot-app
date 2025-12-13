import os
import shutil
from fastapi import UploadFile
import uuid
import json
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def save_upload_file(upload_file: UploadFile) -> str:
    """Saves the uploaded file to data/raw and returns the absolute path."""
    file_id = str(uuid.uuid4())
    extension = os.path.splitext(upload_file.filename)[1]
    safe_filename = f"{file_id}{extension}"
    
    # Ensure raw directory exists (in case it was cleaned up)
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        
    file_path = RAW_DIR / safe_filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return str(file_path.resolve())

def save_processed_data(doc_id: str, data: dict):
    """Saves the processed document data to data/processed/{doc_id}/metadata.json"""
    doc_dir = PROCESSED_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    with open(doc_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

def get_images_dir(doc_id: str) -> Path:
    """Returns the directory for storing images for a specific document."""
    doc_dir = PROCESSED_DIR / doc_id / "images"
    doc_dir.mkdir(parents=True, exist_ok=True)
    return doc_dir
