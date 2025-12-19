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

# --- Registry & Caching ---

REGISTRY_FILE = Path("data/document_registry.json")

def _load_registry() -> dict:
    if not REGISTRY_FILE.exists():
        return {}
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_registry(registry: dict):
    """Saves registry atomically to avoid corruption during concurrent writes."""
    temp_file = REGISTRY_FILE.with_suffix(".tmp")
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        # Atomic rename (replace)
        temp_file.replace(REGISTRY_FILE)
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise e

def calculate_content_hash(file: UploadFile) -> str:
    """Calculates SHA256 hash of the uploaded file content."""
    import hashlib
    sha256_hash = hashlib.sha256()
    file.file.seek(0)
    # Read in chunks to avoid memory issues
    for byte_block in iter(lambda: file.file.read(4096), b""):
        sha256_hash.update(byte_block)
    file.file.seek(0) # Reset cursor
    return sha256_hash.hexdigest()

def get_cached_document(file_hash: str) -> dict | None:
    """Checks registry for existing document with this hash."""
    registry = _load_registry()
    entry = registry.get(file_hash)
    if entry and entry.get("status") == "completed":
        return entry
    return None

def add_to_registry(doc_id: str, file_hash: str, filename: str):
    """Registers a new document."""
    registry = _load_registry()
    registry[file_hash] = {
        "doc_id": doc_id,
        "filename": filename,
        "status": "processing",
        # We can store extra metadata if useful for quick retrieval
    }
    _save_registry(registry)

def update_registry_status(doc_id: str, status: str):
    """Updates the status of a document in the registry."""
    registry = _load_registry()
    # Find entry by doc_id (since we lookup by hash usually, this is less efficient but fine for small scale)
    # Ideally we'd pass hash too, but let's just search values
    for file_hash, data in registry.items():
        if data["doc_id"] == doc_id:
            registry[file_hash]["status"] = status
            _save_registry(registry)
            return
