import pytest
import shutil
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.append(str(Path(__file__).parent.parent))
from app.services import storage

@pytest.fixture
def mock_dirs(tmp_path):
    # Override storage global paths with tmp_path
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    
    with patch("app.services.storage.RAW_DIR", raw), \
         patch("app.services.storage.PROCESSED_DIR", processed), \
         patch("app.services.storage.REGISTRY_FILE", tmp_path / "registry.json"):
        yield raw, processed

def test_calculate_content_hash():
    # Helper to simulate file
    mock_file = MagicMock()
    mock_file.file.read.side_effect = [b"test data", b""]
    # We need to mock seek too or it will fail
    mock_file.file.seek = MagicMock()
    
    hash_val = storage.calculate_content_hash(mock_file)
    assert len(hash_val) == 64

def test_save_upload_file(mock_dirs):
    raw_dir, _ = mock_dirs
    
    # Create a real file to copy from? Or just mock the file object
    # storage.save_upload_file uses shutil.copyfileobj
    
    # Let's mock the upload_file object but verify it writes to the REAL tmp raw_dir
    mock_upload = MagicMock()
    mock_upload.filename = "test.pdf"
    
    # We can't easily mock shutil.copyfileobj to write to real disk unless we provide a real source stream.
    # Let's provide a BytesIO object
    from io import BytesIO
    mock_upload.file = BytesIO(b"file content")
    
    saved_path = storage.save_upload_file(mock_upload)
    
    assert Path(saved_path).exists()
    assert Path(saved_path).parent == raw_dir
    assert Path(saved_path).read_bytes() == b"file content"

def test_registry_operations(mock_dirs):
    # Since we redirected REGISTRY_FILE to tmp_path, we can test real JSON ops
    
    doc_id = "doc_123"
    f_hash = "abc123hash"
    
    # 1. Add
    storage.add_to_registry(doc_id, f_hash, "test.pdf")
    
    # check
    reg = storage._load_registry()
    assert f_hash in reg
    assert reg[f_hash]["status"] == "processing"
    
    # 2. Update
    storage.update_registry_status(doc_id, "completed")
    reg = storage._load_registry()
    assert reg[f_hash]["status"] == "completed"
    
    # 3. Get Cached
    cached = storage.get_cached_document(f_hash)
    assert cached["doc_id"] == doc_id
