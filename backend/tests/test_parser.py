import pytest
from unittest.mock import MagicMock, patch, ANY
import sys
from pathlib import Path

# Adjust path to import app
sys.path.append(str(Path(__file__).parent.parent))

from app.services import parser
from app.models.schema import FineChunk

@pytest.fixture
def mock_storage(tmp_path):
    with patch("app.services.parser.storage") as mock:
        # Use real tmp paths so open() works
        d = tmp_path / "processed"
        d.mkdir()
        mock.PROCESSED_DIR = d
        mock.get_images_dir.side_effect = lambda doc_id: d / doc_id / "images"
        yield mock

@pytest.fixture
def mock_chunker():
    with patch("app.services.parser.chunker") as mock:
        yield mock

@pytest.fixture
def mock_docling():
    with patch("app.services.parser.DocumentConverter") as mock_cls:
        yield mock_cls

def test_process_document_success(mock_storage, mock_chunker, mock_docling):
    # Setup Mocks
    mock_converter = mock_docling.return_value
    mock_document = MagicMock()
    mock_result = MagicMock()
    mock_result.document = mock_document
    mock_converter.convert.return_value = mock_result
    
    # Mock Document structure
    # iterate_items logic in parser looks for items with 'label' and 'text'
    # Let's create a header item and a paragraph item
    item_header = MagicMock()
    item_header.label = "header"
    item_header.text = "Introduction"
    item_header.prov = [MagicMock(page_no=1)]
    
    item_para = MagicMock()
    item_para.label = "paragraph"
    item_para.text = "This is a test document."
    item_para.prov = [MagicMock(page_no=1)]
    
    # Mock iterator
    mock_document.iterate_items.return_value = [
        (item_header, 0),
        (item_para, 0)
    ]
    mock_document.pages = {1: MagicMock()}
    
    # Mock Chunker return
    mock_chunker.create_chunks.return_value = [
        FineChunk(
            chunk_id="test_chunk",
            doc_id="test_doc",
            section_id="sec_1",
            content="This is a test chunk",
            page_no=1,
            image_ids=[]
        )
    ]
    
    # Execute Test
    doc_id = "test_doc_id"
    file_path = "dummy.pdf"
    
    parser.process_document(doc_id, file_path)
    
    # Assertions
    # 1. Check if Docling converted the file
    mock_converter.convert.assert_called_once_with(file_path)
    
    # 2. Check if Chunker was called
    mock_chunker.create_chunks.assert_called_once()
    
    # 3. Check if Data was saved
    mock_storage.save_processed_data.assert_called_once()
    args, _ = mock_storage.save_processed_data.call_args
    saved_doc_id, saved_data = args
    
    assert saved_doc_id == doc_id
    assert "sections" in saved_data
    assert "blocks" in saved_data
    assert "chunks" in saved_data
    assert len(saved_data["chunks"]) == 1
    
    # Verify extraction logic logic
    # We expect at least one section (Root) + likely one from header
    assert len(saved_data["sections"]) >= 1 
    
    # 4. Check status update
    mock_storage.update_registry_status.assert_called_with(doc_id, "completed")

def test_process_document_failure(mock_storage, mock_docling):
    # Simulate a crash in Docling
    mock_docling.return_value.convert.side_effect = Exception("Docling crashed")
    
    doc_id = "test_doc_bug"
    file_path = "bad.pdf"
    
    parser.process_document(doc_id, file_path)
    
    # Assert status updated to failed
    mock_storage.update_registry_status.assert_called_with(doc_id, "failed")
    
    # Assert error file was intended to be written (we can check if open was called since we implicitly mocked it via context manager or storage, but storage.PROCESSED_DIR / ... is mocked)
    # The parser uses `open(..., 'w')` internally for error file. Since we didn't patch `builtins.open`, it might fail if path is purely mock.
    # However, since we mocked storage.PROCESSED_DIR as a MagicMock, the `error_file.parent.exists` checks will pass/fail based on mock default.
    # Let's trust just logic for now: exception caught -> status=failed.
