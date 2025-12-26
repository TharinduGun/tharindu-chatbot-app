import os
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

# Mock dependencies before importing the module under test to avoid loading heavy models
with patch('app.services.multimodal.SiglipProcessor'), \
     patch('app.services.multimodal.SiglipModel'), \
     patch('app.services.multimodal.AutoTokenizer'), \
     patch('app.services.multimodal.AutoModel'), \
     patch('app.services.multimodal.BlipProcessor'), \
     patch('app.services.multimodal.BlipForConditionalGeneration'):
    from app.services.multimodal import MultimodalPipeline

@pytest.fixture
def mock_pipeline():
    with patch('app.services.multimodal.SiglipProcessor'), \
         patch('app.services.multimodal.SiglipModel'), \
         patch('app.services.multimodal.AutoTokenizer'), \
         patch('app.services.multimodal.AutoModel'):
        pipeline = MultimodalPipeline()
        # Setup standard mocks for models
        pipeline.bge_model = MagicMock()
        pipeline.bge_tokenizer = MagicMock()
        pipeline.siglip_model = MagicMock()
        pipeline.siglip_processor = MagicMock()
        pipeline.blip_model = MagicMock()
        pipeline.blip_processor = MagicMock()
        return pipeline

def test_validate_caption(mock_pipeline):
    assert mock_pipeline.validate_caption("A valid long caption description.") is True
    assert mock_pipeline.validate_caption("") is False
    assert mock_pipeline.validate_caption("Short") is False # split length < 3
    assert mock_pipeline.validate_caption("Figure 1") is False # Figure generic and short

def test_initialization(mock_pipeline):
    assert mock_pipeline.device is not None
    assert mock_pipeline.siglip_model is not None
    assert mock_pipeline.bge_model is not None

def test_get_bge_embedding(mock_pipeline):
    # Mock return values
    mock_output = MagicMock()
    # Mocking last_hidden_state: [batch, seq, hidden] -> [1, 10, 768]
    mock_output.last_hidden_state = pytest.importorskip("torch").randn(1, 10, 768)
    mock_pipeline.bge_model.return_value = mock_output
    
    embedding = mock_pipeline.get_bge_embedding("test text")
    
    assert isinstance(embedding, list)
    assert len(embedding) == 768 # Default size from the random tensor

def test_get_siglip_image_embedding(mock_pipeline):
    # Mock Image.open to avoid needing a real file
    with patch("app.services.multimodal.Image.open") as mock_open_img:
        mock_open_img.return_value.convert.return_value = MagicMock()
        
        # Mock SigLIP model output
        mock_pipeline.siglip_model.get_image_features.return_value = pytest.importorskip("torch").randn(1, 1024) # Example dim
        
        embedding = mock_pipeline.get_siglip_image_embedding("dummy/path.jpg")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 1024

def test_get_siglip_text_embedding(mock_pipeline):
    # Mock SigLIP model output
    mock_pipeline.siglip_model.get_text_features.return_value = pytest.importorskip("torch").randn(1, 1024)
    
    embedding = mock_pipeline.get_siglip_text_embedding("dummy text")
    
    assert isinstance(embedding, list)
    assert len(embedding) == 1024

def test_run_flow(mock_pipeline):
    doc_id = "test_doc_id"
    mock_data = {
        "images": [
            {"image_id": "img1", "file_path": "path/img1.png", "page_no": 1},
            {"image_id": "img2", "file_path": "path/img2.png", "page_no": 2} # Missing file case
        ],
        "chunks": [
            {"chunk_id": "chk1", "content": "Chunk content 1", "page_no": 1, "section_id": "sec1"},
            {"chunk_id": "chk2", "content": "Chunk content 2", "page_no": 3, "section_id": "sec2"}
        ],
        "sections": [{"section_id": "sec1", "title": "Section 1"}],
        "blocks": []
    }
    
    # Mock Storage and File Ops
    with patch("app.services.multimodal.storage") as mock_storage, \
         patch("builtins.open", mock_open(read_data=json.dumps(mock_data))), \
         patch("json.load", return_value=mock_data), \
         patch("os.path.exists") as mock_os_path_exists:
         
        # Setup metadata path existence via Path objects
        # We need to ensure metadata_path.exists() returns True
        mock_doc_dir = MagicMock()
        mock_metadata_path = MagicMock()
        mock_metadata_path.exists.return_value = True
        
        # storage.PROCESSED_DIR / doc_id -> mock_doc_dir
        mock_storage.PROCESSED_DIR.__truediv__.return_value = mock_doc_dir
        # mock_doc_dir / "metadata.json" -> mock_metadata_path
        mock_doc_dir.__truediv__.return_value = mock_metadata_path
        
        # Setup image existence
        def exists_side_effect(path):
            if "img1" in str(path): return True
            if "img2" in str(path): return False
            return False
            
        mock_os_path_exists.side_effect = exists_side_effect
        
        # Mock embeddings to be deterministic arrays for dot product check
        # img1 embedding
        mock_pipeline.get_siglip_image_embedding = MagicMock(return_value=[1.0, 0.0])
        # chunk1 embedding (match img1)
        mock_pipeline.get_siglip_text_embedding = MagicMock(side_effect=[
             [1.0, 0.0], # chunk1 match
             [0.0, 1.0]  # chunk2 no match
        ])
        # bge embedding
        mock_pipeline.get_bge_embedding = MagicMock(return_value=[0.1]*768)
        
        mock_pipeline.run(doc_id)
        
        # Verify save called
        assert mock_storage.save_processed_data.called
        saved_data = mock_storage.save_processed_data.call_args[0][1]
        
        # Check if img1 got linked
        img1 = saved_data["images"][0]
        assert "linked_chunk_id" in img1
        assert img1["linked_chunk_id"] == "chk1"
        assert img1["match_score"] > 0.9 # Should be high due to perfect match
