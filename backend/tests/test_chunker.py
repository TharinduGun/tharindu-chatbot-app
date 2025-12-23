import pytest
import sys
from pathlib import Path

# Adjust path to import app
sys.path.append(str(Path(__file__).parent.parent))

from app.services import chunker
from app.models.schema import SectionNode, ParagraphBlock

def test_create_chunks_basic():
    # Setup Data
    # Section 1
    s1 = SectionNode(
        section_id="s1",
        doc_id="d1", title="Intro", level=1, page_start=1, page_end=1, block_ids=["b1", "b2"]
    )
    
    # Blocks
    b1 = ParagraphBlock(
        block_id="b1", content="Hello world. ", page_no=1, section_id="s1", element_type="paragraph", doc_id="d1"
    )
    b2 = ParagraphBlock(
        block_id="b2", content="This is a test.", page_no=1, section_id="s1", element_type="paragraph", doc_id="d1"
    )
    
    chunks = chunker.create_chunks([s1], [b1, b2])
    
    assert len(chunks) == 1
    assert chunks[0].content.strip() == "Hello world. \n\nThis is a test.".strip()
    assert chunks[0].section_id == "s1"
    
def test_create_chunks_image_linking():
    # Test if image IDs are carried over
    s1 = SectionNode(section_id="s1", doc_id="d1", title="ImgSec", level=1, page_start=1, page_end=1, block_ids=["b3"])
    
    b3 = ParagraphBlock(
        block_id="b3", content="Image below.", page_no=1, section_id="s1", image_ids=["img1"], element_type="paragraph", doc_id="d1"
    )
    
    chunks = chunker.create_chunks([s1], [b3])
    
    assert len(chunks) == 1
    assert "img1" in chunks[0].image_ids 
