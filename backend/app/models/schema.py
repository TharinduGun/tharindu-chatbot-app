from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime
import uuid

class ImageAsset(BaseModel):
    image_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    page_no: int
    file_path: str
    bbox: Optional[List[float]] = None  # [x, y, width, height] or similar
    caption_raw: Optional[str] = None
    caption_generated: Optional[str] = None
    caption_source: Optional[str] = None # 'pdf' or 'blip2'
    embedding_siglip_image: Optional[List[float]] = None
    embedding_siglip_caption: Optional[List[float]] = None
    linked_chunk_id: Optional[str] = None
    match_score: Optional[float] = None

class ParagraphBlock(BaseModel):
    block_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    page_no: int
    section_id: Optional[str] = None
    element_type: str  # paragraph, list, table, caption
    content: str
    image_ids: List[str] = []

class SectionNode(BaseModel):
    section_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    title: str
    level: int  # 1=chapter, 2=section, etc.
    page_start: int
    page_end: int
    parent_section_id: Optional[str] = None
    child_section_ids: List[str] = []
    block_ids: List[str] = []
    summary: Optional[str] = None

class FineChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    page_no: int
    section_id: Optional[str] = None
    block_ids: List[str] = []
    content: str
    image_ids: List[str] = []
    embedding_bge: Optional[List[float]] = None
    embedding_siglip: Optional[List[float]] = None
    linked_image_ids: List[str] = []

class DocumentRecord(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    title: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=datetime.now)
    num_pages: int
