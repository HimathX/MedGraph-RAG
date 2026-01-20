import uuid
from typing import List, Optional
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    """Smallest unit of text for embedding and retrieval."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    section_id: str
    metadata: dict = Field(default_factory=dict)

class Section(BaseModel):
    """Hierarchical section of a document (e.g. Header 1, Header 2)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str  # The full text content of this section (including chunks)
    level: int  # 1 for #, 2 for ##, etc.
    document_id: str
    parent_id: Optional[str] = None
    chunks: List[Chunk] = Field(default_factory=list)

class Document(BaseModel):
    """Represents a full PDF document."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    source_path: str
    sections: List[Section] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
