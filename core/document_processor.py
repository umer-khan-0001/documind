"""
Document Processor Module

Handles multimodal document parsing including PDFs, images, and mixed content.
Supports text extraction, table detection, and image captioning.
"""

import os
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image
from unstructured.documents.elements import (
    Title, NarrativeText, Table, Image, ListItem
)
from PIL import Image as PILImage
import pytesseract

from utils.chunking import SemanticChunker
from utils.preprocessing import TextPreprocessor
from core.vision_parser import VisionParser


@dataclass
class DocumentChunk:
    """Represents a chunk of document content."""
    id: str
    content: str
    chunk_type: str  # text, table, image_caption
    page_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class ProcessedDocument:
    """Represents a fully processed document."""
    id: str
    filename: str
    file_path: str
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def num_pages(self) -> int:
        return self.metadata.get("num_pages", 0)
    
    @property
    def num_chunks(self) -> int:
        return len(self.chunks)


class DocumentProcessor:
    """
    Main document processing class.
    
    Handles the extraction and processing of various document types
    including PDFs, images, and mixed-content documents.
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        enable_ocr: bool = True,
        enable_vision: bool = True
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Target size for text chunks
            chunk_overlap: Overlap between consecutive chunks
            enable_ocr: Whether to enable OCR for scanned documents
            enable_vision: Whether to enable vision model for images
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_ocr = enable_ocr
        self.enable_vision = enable_vision
        
        self.chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.preprocessor = TextPreprocessor()
        self.vision_parser = VisionParser() if enable_vision else None
        
    def process(self, file_path: str) -> ProcessedDocument:
        """
        Process a document file.
        
        Args:
            file_path: Path to the document
            
        Returns:
            ProcessedDocument: Fully processed document with chunks
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
            
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Extract elements based on file type
        if file_path.suffix.lower() == '.pdf':
            elements = self._process_pdf(file_path)
        else:
            elements = self._process_image(file_path)
        
        # Convert elements to chunks
        chunks = self._create_chunks(elements, doc_id)
        
        # Build metadata
        metadata = self._extract_metadata(file_path, elements)
        
        return ProcessedDocument(
            id=doc_id,
            filename=file_path.name,
            file_path=str(file_path),
            chunks=chunks,
            metadata=metadata
        )
    
    def _process_pdf(self, file_path: Path) -> List[Any]:
        """Process a PDF document."""
        elements = partition_pdf(
            filename=str(file_path),
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=True,
            include_page_breaks=True,
            languages=["eng"],
            ocr_languages="eng" if self.enable_ocr else None
        )
        
        return elements
    
    def _process_image(self, file_path: Path) -> List[Any]:
        """Process an image file."""
        elements = partition_image(
            filename=str(file_path),
            strategy="hi_res",
            infer_table_structure=True,
            languages=["eng"]
        )
        
        # If vision model is enabled, get additional context
        if self.vision_parser:
            image_analysis = self.vision_parser.analyze(str(file_path))
            if image_analysis:
                elements.append(image_analysis)
        
        return elements
    
    def _create_chunks(
        self, 
        elements: List[Any], 
        doc_id: str
    ) -> List[DocumentChunk]:
        """Convert extracted elements into document chunks."""
        chunks = []
        current_page = 1
        text_buffer = []
        
        for element in elements:
            # Update page number
            if hasattr(element, 'metadata') and hasattr(element.metadata, 'page_number'):
                current_page = element.metadata.page_number
            
            # Handle different element types
            if isinstance(element, (Title, NarrativeText, ListItem)):
                text_buffer.append(str(element))
                
            elif isinstance(element, Table):
                # Flush text buffer first
                if text_buffer:
                    chunks.extend(
                        self._chunk_text(text_buffer, doc_id, current_page, "text")
                    )
                    text_buffer = []
                
                # Add table as single chunk
                chunks.append(DocumentChunk(
                    id=f"{doc_id}_table_{len(chunks)}",
                    content=str(element),
                    chunk_type="table",
                    page_number=current_page,
                    metadata={"element_type": "table"}
                ))
                
            elif isinstance(element, Image):
                # Process image with vision model if available
                if self.vision_parser:
                    caption = self.vision_parser.caption(element)
                    chunks.append(DocumentChunk(
                        id=f"{doc_id}_img_{len(chunks)}",
                        content=caption,
                        chunk_type="image_caption",
                        page_number=current_page,
                        metadata={"element_type": "image"}
                    ))
        
        # Flush remaining text
        if text_buffer:
            chunks.extend(
                self._chunk_text(text_buffer, doc_id, current_page, "text")
            )
        
        return chunks
    
    def _chunk_text(
        self, 
        texts: List[str], 
        doc_id: str, 
        page: int,
        chunk_type: str
    ) -> List[DocumentChunk]:
        """Chunk text content semantically."""
        combined_text = "\n".join(texts)
        combined_text = self.preprocessor.clean(combined_text)
        
        text_chunks = self.chunker.chunk(combined_text)
        
        return [
            DocumentChunk(
                id=f"{doc_id}_chunk_{i}",
                content=chunk,
                chunk_type=chunk_type,
                page_number=page,
                metadata={}
            )
            for i, chunk in enumerate(text_chunks)
        ]
    
    def _extract_metadata(
        self, 
        file_path: Path, 
        elements: List[Any]
    ) -> Dict[str, Any]:
        """Extract document metadata."""
        # Count pages
        page_numbers = set()
        for element in elements:
            if hasattr(element, 'metadata') and hasattr(element.metadata, 'page_number'):
                page_numbers.add(element.metadata.page_number)
        
        return {
            "filename": file_path.name,
            "file_size": file_path.stat().st_size,
            "file_type": file_path.suffix,
            "num_pages": len(page_numbers) if page_numbers else 1,
            "num_elements": len(elements),
            "element_types": list(set(type(e).__name__ for e in elements))
        }
