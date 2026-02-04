"""
Semantic Chunking Utilities

Provides intelligent document chunking that preserves
context and semantic boundaries.
"""

from typing import List, Optional
import re
from dataclasses import dataclass


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    start_char: int
    end_char: int
    word_count: int
    sentence_count: int


class SemanticChunker:
    """
    Semantic-aware text chunker.
    
    Splits text into chunks while respecting sentence boundaries,
    paragraph structure, and semantic coherence.
    """
    
    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    # Paragraph patterns
    PARAGRAPH_BREAK = re.compile(r'\n\s*\n')
    
    # Section header patterns
    SECTION_HEADERS = re.compile(
        r'^(?:#{1,6}\s+|(?:\d+\.)+\s+|[A-Z][A-Z\s]+:)',
        re.MULTILINE
    )
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        respect_sentences: bool = True,
        respect_paragraphs: bool = True
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size to avoid tiny fragments
            respect_sentences: Whether to avoid splitting sentences
            respect_paragraphs: Whether to prefer paragraph boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sentences = respect_sentences
        self.respect_paragraphs = respect_paragraphs
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into semantic chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
        
        # First, split by paragraphs if respecting them
        if self.respect_paragraphs:
            paragraphs = self.PARAGRAPH_BREAK.split(text)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
        else:
            paragraphs = [text]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Check if paragraph fits in current chunk
            if len(current_chunk) + len(paragraph) + 1 <= self.chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
            else:
                # Need to handle this paragraph
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If paragraph itself is too large, split it
                if len(paragraph) > self.chunk_size:
                    sub_chunks = self._split_large_text(paragraph)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = paragraph
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Apply overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)
        
        # Filter out chunks that are too small
        chunks = [c for c in chunks if len(c) >= self.min_chunk_size]
        
        return chunks
    
    def _split_large_text(self, text: str) -> List[str]:
        """Split text that exceeds chunk size."""
        if self.respect_sentences:
            return self._split_by_sentences(text)
        else:
            return self._split_by_size(text)
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text respecting sentence boundaries."""
        sentences = self.SENTENCE_ENDINGS.split(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If single sentence is too long, force split
                if len(sentence) > self.chunk_size:
                    sub_chunks = self._split_by_size(sentence)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1]
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_size(self, text: str) -> List[str]:
        """Split text by size, trying to break at word boundaries."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to find a word boundary
            space_idx = text.rfind(' ', start, end)
            if space_idx > start:
                end = space_idx
            
            chunks.append(text[start:end].strip())
            start = end
        
        return chunks
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Apply overlap between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            # Get overlap from end of previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap:]
            
            # Find sentence boundary in overlap if possible
            if self.respect_sentences:
                sentence_end = overlap_text.rfind('. ')
                if sentence_end > 0:
                    overlap_text = overlap_text[sentence_end + 2:]
            
            # Prepend overlap to current chunk
            overlapped.append(overlap_text + " " + curr_chunk)
        
        return overlapped
    
    def chunk_with_metadata(self, text: str) -> List[tuple]:
        """
        Chunk text and return with metadata.
        
        Args:
            text: Input text
            
        Returns:
            List of (chunk_text, ChunkMetadata) tuples
        """
        chunks = self.chunk(text)
        result = []
        
        char_offset = 0
        for chunk in chunks:
            # Find actual position in original text
            start = text.find(chunk[:50], char_offset)
            if start == -1:
                start = char_offset
            
            metadata = ChunkMetadata(
                start_char=start,
                end_char=start + len(chunk),
                word_count=len(chunk.split()),
                sentence_count=len(self.SENTENCE_ENDINGS.findall(chunk)) + 1
            )
            
            result.append((chunk, metadata))
            char_offset = start + len(chunk) - self.chunk_overlap
        
        return result
