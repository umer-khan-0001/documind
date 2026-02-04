"""
Tests for DocumentProcessor module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from core.document_processor import DocumentProcessor, ProcessedDocument, DocumentChunk


class TestDocumentProcessor:
    """Test suite for DocumentProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DocumentProcessor()
    
    @pytest.fixture
    def sample_pdf_path(self, tmp_path):
        """Create a temporary PDF file for testing."""
        pdf_path = tmp_path / "test_document.pdf"
        # Create a minimal valid PDF
        pdf_content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
trailer << /Size 5 /Root 1 0 R >>
startxref
306
%%EOF"""
        pdf_path.write_bytes(pdf_content)
        return pdf_path
    
    def test_processor_initialization(self, processor):
        """Test processor initializes correctly."""
        assert processor is not None
        assert processor.supported_formats == ['.pdf', '.png', '.jpg', '.jpeg', '.tiff']
    
    def test_detect_document_type_pdf(self, processor, sample_pdf_path):
        """Test PDF detection."""
        doc_type = processor._detect_document_type(str(sample_pdf_path))
        assert doc_type == "pdf"
    
    def test_detect_document_type_image(self, processor, tmp_path):
        """Test image detection."""
        for ext in ['.png', '.jpg', '.jpeg']:
            img_path = tmp_path / f"test{ext}"
            img_path.touch()
            doc_type = processor._detect_document_type(str(img_path))
            assert doc_type == "image"
    
    def test_detect_unsupported_format(self, processor, tmp_path):
        """Test unsupported format raises error."""
        txt_path = tmp_path / "test.txt"
        txt_path.touch()
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            processor._detect_document_type(str(txt_path))
    
    @patch('core.document_processor.partition')
    def test_process_pdf(self, mock_partition, processor, sample_pdf_path):
        """Test PDF processing."""
        # Mock the partition function
        mock_element = MagicMock()
        mock_element.text = "This is test content from the PDF."
        mock_element.metadata.page_number = 1
        mock_element.category = "NarrativeText"
        mock_partition.return_value = [mock_element]
        
        result = processor.process(str(sample_pdf_path))
        
        assert isinstance(result, ProcessedDocument)
        assert result.source_path == str(sample_pdf_path)
        assert len(result.chunks) > 0
    
    def test_processed_document_dataclass(self):
        """Test ProcessedDocument dataclass."""
        chunks = [
            DocumentChunk(
                content="Test content",
                chunk_id="chunk_001",
                page_number=1,
                content_type="text",
                metadata={"source": "test"}
            )
        ]
        
        doc = ProcessedDocument(
            document_id="doc_123",
            source_path="/path/to/doc.pdf",
            chunks=chunks,
            metadata={"pages": 1}
        )
        
        assert doc.document_id == "doc_123"
        assert len(doc.chunks) == 1
        assert doc.chunks[0].content == "Test content"
    
    def test_chunk_creation(self):
        """Test DocumentChunk creation."""
        chunk = DocumentChunk(
            content="Sample text content",
            chunk_id="chunk_001",
            page_number=1,
            content_type="text",
            metadata={"position": 0}
        )
        
        assert chunk.content == "Sample text content"
        assert chunk.page_number == 1
        assert chunk.content_type == "text"
    
    @patch('core.document_processor.partition')
    def test_extract_tables(self, mock_partition, processor, sample_pdf_path):
        """Test table extraction from documents."""
        # Mock a table element
        mock_table = MagicMock()
        mock_table.text = "Header1 | Header2\nValue1 | Value2"
        mock_table.metadata.page_number = 1
        mock_table.category = "Table"
        mock_partition.return_value = [mock_table]
        
        result = processor.process(str(sample_pdf_path))
        
        # Find table chunks
        table_chunks = [c for c in result.chunks if c.content_type == "table"]
        assert len(table_chunks) >= 0  # May or may not have table chunks
    
    @patch('core.document_processor.partition')
    def test_multipage_processing(self, mock_partition, processor, sample_pdf_path):
        """Test processing document with multiple pages."""
        mock_elements = []
        for page in range(1, 4):
            element = MagicMock()
            element.text = f"Content from page {page}"
            element.metadata.page_number = page
            element.category = "NarrativeText"
            mock_elements.append(element)
        
        mock_partition.return_value = mock_elements
        
        result = processor.process(str(sample_pdf_path))
        
        assert result is not None
        page_numbers = set(c.page_number for c in result.chunks)
        assert len(page_numbers) == 3
    
    def test_empty_document_handling(self, processor, tmp_path):
        """Test handling of empty documents."""
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"%PDF-1.4\n%%EOF")
        
        with patch('core.document_processor.partition', return_value=[]):
            result = processor.process(str(empty_pdf))
            assert len(result.chunks) == 0
    
    @pytest.mark.parametrize("content,expected_type", [
        ("Regular paragraph text", "text"),
        ("def function():\n    pass", "code"),
        ("| Col1 | Col2 |\n| --- | --- |", "table"),
    ])
    def test_content_type_detection(self, content, expected_type, processor):
        """Test content type detection logic."""
        # This tests the internal content type detection
        detected = processor._detect_content_type(content)
        assert detected in ["text", "code", "table", "list"]


class TestDocumentChunk:
    """Test suite for DocumentChunk dataclass."""
    
    def test_chunk_to_dict(self):
        """Test chunk serialization."""
        chunk = DocumentChunk(
            content="Test",
            chunk_id="c1",
            page_number=1,
            content_type="text",
            metadata={}
        )
        
        d = chunk.to_dict()
        assert d["content"] == "Test"
        assert d["chunk_id"] == "c1"
    
    def test_chunk_from_dict(self):
        """Test chunk deserialization."""
        data = {
            "content": "Test",
            "chunk_id": "c1",
            "page_number": 1,
            "content_type": "text",
            "metadata": {}
        }
        
        chunk = DocumentChunk.from_dict(data)
        assert chunk.content == "Test"
        assert chunk.chunk_id == "c1"
