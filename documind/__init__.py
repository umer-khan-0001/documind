"""
DocuMind - Multimodal RAG Document Q&A System
"""

from core.document_processor import DocumentProcessor
from core.retriever import HybridRetriever
from core.llm_chain import LLMChain
from core.embeddings import EmbeddingManager

__version__ = "1.0.0"
__author__ = "Umer Ahmed"


class DocuMind:
    """
    Main interface for the DocuMind RAG system.
    
    Provides a unified API for document processing, indexing, and Q&A.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize DocuMind with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.processor = DocumentProcessor()
        self.embeddings = EmbeddingManager()
        self.retriever = HybridRetriever(self.embeddings)
        self.llm_chain = LLMChain()
        self.documents = []
        
    def load_document(self, file_path: str) -> dict:
        """
        Load and process a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            dict: Processing result with document metadata
        """
        # Process the document
        processed = self.processor.process(file_path)
        
        # Generate embeddings and index
        self.retriever.index_document(processed)
        
        self.documents.append(processed)
        
        return {
            "status": "success",
            "document_id": processed.id,
            "num_chunks": len(processed.chunks),
            "metadata": processed.metadata
        }
    
    def query(self, question: str, top_k: int = 5) -> "QueryResponse":
        """
        Query the document collection.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            QueryResponse: Answer with sources
        """
        # Retrieve relevant chunks
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        
        # Generate answer with LLM
        response = self.llm_chain.generate(
            question=question,
            context=retrieved.chunks,
            sources=retrieved.sources
        )
        
        return response
    
    def clear(self):
        """Clear all indexed documents."""
        self.retriever.clear()
        self.documents = []


class QueryResponse:
    """Response object for document queries."""
    
    def __init__(self, answer: str, sources: list, confidence: float):
        self.answer = answer
        self.sources = sources
        self.confidence = confidence
        
    def __repr__(self):
        return f"QueryResponse(answer='{self.answer[:50]}...', sources={len(self.sources)})"
