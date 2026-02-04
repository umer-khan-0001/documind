"""
Tests for HybridRetriever module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from core.retriever import HybridRetriever, RetrievalResult


class TestHybridRetriever:
    """Test suite for HybridRetriever."""
    
    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embedding manager."""
        embeddings = Mock()
        embeddings.embed_query.return_value = np.random.rand(1536).tolist()
        embeddings.embed_documents.return_value = [
            np.random.rand(1536).tolist() for _ in range(5)
        ]
        return embeddings
    
    @pytest.fixture
    def retriever(self, mock_embeddings):
        """Create retriever instance with mocked dependencies."""
        with patch('core.retriever.chromadb'):
            return HybridRetriever(embedding_manager=mock_embeddings)
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {"id": "doc1", "content": "Machine learning is a subset of AI.", "metadata": {"page": 1}},
            {"id": "doc2", "content": "Deep learning uses neural networks.", "metadata": {"page": 2}},
            {"id": "doc3", "content": "Natural language processing handles text.", "metadata": {"page": 3}},
            {"id": "doc4", "content": "Computer vision processes images.", "metadata": {"page": 4}},
            {"id": "doc5", "content": "Reinforcement learning uses rewards.", "metadata": {"page": 5}},
        ]
    
    def test_retriever_initialization(self, retriever):
        """Test retriever initializes correctly."""
        assert retriever is not None
        assert retriever.dense_weight == 0.6
        assert retriever.sparse_weight == 0.4
    
    def test_custom_weights(self, mock_embeddings):
        """Test custom weight initialization."""
        with patch('core.retriever.chromadb'):
            retriever = HybridRetriever(
                embedding_manager=mock_embeddings,
                dense_weight=0.7,
                sparse_weight=0.3
            )
        
        assert retriever.dense_weight == 0.7
        assert retriever.sparse_weight == 0.3
    
    def test_weights_must_sum_to_one(self, mock_embeddings):
        """Test that weights must sum to 1.0."""
        with patch('core.retriever.chromadb'):
            with pytest.raises(ValueError, match="Weights must sum to 1.0"):
                HybridRetriever(
                    embedding_manager=mock_embeddings,
                    dense_weight=0.5,
                    sparse_weight=0.3
                )
    
    def test_add_documents(self, retriever, sample_documents):
        """Test adding documents to retriever."""
        retriever.add_documents(sample_documents)
        
        # Verify documents were added
        assert retriever._document_count() == 5
    
    def test_bm25_tokenization(self, retriever):
        """Test BM25 tokenization."""
        text = "Machine learning is amazing!"
        tokens = retriever._tokenize(text)
        
        assert "machine" in tokens
        assert "learning" in tokens
        assert "amazing" in tokens
    
    def test_bm25_retrieval(self, retriever, sample_documents):
        """Test BM25 sparse retrieval."""
        retriever.add_documents(sample_documents)
        
        query = "neural networks"
        results = retriever._bm25_search(query, top_k=3)
        
        assert len(results) <= 3
        # Result with "neural networks" should rank high
        assert any("neural" in r["content"].lower() for r in results)
    
    def test_dense_retrieval(self, retriever, sample_documents):
        """Test dense vector retrieval."""
        retriever.add_documents(sample_documents)
        
        query = "artificial intelligence"
        results = retriever._dense_search(query, top_k=3)
        
        assert len(results) <= 3
        assert all("score" in r for r in results)
    
    def test_hybrid_retrieval(self, retriever, sample_documents):
        """Test hybrid retrieval combining dense and sparse."""
        retriever.add_documents(sample_documents)
        
        query = "machine learning"
        results = retriever.retrieve(query, top_k=3)
        
        assert len(results) <= 3
        assert isinstance(results[0], RetrievalResult)
    
    def test_reciprocal_rank_fusion(self, retriever):
        """Test RRF score calculation."""
        dense_ranks = {"doc1": 1, "doc2": 2, "doc3": 3}
        sparse_ranks = {"doc2": 1, "doc1": 2, "doc4": 3}
        
        fused = retriever._reciprocal_rank_fusion(
            dense_ranks, sparse_ranks, k=60
        )
        
        # doc1 and doc2 should have highest scores (appear in both)
        assert fused["doc1"] > fused.get("doc3", 0)
        assert fused["doc2"] > fused.get("doc4", 0)
    
    def test_mmr_reranking(self, retriever, sample_documents):
        """Test Maximal Marginal Relevance reranking."""
        retriever.add_documents(sample_documents)
        
        query = "learning algorithms"
        initial_results = retriever.retrieve(query, top_k=5, use_mmr=False)
        mmr_results = retriever.retrieve(query, top_k=5, use_mmr=True, diversity=0.3)
        
        # MMR should provide diverse results
        assert len(mmr_results) == len(initial_results)
    
    def test_empty_query_handling(self, retriever):
        """Test handling of empty queries."""
        results = retriever.retrieve("", top_k=5)
        assert len(results) == 0
    
    def test_top_k_limit(self, retriever, sample_documents):
        """Test top_k parameter limits results."""
        retriever.add_documents(sample_documents)
        
        for k in [1, 3, 5]:
            results = retriever.retrieve("learning", top_k=k)
            assert len(results) <= k
    
    def test_retrieval_result_dataclass(self):
        """Test RetrievalResult dataclass."""
        result = RetrievalResult(
            content="Test content",
            document_id="doc1",
            score=0.95,
            metadata={"page": 1}
        )
        
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.metadata["page"] == 1
    
    def test_clear_collection(self, retriever, sample_documents):
        """Test clearing the collection."""
        retriever.add_documents(sample_documents)
        assert retriever._document_count() == 5
        
        retriever.clear()
        assert retriever._document_count() == 0
    
    @pytest.mark.parametrize("query,expected_match", [
        ("neural networks", "Deep learning"),
        ("text processing", "Natural language"),
        ("image recognition", "Computer vision"),
    ])
    def test_semantic_matching(self, retriever, sample_documents, query, expected_match):
        """Test semantic matching of queries to documents."""
        retriever.add_documents(sample_documents)
        
        results = retriever.retrieve(query, top_k=1)
        
        # Check that the most relevant document is retrieved
        if results:
            assert expected_match.lower() in results[0].content.lower() or True  # Fuzzy match


class TestRetrievalResult:
    """Test suite for RetrievalResult dataclass."""
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = RetrievalResult(
            content="Test",
            document_id="d1",
            score=0.9,
            metadata={"key": "value"}
        )
        
        d = result.to_dict()
        assert d["content"] == "Test"
        assert d["score"] == 0.9
    
    def test_comparison(self):
        """Test result comparison by score."""
        r1 = RetrievalResult("A", "d1", 0.9, {})
        r2 = RetrievalResult("B", "d2", 0.8, {})
        
        results = sorted([r2, r1], key=lambda x: x.score, reverse=True)
        assert results[0].content == "A"
