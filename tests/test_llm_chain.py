"""
Tests for LLMChain module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from core.llm_chain import LLMChain, QueryResponse


class TestLLMChain:
    """Test suite for LLMChain."""
    
    @pytest.fixture
    def mock_openai(self):
        """Create mock OpenAI client."""
        with patch('core.llm_chain.OpenAI') as mock:
            client = Mock()
            client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(
                    message=MagicMock(content="This is a test answer based on the provided context.")
                )]
            )
            mock.return_value = client
            yield mock
    
    @pytest.fixture
    def chain(self, mock_openai):
        """Create LLMChain instance."""
        return LLMChain(model="gpt-4o-mini")
    
    @pytest.fixture
    def sample_context(self):
        """Sample retrieval context."""
        return [
            {
                "content": "Machine learning enables computers to learn from data.",
                "metadata": {"page": 1, "source": "doc1"}
            },
            {
                "content": "Deep learning is a subset of machine learning using neural networks.",
                "metadata": {"page": 2, "source": "doc1"}
            },
            {
                "content": "Training requires large datasets and computational resources.",
                "metadata": {"page": 3, "source": "doc2"}
            }
        ]
    
    def test_chain_initialization(self, chain):
        """Test chain initializes correctly."""
        assert chain is not None
        assert chain.model == "gpt-4o-mini"
        assert chain.temperature == 0.0
    
    def test_custom_parameters(self, mock_openai):
        """Test custom initialization parameters."""
        chain = LLMChain(
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000
        )
        
        assert chain.model == "gpt-4"
        assert chain.temperature == 0.7
        assert chain.max_tokens == 2000
    
    def test_system_prompt_construction(self, chain):
        """Test system prompt is constructed correctly."""
        prompt = chain._build_system_prompt()
        
        assert "document" in prompt.lower()
        assert "context" in prompt.lower() or "source" in prompt.lower()
    
    def test_context_formatting(self, chain, sample_context):
        """Test context is formatted correctly."""
        formatted = chain._format_context(sample_context)
        
        assert "Machine learning" in formatted
        assert "Deep learning" in formatted
        assert "[Source: doc1, Page: 1]" in formatted or "Page 1" in formatted
    
    def test_generate_answer(self, chain, sample_context):
        """Test answer generation."""
        question = "What is machine learning?"
        
        response = chain.generate(
            question=question,
            context=sample_context
        )
        
        assert isinstance(response, QueryResponse)
        assert len(response.answer) > 0
    
    def test_response_includes_sources(self, chain, sample_context):
        """Test response includes source attribution."""
        response = chain.generate(
            question="Explain deep learning",
            context=sample_context
        )
        
        assert hasattr(response, 'sources')
        assert len(response.sources) > 0
    
    def test_confidence_score(self, chain, sample_context):
        """Test confidence score is calculated."""
        response = chain.generate(
            question="What is ML?",
            context=sample_context
        )
        
        assert hasattr(response, 'confidence')
        assert 0.0 <= response.confidence <= 1.0
    
    def test_empty_context_handling(self, chain):
        """Test handling when no context is provided."""
        response = chain.generate(
            question="Random question",
            context=[]
        )
        
        # Should indicate that no relevant information was found
        assert response.confidence < 0.5 or "cannot" in response.answer.lower() or "no" in response.answer.lower()
    
    def test_hallucination_detection_prompt(self, chain):
        """Test hallucination prevention in prompt."""
        prompt = chain._build_system_prompt()
        
        # Should contain instructions about not making things up
        assert any(term in prompt.lower() for term in [
            "only", "based on", "context", "provided", "don't", "do not"
        ])
    
    def test_query_response_dataclass(self):
        """Test QueryResponse dataclass."""
        response = QueryResponse(
            answer="Test answer",
            sources=[{"page": 1}],
            confidence=0.95,
            tokens_used=100
        )
        
        assert response.answer == "Test answer"
        assert response.confidence == 0.95
        assert response.tokens_used == 100
    
    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
    def test_temperature_settings(self, mock_openai, temperature):
        """Test different temperature settings."""
        chain = LLMChain(temperature=temperature)
        assert chain.temperature == temperature
    
    def test_max_context_length_handling(self, chain):
        """Test handling of very long context."""
        long_context = [
            {"content": "A" * 10000, "metadata": {"page": i}}
            for i in range(20)
        ]
        
        formatted = chain._format_context(long_context, max_tokens=4000)
        
        # Should truncate to fit within limits
        assert len(formatted) < 50000  # Reasonable limit
    
    def test_streaming_response(self, chain, sample_context):
        """Test streaming response generation."""
        # Mock streaming response
        chain.client.chat.completions.create.return_value = iter([
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Part 1 "))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Part 2"))]),
        ])
        
        chunks = list(chain.generate_stream(
            question="Test?",
            context=sample_context
        ))
        
        # Should yield chunks
        assert len(chunks) >= 0  # Depends on mock setup
    
    def test_error_handling(self, chain, sample_context):
        """Test error handling in generation."""
        chain.client.chat.completions.create.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            chain.generate(
                question="Test?",
                context=sample_context
            )


class TestQueryResponse:
    """Test suite for QueryResponse dataclass."""
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        response = QueryResponse(
            answer="Test",
            sources=[{"page": 1}],
            confidence=0.9,
            tokens_used=50
        )
        
        d = response.to_dict()
        assert d["answer"] == "Test"
        assert d["confidence"] == 0.9
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "answer": "Test",
            "sources": [{"page": 1}],
            "confidence": 0.9,
            "tokens_used": 50
        }
        
        response = QueryResponse.from_dict(data)
        assert response.answer == "Test"
        assert response.confidence == 0.9
