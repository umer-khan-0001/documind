"""
LLM Chain Module

Handles LLM interactions for answer generation with context,
including prompt management and hallucination mitigation.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks import get_openai_callback


@dataclass
class GeneratedResponse:
    """Response from the LLM chain."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    tokens_used: int
    model: str


class LLMChain:
    """
    LLM Chain for RAG-based answer generation.
    
    Supports multiple LLM providers and includes built-in
    prompts for hallucination mitigation and source attribution.
    """
    
    SYSTEM_PROMPT = """You are DocuMind, an AI assistant specialized in answering questions about documents.
Your role is to provide accurate, helpful answers based ONLY on the provided context.

IMPORTANT RULES:
1. Only use information from the provided context to answer questions
2. If the context doesn't contain enough information, say so clearly
3. Always cite which part of the context your answer comes from
4. Be concise but comprehensive
5. If you're uncertain, express your uncertainty
6. Never make up information not present in the context

When citing sources, use the format: [Source: Page X, Type: Y]
"""

    RAG_PROMPT = """Context from the documents:
{context}

---
Sources available:
{sources}

---
User Question: {question}

Please provide a comprehensive answer based on the context above. Remember to:
1. Only use information from the provided context
2. Cite your sources using the format [Source: Page X]
3. If the context doesn't fully answer the question, acknowledge this

Answer:"""

    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        provider: str = "openai"
    ):
        """
        Initialize the LLM chain.
        
        Args:
            model_name: Name of the LLM model
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            provider: LLM provider (openai or anthropic)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        
        if provider == "openai":
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        elif provider == "anthropic":
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", self.RAG_PROMPT)
        ])
        
        self.chat_history: List[Dict[str, str]] = []
    
    def generate(
        self,
        question: str,
        context: List[Any],
        sources: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> GeneratedResponse:
        """
        Generate an answer for the given question.
        
        Args:
            question: User's question
            context: Retrieved document chunks
            sources: Source information for chunks
            chat_history: Optional conversation history
            
        Returns:
            GeneratedResponse with answer and metadata
        """
        # Format context
        context_str = self._format_context(context)
        sources_str = self._format_sources(sources)
        
        # Build messages
        messages = self.prompt.format_messages(
            context=context_str,
            sources=sources_str,
            question=question
        )
        
        # Add chat history if provided
        if chat_history:
            history_messages = []
            for msg in chat_history[-6:]:  # Keep last 6 messages
                if msg["role"] == "user":
                    history_messages.append(HumanMessage(content=msg["content"]))
                else:
                    history_messages.append(AIMessage(content=msg["content"]))
            messages = [messages[0]] + history_messages + messages[1:]
        
        # Generate response
        tokens_used = 0
        if self.provider == "openai":
            with get_openai_callback() as cb:
                response = self.llm.invoke(messages)
                tokens_used = cb.total_tokens
        else:
            response = self.llm.invoke(messages)
        
        # Calculate confidence based on source coverage
        confidence = self._calculate_confidence(response.content, sources)
        
        # Update chat history
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": response.content})
        
        return GeneratedResponse(
            answer=response.content,
            sources=sources,
            confidence=confidence,
            tokens_used=tokens_used,
            model=self.model_name
        )
    
    def _format_context(self, chunks: List[Any]) -> str:
        """Format chunks into context string."""
        formatted = []
        for i, chunk in enumerate(chunks):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            page = chunk.page_number if hasattr(chunk, 'page_number') else 'N/A'
            chunk_type = chunk.chunk_type if hasattr(chunk, 'chunk_type') else 'text'
            
            formatted.append(
                f"[Chunk {i+1}] (Page {page}, Type: {chunk_type})\n{content}"
            )
        
        return "\n\n---\n\n".join(formatted)
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources into readable string."""
        formatted = []
        for i, source in enumerate(sources):
            formatted.append(
                f"- Source {i+1}: Page {source.get('page', 'N/A')}, "
                f"Type: {source.get('type', 'unknown')}"
            )
        
        return "\n".join(formatted)
    
    def _calculate_confidence(
        self, 
        answer: str, 
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate confidence score based on source citations.
        
        Higher confidence if the answer cites sources and doesn't
        contain uncertainty markers.
        """
        confidence = 0.8  # Base confidence
        
        # Check for source citations
        citation_count = answer.lower().count("source:") + answer.lower().count("page")
        if citation_count > 0:
            confidence += min(0.1, citation_count * 0.02)
        
        # Reduce confidence for uncertainty markers
        uncertainty_markers = [
            "i'm not sure", "i don't know", "unclear",
            "cannot determine", "not enough information",
            "may", "might", "possibly", "perhaps"
        ]
        
        answer_lower = answer.lower()
        for marker in uncertainty_markers:
            if marker in answer_lower:
                confidence -= 0.1
                break
        
        return max(0.0, min(1.0, confidence))
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.chat_history = []
