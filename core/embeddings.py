"""
Embedding Manager

Handles embedding generation for documents and queries using
multiple embedding models with caching support.
"""

import os
from typing import List, Optional, Dict, Any
from functools import lru_cache
import hashlib
import numpy as np

from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """
    Manages embedding generation with support for multiple models.
    
    Supports OpenAI embeddings for production use and local
    sentence-transformers for offline/development use.
    """
    
    OPENAI_MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    }
    
    LOCAL_MODELS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "multi-qa-mpnet-base-dot-v1": 768
    }
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        use_local: bool = False,
        cache_embeddings: bool = True
    ):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the embedding model to use
            use_local: Whether to use local sentence-transformers
            cache_embeddings: Whether to cache generated embeddings
        """
        self.model_name = model_name
        self.use_local = use_local
        self.cache_embeddings = cache_embeddings
        self._embedding_cache: Dict[str, List[float]] = {}
        
        if use_local or model_name in self.LOCAL_MODELS:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.LOCAL_MODELS.get(model_name, 384)
        else:
            self.model = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            self.dimension = self.OPENAI_MODELS.get(model_name, 1536)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            text: Query text
            
        Returns:
            List of floats representing the embedding
        """
        # Check cache
        if self.cache_embeddings:
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]
        
        # Generate embedding
        if self.use_local or isinstance(self.model, SentenceTransformer):
            embedding = self.model.encode(text).tolist()
        else:
            embedding = self.model.embed_query(text)
        
        # Cache result
        if self.cache_embeddings:
            self._embedding_cache[cache_key] = embedding
        
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of embeddings
        """
        # Check cache for each text
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            if self.cache_embeddings:
                cache_key = self._get_cache_key(text)
                if cache_key in self._embedding_cache:
                    embeddings.append(self._embedding_cache[cache_key])
                    continue
            
            texts_to_embed.append(text)
            indices_to_embed.append(i)
            embeddings.append(None)  # Placeholder
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            if self.use_local or isinstance(self.model, SentenceTransformer):
                new_embeddings = self.model.encode(texts_to_embed).tolist()
            else:
                new_embeddings = self.model.embed_documents(texts_to_embed)
            
            # Fill in results and update cache
            for idx, embedding in zip(indices_to_embed, new_embeddings):
                embeddings[idx] = embedding
                if self.cache_embeddings:
                    cache_key = self._get_cache_key(texts[idx])
                    self._embedding_cache[cache_key] = embedding
        
        return embeddings
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text."""
        return hashlib.md5(
            f"{self.model_name}:{text}".encode()
        ).hexdigest()
    
    def similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
    
    @property
    def cache_size(self) -> int:
        """Get the number of cached embeddings."""
        return len(self._embedding_cache)
