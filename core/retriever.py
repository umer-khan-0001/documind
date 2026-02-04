"""
Hybrid Retrieval System

Combines dense and sparse retrieval methods with Reciprocal Rank Fusion
for optimal document chunk retrieval.
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings

from core.embeddings import EmbeddingManager
from core.document_processor import DocumentChunk, ProcessedDocument


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    chunks: List[DocumentChunk]
    sources: List[Dict[str, Any]]
    scores: List[float]
    
    def __len__(self):
        return len(self.chunks)


class HybridRetriever:
    """
    Hybrid retrieval system combining dense and sparse methods.
    
    Uses ChromaDB for dense vector retrieval and BM25 for sparse retrieval,
    with Reciprocal Rank Fusion to combine results.
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        collection_name: str = "documind_docs",
        persist_directory: str = "./data/chroma",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            embedding_manager: Manager for generating embeddings
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persisting vector store
            dense_weight: Weight for dense retrieval scores
            sparse_weight: Weight for sparse retrieval scores
        """
        self.embeddings = embedding_manager
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # BM25 index (rebuilt when documents are added)
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_chunks: List[DocumentChunk] = []
        self.chunk_lookup: Dict[str, DocumentChunk] = {}
        
    def index_document(self, document: ProcessedDocument) -> None:
        """
        Index a processed document.
        
        Args:
            document: The processed document to index
        """
        if not document.chunks:
            return
        
        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in document.chunks]
        contents = [chunk.content for chunk in document.chunks]
        metadatas = [
            {
                "document_id": document.id,
                "filename": document.filename,
                "chunk_type": chunk.chunk_type,
                "page_number": chunk.page_number,
                **chunk.metadata
            }
            for chunk in document.chunks
        ]
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(contents)
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        # Update BM25 index
        self.bm25_chunks.extend(document.chunks)
        for chunk in document.chunks:
            self.chunk_lookup[chunk.id] = chunk
        self._rebuild_bm25_index()
        
    def _rebuild_bm25_index(self) -> None:
        """Rebuild the BM25 index with current chunks."""
        if not self.bm25_chunks:
            self.bm25_index = None
            return
            
        tokenized_corpus = [
            chunk.content.lower().split() 
            for chunk in self.bm25_chunks
        ]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        use_mmr: bool = True,
        mmr_diversity: float = 0.3
    ) -> RetrievalResult:
        """
        Retrieve relevant document chunks.
        
        Args:
            query: The search query
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            use_mmr: Whether to use Maximum Marginal Relevance
            mmr_diversity: Diversity parameter for MMR
            
        Returns:
            RetrievalResult: Retrieved chunks with scores
        """
        # Get more candidates for fusion
        n_candidates = top_k * 3
        
        # Dense retrieval
        dense_results = self._dense_retrieve(
            query, n_candidates, filter_dict
        )
        
        # Sparse retrieval
        sparse_results = self._sparse_retrieve(query, n_candidates)
        
        # Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            dense_results, sparse_results, k=60
        )
        
        # Apply MMR if requested
        if use_mmr:
            fused_results = self._apply_mmr(
                query, fused_results, top_k, mmr_diversity
            )
        else:
            fused_results = fused_results[:top_k]
        
        # Build result
        chunks = []
        sources = []
        scores = []
        
        for chunk_id, score in fused_results:
            if chunk_id in self.chunk_lookup:
                chunk = self.chunk_lookup[chunk_id]
                chunks.append(chunk)
                scores.append(score)
                sources.append({
                    "chunk_id": chunk_id,
                    "page": chunk.page_number,
                    "type": chunk.chunk_type,
                    "preview": chunk.content[:100] + "..."
                })
        
        return RetrievalResult(
            chunks=chunks,
            sources=sources,
            scores=scores
        )
    
    def _dense_retrieve(
        self,
        query: str,
        n_results: int,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """Perform dense vector retrieval."""
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        
        if not results['ids'][0]:
            return []
        
        # ChromaDB returns distances, convert to similarity scores
        ids = results['ids'][0]
        distances = results['distances'][0]
        scores = [1 / (1 + d) for d in distances]  # Convert distance to similarity
        
        return list(zip(ids, scores))
    
    def _sparse_retrieve(
        self,
        query: str,
        n_results: int
    ) -> List[Tuple[str, float]]:
        """Perform sparse BM25 retrieval."""
        if self.bm25_index is None:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top indices
        top_indices = np.argsort(scores)[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self.bm25_chunks[idx]
                results.append((chunk.id, float(scores[idx])))
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[str, float]],
        sparse_results: List[Tuple[str, float]],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            k: Constant for RRF formula
            
        Returns:
            Fused and ranked results
        """
        rrf_scores: Dict[str, float] = {}
        
        # Score dense results
        for rank, (doc_id, _) in enumerate(dense_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + \
                self.dense_weight / (k + rank + 1)
        
        # Score sparse results
        for rank, (doc_id, _) in enumerate(sparse_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + \
                self.sparse_weight / (k + rank + 1)
        
        # Sort by fused score
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return fused
    
    def _apply_mmr(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        top_k: int,
        diversity: float
    ) -> List[Tuple[str, float]]:
        """Apply Maximum Marginal Relevance for diversity."""
        if len(candidates) <= top_k:
            return candidates
        
        query_embedding = np.array(self.embeddings.embed_query(query))
        
        # Get embeddings for candidates
        candidate_ids = [c[0] for c in candidates]
        candidate_embeddings = []
        
        for cid in candidate_ids:
            result = self.collection.get(ids=[cid], include=['embeddings'])
            if result['embeddings']:
                candidate_embeddings.append(np.array(result['embeddings'][0]))
            else:
                candidate_embeddings.append(np.zeros_like(query_embedding))
        
        selected = []
        remaining = list(range(len(candidates)))
        
        while len(selected) < top_k and remaining:
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining:
                # Relevance to query
                relevance = np.dot(query_embedding, candidate_embeddings[idx])
                
                # Maximum similarity to already selected
                if selected:
                    similarities = [
                        np.dot(candidate_embeddings[idx], candidate_embeddings[s])
                        for s in selected
                    ]
                    max_sim = max(similarities)
                else:
                    max_sim = 0
                
                # MMR score
                mmr_score = (1 - diversity) * relevance - diversity * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        return [candidates[i] for i in selected]
    
    def clear(self) -> None:
        """Clear all indexed documents."""
        self.chroma_client.delete_collection(self.collection.name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25_index = None
        self.bm25_chunks = []
        self.chunk_lookup = {}
