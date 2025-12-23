"""
RAG (Retrieval-Augmented Generation) layer using ChromaDB and OpenAI embeddings.
Stores and retrieves similar repository examples for classification.
"""

import hashlib
from dataclasses import dataclass

import chromadb
from openai import AsyncOpenAI

from repo_classifier.cache import embedding_cache
from repo_classifier.config import settings


@dataclass
class Precedent:
    """A similar repository found in the RAG store."""
    url: str
    category: str
    similarity: float


class RAGStore:
    """
    Vector store for taxonomy repository examples.
    
    Uses ChromaDB for persistence and OpenAI for embeddings.
    Supports adding repositories and querying for similar ones.
    """

    COLLECTION_NAME = "taxonomy"

    def __init__(self, db_path: str | None = None):
        """
        Initialize the RAG store.
        
        Args:
            db_path: Path to ChromaDB storage (uses config default if not specified)
        """
        db_path = db_path or settings.chroma_db_path
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Taxonomy repository examples for RAG"}
        )
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key)

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Check cache first
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        cache_key = f"embedding:{text_hash}"
        
        cached = embedding_cache.get(cache_key)
        if cached is not None:
            return cached

        # Generate embedding
        response = await self.openai.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        
        embedding = response.data[0].embedding
        embedding_cache.set(cache_key, embedding, settings.cache_ttl_seconds)
        
        return embedding

    async def add_repository(
        self,
        url: str,
        readme: str,
        category: str,
        description: str | None = None,
    ) -> None:
        """
        Add a taxonomy repository to the store.
        
        Args:
            url: Repository URL (used as unique ID)
            readme: README content
            category: Assigned category
            description: Optional repository description
        """
        # Build text for embedding
        text_parts = []
        if description:
            text_parts.append(description)
        if readme:
            text_parts.append(readme[:4000])  # Limit for embedding
        
        text = " ".join(text_parts) if text_parts else url
        
        # Generate embedding
        embedding = await self.embed_text(text)
        
        # Use URL as unique ID (hash it for ChromaDB compatibility)
        doc_id = hashlib.sha256(url.encode()).hexdigest()[:32]
        
        # Upsert (add or update)
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[{
                "url": url,
                "category": category,
                "description": description or "",
            }],
            documents=[text[:1000]],  # Store truncated text for reference
        )

    async def query_similar(
        self,
        readme: str,
        description: str | None = None,
        k: int | None = None,
    ) -> list[Precedent]:
        """
        Find similar repositories in the store.
        
        Args:
            readme: README content to match against
            description: Optional description to include
            k: Number of results (uses config default if not specified)
            
        Returns:
            List of similar repositories with categories and scores
        """
        k = k or settings.rag_k_neighbors
        
        # Build query text
        text_parts = []
        if description:
            text_parts.append(description)
        if readme:
            text_parts.append(readme[:4000])
        
        if not text_parts:
            return []
        
        query_text = " ".join(text_parts)
        
        # Generate query embedding
        embedding = await self.embed_text(query_text)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["metadatas", "distances"],
        )
        
        # Convert to Precedent objects
        precedents = []
        
        if results["metadatas"] and results["distances"]:
            for metadata, distance in zip(
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB returns L2 distance; convert to similarity (0-1)
                # Lower distance = higher similarity
                similarity = 1.0 / (1.0 + distance)
                
                precedents.append(Precedent(
                    url=metadata["url"],
                    category=metadata["category"],
                    similarity=similarity,
                ))
        
        return precedents

    def count(self) -> int:
        """Return number of documents in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Taxonomy repository examples for RAG"}
        )

    def list_categories(self) -> dict[str, int]:
        """
        Get counts of repositories per category.
        
        Returns:
            Dict mapping category names to counts
        """
        # Get all documents
        results = self.collection.get(include=["metadatas"])
        
        counts: dict[str, int] = {}
        for metadata in results.get("metadatas", []):
            category = metadata.get("category", "Unknown")
            counts[category] = counts.get(category, 0) + 1
        
        return counts


# Global RAG store instance (lazy initialization)
_rag_store: RAGStore | None = None


def get_rag_store() -> RAGStore:
    """Get or create the global RAG store instance."""
    global _rag_store
    if _rag_store is None:
        _rag_store = RAGStore()
    return _rag_store

