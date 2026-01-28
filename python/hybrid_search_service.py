"""
Hybrid Search Service with Vector Embeddings

A production-ready service for semantic and keyword search using:
- sentence-transformers for multilingual embeddings
- PostgreSQL + pgvector for vector storage
- Hybrid search combining keyword matching with semantic similarity

Features:
- Lazy model loading with caching
- Batch processing for efficiency
- Russian language stemming support
- Configurable similarity thresholds
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


# Global variable for model caching
_embedding_model = None


@dataclass
class SearchConfig:
    """Configuration for the search service"""
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dimension: int = 384
    default_limit: int = 10
    min_similarity: float = 0.3
    max_text_length: int = 8000


def get_embedding_model(model_name: str = "intfloat/multilingual-e5-small"):
    """
    Get embedding model with lazy loading and caching.
    Uses sentence-transformers for creating real embeddings.

    Args:
        model_name: Name of the sentence-transformers model

    Returns:
        Loaded SentenceTransformer model
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
        print(f"Model loaded. Dimension: {_embedding_model.get_sentence_embedding_dimension()}")
    return _embedding_model


def text_to_embedding(text: str, max_length: int = 8000) -> List[float]:
    """
    Generate real embedding for text using sentence-transformers.

    Args:
        text: Text to create embedding for
        max_length: Maximum text length (will be truncated)

    Returns:
        List of floats - embedding vector
    """
    model = get_embedding_model()
    text = text.strip()[:max_length]
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def text_to_embeddings_batch(texts: List[str], max_length: int = 8000) -> List[List[float]]:
    """
    Generate embeddings for a list of texts (batch processing).
    More efficient for large numbers of texts.

    Args:
        texts: List of texts
        max_length: Maximum text length per item

    Returns:
        List of embeddings
    """
    model = get_embedding_model()
    texts = [t.strip()[:max_length] for t in texts]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 10
    )
    return [e.tolist() for e in embeddings]


class EmbeddingService:
    """Service for generating embeddings using sentence-transformers"""

    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self._model = None

    def _get_model(self):
        """Lazy model loading"""
        if self._model is None:
            self._model = get_embedding_model(self.config.embedding_model)
        return self._model

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        return text_to_embedding(text, self.config.max_text_length)

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for list of texts"""
        return text_to_embeddings_batch(texts, self.config.max_text_length)


def stem_russian(word: str) -> str:
    """
    Simple Russian stemmer - removes common endings for better search.

    Args:
        word: Word to stem

    Returns:
        Stemmed word
    """
    if len(word) <= 4:
        return word

    endings = [
        'ами', 'ями', 'ого', 'его', 'ому', 'ему', 'ой', 'ей', 'ом', 'ем',
        'ах', 'ях', 'ые', 'ие', 'ую', 'юю', 'ая', 'яя', 'ый', 'ий', 'ое', 'ее',
        'ов', 'ев', 'ей', 'ам', 'ям', 'ах', 'ях',
        'ть', 'ет', 'ут', 'ют', 'ит', 'ат', 'ят',
        'ы', 'и', 'а', 'я', 'у', 'ю', 'е', 'о'
    ]

    for ending in endings:
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            return word[:-len(ending)]
    return word


def extract_keywords(query: str) -> List[str]:
    """
    Extract keywords from query with stemming support.

    Args:
        query: Search query

    Returns:
        List of keywords including stemmed variants
    """
    query_lower = query.lower().strip()
    raw_keywords = [w for w in query_lower.split() if len(w) >= 3]

    keywords = []
    for w in raw_keywords:
        keywords.append(w)
        stemmed = stem_russian(w)
        if stemmed != w and stemmed not in keywords:
            keywords.append(stemmed)

    return keywords


class HybridSearchService:
    """
    Hybrid search service combining keyword and semantic search.

    Priority is given to exact keyword matches, then supplemented
    with semantic search results.
    """

    def __init__(self, db_session, config: Optional[SearchConfig] = None):
        """
        Initialize the search service.

        Args:
            db_session: Async SQLAlchemy session
            config: Search configuration
        """
        self.db = db_session
        self.config = config or SearchConfig()
        self.embedding_service = EmbeddingService(config)

    async def keyword_search(
        self,
        user_id: int,
        keywords: List[str],
        limit: int = 10,
        source_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search.

        Args:
            user_id: User ID for filtering
            keywords: List of keywords to search
            limit: Maximum results
            source_type: Optional filter by source type

        Returns:
            List of search results
        """
        from sqlalchemy import text

        if not keywords:
            return []

        like_conditions = []
        params = {"user_id": user_id, "limit": limit}

        for i, kw in enumerate(keywords[:5]):
            param_name = f"kw{i}"
            like_conditions.append(
                f"(LOWER(c.content) LIKE :{param_name} OR "
                f"LOWER(d.subject) LIKE :{param_name} OR "
                f"LOWER(d.sender) LIKE :{param_name})"
            )
            params[param_name] = f"%{kw}%"

        where_clause = " OR ".join(like_conditions)

        sql = f"""
            SELECT
                c.id,
                c.content,
                c.chunk_index,
                c.metadata,
                d.id as document_id,
                d.source_type,
                d.subject,
                d.sender,
                d.sent_at,
                1.0 as similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.user_id = :user_id
                AND ({where_clause})
        """

        if source_type:
            sql += " AND d.source_type = :source_type"
            params["source_type"] = source_type

        sql += " ORDER BY d.sent_at DESC NULLS LAST LIMIT :limit"

        result = await self.db.execute(text(sql), params)
        rows = result.fetchall()

        return [self._row_to_dict(row, similarity=1.0) for row in rows]

    async def semantic_search(
        self,
        user_id: int,
        query: str,
        limit: int = 10,
        source_type: Optional[str] = None,
        exclude_ids: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector embeddings.

        Args:
            user_id: User ID for filtering
            query: Search query
            limit: Maximum results
            source_type: Optional filter by source type
            exclude_ids: IDs to exclude from results

        Returns:
            List of search results with similarity scores
        """
        from sqlalchemy import text

        # Generate query embedding
        query_embedding = text_to_embedding(query)
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        exclude_clause = ""
        params = {"user_id": user_id, "limit": limit}

        if exclude_ids:
            exclude_clause = f" AND c.id NOT IN ({','.join(map(str, exclude_ids))})"

        sql = f"""
            SELECT
                c.id,
                c.content,
                c.chunk_index,
                c.metadata,
                d.id as document_id,
                d.source_type,
                d.subject,
                d.sender,
                d.sent_at,
                1 - (c.embedding <=> '{embedding_str}'::vector) as similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.user_id = :user_id
                AND c.embedding IS NOT NULL
                {exclude_clause}
        """

        if source_type:
            sql += " AND d.source_type = :source_type"
            params["source_type"] = source_type

        sql += f"""
            ORDER BY c.embedding <=> '{embedding_str}'::vector
            LIMIT :limit
        """

        result = await self.db.execute(text(sql), params)
        rows = result.fetchall()

        return [self._row_to_dict(row) for row in rows]

    async def hybrid_search(
        self,
        user_id: int,
        query: str,
        limit: int = 10,
        source_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: keyword search + semantic search.

        Priority is given to exact keyword matches,
        then supplemented with semantic search.

        Args:
            user_id: User ID for filtering
            query: Search query
            limit: Maximum results
            source_type: Optional filter by source type

        Returns:
            Combined list of results
        """
        keywords = extract_keywords(query)
        results = []
        seen_ids = set()

        # 1. KEYWORD SEARCH - find exact matches first
        if keywords:
            keyword_results = await self.keyword_search(
                user_id=user_id,
                keywords=keywords,
                limit=limit,
                source_type=source_type
            )

            for result in keyword_results:
                if result["chunk_id"] not in seen_ids:
                    seen_ids.add(result["chunk_id"])
                    results.append(result)

        # 2. SEMANTIC SEARCH - supplement with semantic results
        if len(results) < limit:
            remaining = limit - len(results)
            semantic_results = await self.semantic_search(
                user_id=user_id,
                query=query,
                limit=remaining,
                source_type=source_type,
                exclude_ids=seen_ids
            )

            for result in semantic_results:
                if result["chunk_id"] not in seen_ids:
                    results.append(result)

        return results[:limit]

    async def get_context_for_query(
        self,
        user_id: int,
        query: str,
        max_chunks: int = 5,
        max_context_length: int = 4000
    ) -> str:
        """
        Get formatted context for RAG query.

        Args:
            user_id: User ID
            query: User query
            max_chunks: Maximum number of chunks
            max_context_length: Maximum context length in characters

        Returns:
            Formatted context string for AI
        """
        results = await self.hybrid_search(
            user_id=user_id,
            query=query,
            limit=max_chunks
        )

        if not results:
            return ""

        context_parts = []
        current_length = 0

        for result in results:
            # Format context block
            source_info = f"[{result['source_type'].upper()}]"
            if result.get('subject'):
                source_info += f" {result['subject']}"
            if result.get('sender'):
                source_info += f" from {result['sender']}"
            if result.get('sent_at'):
                source_info += f" ({result['sent_at'][:10]})"

            block = f"---\n{source_info}\n{result['content']}\n"

            # Check length limit
            if current_length + len(block) > max_context_length:
                break

            context_parts.append(block)
            current_length += len(block)

        return "\n".join(context_parts)

    def _row_to_dict(self, row, similarity: Optional[float] = None) -> Dict[str, Any]:
        """Convert database row to dictionary"""
        return {
            "chunk_id": row.id,
            "content": row.content,
            "chunk_index": row.chunk_index,
            "extra_data": row.metadata,
            "document_id": row.document_id,
            "source_type": row.source_type,
            "subject": row.subject,
            "sender": row.sender,
            "sent_at": row.sent_at.isoformat() if row.sent_at else None,
            "similarity": similarity if similarity is not None else float(row.similarity)
        }


# Example usage and tests
if __name__ == "__main__":
    # Test embedding generation
    test_texts = [
        "Hello, how are you today?",
        "Привет, как дела?",
        "Machine learning is fascinating"
    ]

    print("Testing embedding generation...")
    for text in test_texts:
        embedding = text_to_embedding(text)
        print(f"Text: {text[:30]}... -> Embedding dim: {len(embedding)}")

    # Test batch processing
    print("\nTesting batch processing...")
    embeddings = text_to_embeddings_batch(test_texts)
    print(f"Generated {len(embeddings)} embeddings")

    # Test keyword extraction
    print("\nTesting keyword extraction...")
    queries = ["поиск документов", "найти письма от партнёров"]
    for q in queries:
        keywords = extract_keywords(q)
        print(f"Query: {q} -> Keywords: {keywords}")
