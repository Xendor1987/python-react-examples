"""
Text Chunking Service for RAG Applications

A service for splitting documents into chunks optimized for:
- Retrieval-Augmented Generation (RAG)
- Vector embeddings
- Semantic search

Features:
- Smart sentence-boundary detection
- Configurable chunk size and overlap
- Email/document metadata handling
- Clean text preprocessing
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TextChunk:
    """Structure representing a text chunk"""
    content: str
    chunk_index: int
    char_start: int
    char_end: int


@dataclass
class ChunkingConfig:
    """Configuration for the chunking service"""
    chunk_size: int = 500          # Target chunk size in characters
    chunk_overlap: int = 50        # Overlap between chunks
    min_chunk_size: int = 100      # Minimum chunk size


class TextChunkingService:
    """Service for splitting text into chunks for RAG applications"""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the chunking service.

        Args:
            config: Chunking configuration (uses defaults if not provided)
        """
        self.config = config or ChunkingConfig()

    def _clean_text(self, text: str) -> str:
        """
        Clean text from extra whitespace and HTML entities.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove HTML numeric entities
        text = re.sub(r'&#\d+;', '', text)
        # Remove HTML named entities
        text = re.sub(r'&\w+;', ' ', text)
        return text.strip()

    def _find_split_point(self, text: str, target_pos: int) -> int:
        """
        Find the best split point near target position.
        Prefers splitting at sentence boundaries or paragraphs.

        Args:
            text: Full text being split
            target_pos: Target position for splitting

        Returns:
            Optimal split position
        """
        # Search range
        search_start = max(0, target_pos - 100)
        search_end = min(len(text), target_pos + 100)
        search_text = text[search_start:search_end]

        # Look for sentence endings (. ! ?) or newlines
        sentence_patterns = [r'\.\s', r'\!\s', r'\?\s', r'\n']

        for pattern in sentence_patterns:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # Choose the match closest to target position
                best_match = min(
                    matches,
                    key=lambda m: abs((search_start + m.end()) - target_pos)
                )
                return search_start + best_match.end()

        # If no sentence boundary found, look for a space
        space_pos = text.rfind(' ', search_start, search_end)
        if space_pos != -1:
            return space_pos + 1

        return target_pos

    def split_text(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks with smart boundary detection.

        Args:
            text: Text to split

        Returns:
            List of TextChunk objects
        """
        text = self._clean_text(text)

        if not text:
            return []

        # If text is short enough, return single chunk
        if len(text) <= self.config.chunk_size:
            return [TextChunk(
                content=text,
                chunk_index=0,
                char_start=0,
                char_end=len(text)
            )]

        chunks = []
        current_pos = 0
        chunk_index = 0

        while current_pos < len(text):
            # Determine end of current chunk
            chunk_end = current_pos + self.config.chunk_size

            if chunk_end >= len(text):
                # Last chunk
                chunk_text = text[current_pos:].strip()
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunks.append(TextChunk(
                        content=chunk_text,
                        chunk_index=chunk_index,
                        char_start=current_pos,
                        char_end=len(text)
                    ))
                elif chunks:
                    # Append to previous chunk if too short
                    prev = chunks[-1]
                    chunks[-1] = TextChunk(
                        content=prev.content + ' ' + chunk_text,
                        chunk_index=prev.chunk_index,
                        char_start=prev.char_start,
                        char_end=len(text)
                    )
                break

            # Find optimal split point
            split_pos = self._find_split_point(text, chunk_end)
            chunk_text = text[current_pos:split_pos].strip()

            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(TextChunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    char_start=current_pos,
                    char_end=split_pos
                ))
                chunk_index += 1

            # Next chunk starts with overlap
            current_pos = split_pos - self.config.chunk_overlap
            if chunks and current_pos <= chunks[-1].char_start:
                current_pos = split_pos

        return chunks

    def split_email(
        self,
        content: str,
        subject: Optional[str] = None,
        sender: Optional[str] = None,
        sent_at: Optional[str] = None
    ) -> List[TextChunk]:
        """
        Split email into chunks with metadata context.

        Adds header information (subject, sender, date) to the first chunk
        for better context in RAG applications.

        Args:
            content: Email body text
            subject: Email subject
            sender: Sender email/name
            sent_at: Send date string

        Returns:
            List of TextChunk objects with metadata
        """
        # Build header for context
        header_parts = []
        if subject:
            header_parts.append(f"Subject: {subject}")
        if sender:
            header_parts.append(f"From: {sender}")
        if sent_at:
            header_parts.append(f"Date: {sent_at}")

        header = " | ".join(header_parts)

        # Split content
        content_chunks = self.split_text(content)

        # Add header to first chunk
        if content_chunks and header:
            first = content_chunks[0]
            content_chunks[0] = TextChunk(
                content=f"{header}\n\n{first.content}",
                chunk_index=first.chunk_index,
                char_start=first.char_start,
                char_end=first.char_end
            )

        return content_chunks

    def split_chat_messages(
        self,
        messages: List[dict],
        chat_name: Optional[str] = None
    ) -> List[TextChunk]:
        """
        Split chat messages into chunks.

        Groups messages together while respecting chunk size limits.

        Args:
            messages: List of message dicts with 'sender', 'content', 'timestamp'
            chat_name: Name of the chat for context

        Returns:
            List of TextChunk objects
        """
        if not messages:
            return []

        # Format messages
        formatted_lines = []
        if chat_name:
            formatted_lines.append(f"Chat: {chat_name}\n")

        for msg in messages:
            sender = msg.get('sender', 'Unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')

            line = f"[{timestamp}] {sender}: {content}"
            formatted_lines.append(line)

        full_text = "\n".join(formatted_lines)
        return self.split_text(full_text)


# Utility functions for common use cases

def chunk_document(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Simple function to chunk a document.

    Args:
        text: Document text
        chunk_size: Target chunk size
        overlap: Overlap between chunks

    Returns:
        List of chunk strings
    """
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    service = TextChunkingService(config)
    chunks = service.split_text(text)
    return [chunk.content for chunk in chunks]


def chunk_for_embedding(
    text: str,
    max_tokens: int = 512
) -> List[str]:
    """
    Chunk text optimized for embedding models.

    Most embedding models have a token limit (often 512).
    This function estimates ~4 chars per token.

    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk

    Returns:
        List of chunk strings
    """
    # Estimate: ~4 characters per token
    chunk_size = max_tokens * 4

    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=50,
        min_chunk_size=100
    )
    service = TextChunkingService(config)
    chunks = service.split_text(text)
    return [chunk.content for chunk in chunks]


# Example usage and tests
if __name__ == "__main__":
    # Test basic chunking
    sample_text = """
    Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience without being explicitly programmed.
    It focuses on developing computer programs that can access data and use it
    to learn for themselves.

    The process begins with observations or data, such as examples, direct
    experience, or instruction. It looks for patterns in data and makes better
    decisions in the future based on the examples provided.

    The primary aim is to allow computers to learn automatically without human
    intervention and adjust actions accordingly. Deep learning is a subset of
    machine learning that uses neural networks with many layers.
    """

    print("=" * 50)
    print("Testing Text Chunking Service")
    print("=" * 50)

    # Test with default config
    service = TextChunkingService()
    chunks = service.split_text(sample_text)

    print(f"\nOriginal text length: {len(sample_text)} characters")
    print(f"Number of chunks: {len(chunks)}\n")

    for chunk in chunks:
        print(f"Chunk {chunk.chunk_index}:")
        print(f"  Position: {chunk.char_start}-{chunk.char_end}")
        print(f"  Length: {len(chunk.content)} chars")
        print(f"  Preview: {chunk.content[:80]}...")
        print()

    # Test email chunking
    print("=" * 50)
    print("Testing Email Chunking")
    print("=" * 50)

    email_chunks = service.split_email(
        content=sample_text,
        subject="Introduction to Machine Learning",
        sender="professor@university.edu",
        sent_at="2024-01-15"
    )

    print(f"\nFirst chunk with header:")
    print(email_chunks[0].content[:200])

    # Test utility function
    print("\n" + "=" * 50)
    print("Testing Utility Functions")
    print("=" * 50)

    simple_chunks = chunk_document(sample_text, chunk_size=300)
    print(f"\nSimple chunking (300 chars): {len(simple_chunks)} chunks")

    embedding_chunks = chunk_for_embedding(sample_text, max_tokens=128)
    print(f"Embedding-optimized (128 tokens): {len(embedding_chunks)} chunks")
