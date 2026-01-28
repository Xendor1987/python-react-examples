# Python & React Code Examples

Production-ready code samples demonstrating backend and frontend development skills.

## Overview

This repository contains standalone, well-documented code examples extracted from real projects. Each file is self-contained and demonstrates specific patterns and techniques.

## Python Examples

### 1. Hybrid Search Service (`python/hybrid_search_service.py`)

A production-ready semantic search service using vector embeddings.

**Key Features:**
- Hybrid search combining keyword matching with semantic similarity
- PostgreSQL + pgvector integration for vector storage
- Sentence-transformers for multilingual embeddings
- Batch processing with progress tracking
- Russian language stemming support
- Configurable similarity thresholds

**Tech Stack:**
- Python 3.10+
- sentence-transformers
- SQLAlchemy (async)
- PostgreSQL + pgvector
- NumPy

**Demonstrates:**
- Async/await patterns
- Lazy loading with caching
- Dataclasses for configuration
- Raw SQL with parameterized queries
- Vector similarity search (cosine distance)

---

### 2. Text Chunking Service (`python/text_chunking_service.py`)

A service for splitting documents into chunks optimized for RAG (Retrieval-Augmented Generation).

**Key Features:**
- Smart sentence-boundary detection
- Configurable chunk size and overlap
- Email/document metadata handling
- Clean text preprocessing
- Chat message chunking support

**Tech Stack:**
- Python 3.10+
- Standard library only (no dependencies)

**Demonstrates:**
- Clean architecture with dataclasses
- Regex-based text processing
- Algorithm for optimal split point detection
- Utility functions for common use cases
- Comprehensive docstrings and type hints

---

## React Examples

### 3. Chat Interface (`react/ChatInterface.tsx`)

A full-featured chat interface for AI assistants.

**Key Features:**
- Custom Markdown renderer (zero dependencies)
- Session management (history, create, delete)
- Source citations with document preview modal
- Responsive design (desktop sidebar + mobile overlay)
- Loading states and error handling
- Quick action suggestions

**Tech Stack:**
- React 18
- TypeScript
- Tailwind CSS
- Lucide React (icons)

**Demonstrates:**
- React hooks (useState, useEffect, useMemo, useRef)
- Component composition and props typing
- Custom rendering logic without external libraries
- Responsive design patterns
- Accessible UI components
- Event handling and form submission

---

## Usage

### Python

```bash
# Install dependencies
pip install sentence-transformers sqlalchemy asyncpg numpy

# Run example
python python/hybrid_search_service.py
python python/text_chunking_service.py
```

### React

```bash
# Copy ChatInterface.tsx to your Next.js/React project
# Install dependencies
npm install lucide-react

# Import and use
import ChatInterface from './ChatInterface'
```

---

## Code Quality

All examples follow best practices:

- **Type hints** (Python) / **TypeScript** (React)
- **Comprehensive docstrings** and comments
- **Error handling** with meaningful messages
- **Configurable** via dataclasses/interfaces
- **Testable** with example usage in `__main__`
- **Production-ready** patterns from real applications

---

## License

MIT License - feel free to use in your projects.
