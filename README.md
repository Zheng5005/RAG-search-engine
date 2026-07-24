# RAG Search Engine

A Retrieval-Augmented Generation (RAG) search engine for movie discovery, combining multiple search strategies with LLM-powered answer generation.

## Overview

This project implements a production-grade search system that blends traditional keyword matching with modern semantic understanding. It retrieves relevant movies from a dataset and generates natural-language answers using large language models.

### Key Features

| Capability | Description |
|------------|-------------|
| **Keyword Search** | Inverted index with BM25 scoring and TF-IDF weighting |
| **Semantic Search** | Vector embeddings using sentence-transformers (`all-MiniLM-L6-v2`) |
| **Hybrid Search** | Combines BM25 + semantic via Reciprocal Rank Fusion (RRF) |
| **Multimodal Search** | Image-based search using CLIP embeddings |
| **Query Enhancement** | LLM-powered spell correction, query rewriting, and expansion |
| **Re-ranking** | Individual, batch, and cross-encoder re-ranking strategies |
| **RAG Generation** | Retrieval-augmented answers with source citations |
| **Evaluation** | Precision@k, Recall@k, and F1 score metrics |

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenRouter API key (for LLM features)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd rag-search-engine

# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env  # Add your OPENROUTER_API_KEY
```

### First Run

```bash
# Build the inverted index (required for keyword search)
cd cli
python keyword_search_cli.py build

# Verify the embedding model loads correctly
python semantic_search_cli.py verify

# Test a search
python semantic_search_cli.py search "bear movie"
```

## Architecture

```
rag-search-engine/
├── cli/                          # Command-line interfaces
│   ├── lib/                      # Core search libraries
│   │   ├── semantic_search.py    # Vector embeddings & cosine similarity
│   │   ├── keyword_search.py     # Inverted index & BM25
│   │   ├── hybrid_search.py      # RRF fusion & LLM integration
│   │   ├── multimodal_search.py  # CLIP image search
│   │   └── search_utils.py       # Shared formatting utilities
│   ├── semantic_search_cli.py    # Semantic search commands
│   ├── keyword_search_cli.py     # Keyword search commands
│   ├── hybrid_search_cli.py      # Hybrid search commands
│   ├── multimodal_search_cli.py  # Image search commands
│   ├── augmented_generation_cli.py  # RAG generation commands
│   └── evaluation_cli.py         # Search evaluation metrics
├── data/
│   ├── course-rag-movies.json    # Movie dataset (1000+ films)
│   ├── golden_dataset.json       # Evaluation test cases
│   └── stopwords.txt             # Stopword list for tokenization
└── cache/                        # Generated indices & embeddings
```

## Usage

All commands run from the `cli/` directory.

### Semantic Search

```bash
# Search by meaning (not just keywords)
python semantic_search_cli.py search "heartwarming animal story"

# Generate embeddings for text
python semantic_search_cli.py embed "your text here"

# Chunk-based search for longer documents
python semantic_search_cli.py search-chunked "adventure film"
```

### Keyword Search

```bash
# Build the inverted index (run once)
python keyword_search_cli.py build

# Search by exact terms
python keyword_search_cli.py search "bear marmalade"

# BM25 ranked search
python keyword_search_cli.py bm25search "horror movie"

# Inspect TF-IDF scores
python keyword_search_cli.py tfidf 1 "bear"
python keyword_search_cli.py idf "bear"
```

### Hybrid Search

```bash
# Weighted combination (alpha: 0=semantic, 1=keyword)
python hybrid_search_cli.py weighted-search "cute bear" --alpha 0.5

# Reciprocal Rank Fusion
python hybrid_search_cli.py rrf-search "talking animal comedy"

# With query enhancement
python hybrid_search_cli.py rrf-search "scary bear movie" --enhance spell
python hybrid_search_cli.py rrf-search "funny animal film" --enhance rewrite
python hybrid_search_cli.py rrf-search "bear adventure" --enhance expand

# With re-ranking
python hybrid_search_cli.py rrf-search "bear movie" --rerank-method cross_encoder
```

### RAG Generation

```bash
# Search + generate natural language answer
python augmented_generation_cli.py rag "What are some good bear movies?"

# Summarize search results
python augmented_generation_cli.py summarize "children's animated films"

# Answer with citations
python augmented_generation_cli.py citations "best horror movies"

# Conversational Q&A
python augmented_generation_cli.py question "What's that movie where the bear eats marmalade?"
```

### Multimodal Search

```bash
# Search by image
python multimodal_search_cli.py image_search data/paddington.jpeg

# Verify image embedding works
python multimodal_search_cli.py verify_image_embedding data/paddington.jpeg
```

### Evaluation

```bash
# Run evaluation against golden dataset
python evaluation_cli.py --limit 5
```

## Search Strategies

### BM25 (Keyword)

Traditional information retrieval using term frequency and inverse document frequency. Best for exact term matching and known-item search.

**Parameters:**
- `k1=1.5` — Term frequency saturation
- `b=0.75` — Document length normalization

### Semantic (Vector)

Dense vector embeddings capture meaning beyond exact words. Uses `all-MiniLM-L6-v2` for 384-dimensional embeddings with cosine similarity.

### Hybrid (RRF)

Reciprocal Rank Fusion combines rankings from both methods without requiring score normalization:

```
RRF_score(d) = Σ 1/(k + rank_i(d))
```

Default `k=60` balances the contribution of each ranking.

### Re-ranking

Three strategies to refine initial results:

| Method | Approach | Trade-off |
|--------|----------|-----------|
| `individual` | LLM scores each result separately | Most accurate, slowest (rate-limited) |
| `batch` | LLM ranks all results at once | Fast, less granular |
| `cross_encoder` | Cross-encoder model (`ms-marco-TinyBERT-L2`) | No API calls, good quality |

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | API key for LLM queries (spell check, rewriting, RAG) |

### Model Configuration

| Component | Default Model | Notes |
|-----------|---------------|-------|
| Embeddings | `all-MiniLM-L6-v2` | 384 dimensions, fast inference |
| Multimodal | `clip-ViT-B-32` | CLIP for image-text alignment |
| Re-ranking | `ms-marco-TinyBERT-L2` | Cross-encoder for precision |
| LLM | `openrouter/free` | Configurable via `MODEL` constant |

## Evaluation

The project includes a golden dataset of curated test cases for measuring search quality:

```bash
python evaluation_cli.py --limit 5
```

**Metrics:**
- **Precision@k** — Fraction of retrieved results that are relevant
- **Recall@k** — Fraction of relevant results that were retrieved
- **F1 Score** — Harmonic mean of precision and recall

## Dataset

The movie dataset (`data/course-rag-movies.json`) contains 1000+ films with:
- `id` — Unique identifier
- `title` — Movie title
- `description` — Plot synopsis and metadata

The golden dataset (`data/golden_dataset.json`) provides 20+ test cases with known relevant documents for evaluation.

## Technical Details

### Embedding Pipeline

1. **Chunking** — Documents split into sentence-boundary chunks (4 sentences, 1 overlap)
2. **Encoding** — Chunks embedded via `all-MiniLM-L6-v2`
3. **Aggregation** — Best chunk score per document
4. **Caching** — Embeddings saved to `cache/` for fast reload

### Index Structure

The inverted index stores:
- `index.pkl` — Term → document ID mapping
- `docmap.pkl` — Document ID → metadata
- `term_frequencies.pkl` — Per-document term counts
- `doc_lengths.pkl` — Document lengths for BM25 normalization

### Query Enhancement

LLM-powered preprocessing before search:

| Method | Purpose | Example |
|--------|---------|---------|
| `spell` | Fix typos | "movi" → "movie" |
| `rewrite` | Make more specific | "bear movie" → "Leonardo DiCaprio bear attack film" |
| `expand` | Add synonyms | "scary bear" → "scary horror grizzly bear terrifying" |

## License

[Add your license here]
