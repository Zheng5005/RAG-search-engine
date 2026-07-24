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

# Download required data files
mkdir -p data
curl -L -o data/course-rag-movies.json \
  "https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json"
curl -L -o data/paddington.jpeg \
  "https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/0LmrJI7-194x260.jpeg"
```

The `golden_dataset.json` and `stopwords.txt` files are included in the repository (see Dataset section below to recreate them).

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
│   ├── course-rag-movies.json    # Movie dataset (downloaded, see Setup)
│   ├── paddington.jpeg           # Sample image (downloaded, see Setup)
│   ├── golden_dataset.json       # Evaluation test cases (included)
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

## Dataset

### Movie Data

Download the movie dataset (required for all search features):

```bash
curl -L -o data/course-rag-movies.json \
  "https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json"
```

The dataset contains 1000+ films with:
- `id` — Unique identifier
- `title` — Movie title
- `description` — Plot synopsis and metadata

### Test Image

Download the sample Paddington image for multimodal search:

```bash
curl -L -o data/paddington.jpeg \
  "https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/0LmrJI7-194x260.jpeg"
```

### Golden Dataset

The `golden_dataset.json` is included in the repository with 10 curated test cases for evaluation:

```json
{
  "test_cases": [
    {
      "query": "cute british bear marmalade",
      "relevant_docs": ["Paddington"]
    },
    {
      "query": "talking teddy bear comedy",
      "relevant_docs": ["Ted", "Ted 2"]
    },
    {
      "query": "children's animated bear adventure",
      "relevant_docs": [
        "Brother Bear",
        "The Jungle Book",
        "The Many Adventures of Winnie the Pooh",
        "Yogi Bear",
        "The Care Bears Movie",
        "Care Bears Movie II: A New Generation",
        "Care Bears Nutcracker Suite",
        "The Little Polar Bear",
        "The Little Polar Bear 2: The Mysterious Island",
        "Open Season",
        "The Country Bears",
        "The Berenstain Bears' Christmas Tree",
        "Winnie the Pooh"
      ]
    },
    {
      "query": "friendship transformation magic with bears",
      "relevant_docs": [
        "Brother Bear",
        "The Care Bears Movie",
        "The Jungle Book"
      ]
    },
    {
      "query": "dinosaur park",
      "relevant_docs": ["Jurassic Park"]
    },
    {
      "query": "wizards and magic",
      "relevant_docs": [
        "Harry Potter and the Sorcerer's Stone",
        "Harry Potter and the Prisoner of Azkaban",
        "Harry Potter and the Goblet of Fire",
        "Harry Potter and the Order of the Phoenix",
        "Harry Potter and the Deathly Hallows: Part 1",
        "Harry Potter and the Deathly Hallows: Part 2",
        "The Sword in the Stone",
        "Oz the Great and Powerful",
        "The Lord of the Rings: The Fellowship of the Ring"
      ]
    },
    {
      "query": "superhero saves the world",
      "relevant_docs": [
        "The Incredibles",
        "Superman II",
        "Superman/Batman: Public Enemies",
        "Justice League: The Flashpoint Paradox",
        "Up, Up, and Away!",
        "Megamind",
        "Kick-Ass",
        "Sky High"
      ]
    },
    {
      "query": "zombie apocalypse",
      "relevant_docs": [
        "Shaun of the Dead",
        "Dance of the Dead",
        "The Return of the Living Dead",
        "Pride and Prejudice and Zombies",
        "I Am Legend",
        "Resident Evil: Apocalypse",
        "Colin",
        "Død snø"
      ]
    },
    {
      "query": "car racing",
      "relevant_docs": [
        "The Fast and the Furious",
        "Rush",
        "Need for Speed",
        "Talladega Nights: The Ballad of Ricky Bobby",
        "The Love Bug",
        "Cars",
        "Furious Seven"
      ]
    },
    {
      "query": "romantic comedy wedding",
      "relevant_docs": [
        "Runaway Bride",
        "27 Dresses",
        "Just Go with It",
        "The Wedding Planner",
        "Wedding Crashers",
        "The Accidental Husband",
        "You, Me and Dupree"
      ]
    }
  ]
}
```

Run evaluation against this dataset:

```bash
python evaluation_cli.py --limit 5
```

**Metrics:**
- **Precision@k** — Fraction of retrieved results that are relevant
- **Recall@k** — Fraction of relevant results that were retrieved
- **F1 Score** — Harmonic mean of precision and recall

### Stopwords

The `stopwords.txt` file is included in the repository with 198 common English stopwords used for tokenization:

```
a
about
above
after
again
against
ain
all
am
an
and
any
are
aren
aren't
as
at
be
because
been
before
being
below
between
both
but
by
can
couldn
couldn't
d
did
didn
didn't
do
does
doesn
doesn't
doing
don
don't
down
during
each
few
for
from
further
had
hadn
hadn't
has
hasn
hasn't
have
haven
haven't
having
he
he'd
he'll
he's
her
here
hers
herself
him
himself
his
how
i
i'd
i'll
i'm
i've
if
in
into
is
isn
isn't
it
it'd
it'll
it's
its
itself
just
ll
m
ma
me
mightn
mightn't
more
most
mustn
mustn't
my
myself
needn
needn't
no
nor
not
now
o
of
off
on
once
only
or
other
our
ours
ourselves
out
over
own
re
s
same
shan
shan't
she
she'd
she'll
she's
should
should've
shouldn
shouldn't
so
some
such
t
than
that
that'll
the
their
theirs
them
themselves
then
there
these
they
they'd
they'll
they're
they've
this
those
through
to
too
under
until
up
ve
very
was
wasn
wasn't
we
we'd
we'll
we're
we've
were
weren
weren't
what
when
where
which
while
who
whom
why
will
with
won
won't
wouldn
wouldn't
y
you
you'd
you'll
you're
you've
your
yours
yourself
yourselves
```

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
