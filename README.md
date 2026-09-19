# Movie Search & RAG CLI

A modular command-line search and Retrieval-Augmented Generation (RAG) system built on a dataset of 5,000 movies. The system provides hybrid search combining BM25/TF-IDF keyword matching with dense semantic vector search (via ChromaDB and Sentence-Transformers), and an augmented generation pipeline powered by Google Gemini to answer questions grounded in movie synopses.

---

## Features

- **TF-IDF Keyword Search**: Fast inverted index tokenized with NLTK and scored with TF-IDF.
- **Dense Semantic Search**: Cosine similarity search using `sentence-transformers/all-MiniLM-L6-v2` embeddings stored in ChromaDB.
- **Hybrid Retrieval**: Combines normalized semantic and keyword scores via convex combination:
  $$\text{Score} = \alpha \cdot \text{Score}_{\text{semantic}} + (1 - \alpha) \cdot \text{Score}_{\text{keyword}}$$
- **Retrieval-Augmented Generation (RAG)**: Answers user queries using top hybrid retrieval matches as grounding context, powered by Google Gemini (`gemini-2.5-flash`).
- **Evaluation Benchmark**: Evaluates keyword vs. hybrid retrieval performance (Precision, Recall, F1) against `data/golden_dataset.json`.
- **Clean Modular Architecture**: Organized into `movie_search` package with dedicated submodules for indexing, search, RAG, and evaluation.

---

## Project Structure

```text
Movie-Search-CLI/
├── data/
│   ├── movies.json            # 5,000 movie records (id, title, description)
│   └── golden_dataset.json    # Evaluation benchmark queries and expected docs
├── cache/                     # Generated search index files (auto-created)
│   ├── index.pkl              # TF-IDF inverted index term mapping
│   ├── docmap.pkl             # Document ID to Movie mapping
│   ├── tf.pkl                 # Term frequency dictionaries
│   └── embeddings.pkl         # Dense vector embeddings
├── src/
│   └── movie_search/
│       ├── __init__.py        # High-level exports
│       ├── config.py          # Central configuration, file paths, default parameters
│       ├── models.py          # Movie, SearchResult, RAGResponse dataclasses
│       ├── text.py            # Text preprocessing (tokenization, stemming, stop words)
│       ├── indexing/
│       │   ├── __init__.py
│       │   ├── inverted_index.py  # InvertedIndex builder and loader
│       │   └── semantic_index.py  # ChromaDB + SentenceTransformer indexer
│       ├── search/
│       │   ├── __init__.py
│       │   ├── keyword.py         # KeywordSearcher (TF-IDF)
│       │   ├── semantic.py        # SemanticSearcher (Dense vectors)
│       │   └── hybrid.py          # HybridSearcher (Combined scores)
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── prompts.py         # Prompt engineering and context formatting
│       │   ├── providers.py       # GeminiProvider (google-genai) and MockLLMProvider
│       │   └── pipeline.py        # RAGPipeline (Retrieve -> Augment -> Generate)
│       ├── evaluation.py      # Precision, Recall, F1 benchmark runner
│       └── cli.py             # CLI parser and command handlers
├── main.py                    # Root CLI entry point
├── pyproject.toml             # Project dependencies and script entrypoints
└── README.md                  # Project documentation
```

---

## Installation & Setup

### 1. Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) package manager

### 2. Install Dependencies

Clone the repository and run:

```bash
uv sync
```

### 3. API Key Configuration (For RAG)

Set your Gemini API key in one of the following ways:

- **Option A: `.env` file** (recommended for local development):
  Create a `.env` file in the project root:
  ```env
  GEMINI_API_KEY=your_gemini_api_key_here
  ```
- **Option B: Environment variable**:
  ```powershell
  $env:GEMINI_API_KEY="your_gemini_api_key_here"  # Windows PowerShell
  ```
  ```bash
  export GEMINI_API_KEY="your_gemini_api_key_here"  # Linux / macOS
  ```
- **Option C: CLI argument**:
  Pass `--api-key "your_gemini_api_key_here"` directly when executing the `rag` command.

---

## Indexing & Search Commands

### Build Search Indexes

Before searching, build the local TF-IDF and semantic vector embeddings cache:

```bash
uv run python main.py build
```

> **Note**: Building the inverted index takes ~1-2 seconds. Generating sentence transformer embeddings for all 5,000 movies takes ~4-6 minutes on CPU on the first run, and the results are cached under `cache/` for subsequent instant loads.

### 1. Keyword Search (TF-IDF)

Search the movie catalog using keyword matching:

```bash
uv run python main.py search -q "space adventure" --top-k 5
```

### 2. Hybrid Search (Keyword + Semantic)

Perform hybrid search combining keyword match scores and semantic embedding similarity:

```bash
uv run python main.py hybrid_search -q "space adventure" --alpha 0.7 --top-n 5
```

- `--alpha`: Controls semantic vs. keyword weighting from `0.0` (keyword only) to `1.0` (semantic only). Default is `0.7`.
- `--top-n`: Number of results to return. Default is `5`.

### 3. Retrieval-Augmented Generation (RAG)

Ask questions or request recommendations. The RAG pipeline performs hybrid retrieval to find the most relevant movies and provides them as grounded context to Google Gemini:

```bash
uv run python main.py rag -q "Recommend a heartwarming movie about an animal adventure and describe its plot"
```

Additional options:
```bash
uv run python main.py rag -q "What movies feature time travel or time loops?" --alpha 0.7 --top-n 5 --model gemini-2.5-flash
```

Use `--no-sources` if you prefer to hide the source metadata breakdown:
```bash
uv run python main.py rag -q "Suggest a dark psychological thriller" --no-sources
```

For offline or dry-run testing without an API key:
```bash
uv run python main.py rag -q "space travel" --provider mock
```

### 4. Evaluate Search Quality

Evaluate keyword and hybrid search precision, recall, and F1 metrics against the golden dataset:

```bash
uv run python main.py evaluate --alpha 0.7 --top-n 5
```

---

## Python API Usage

You can also import and use the components directly in Python:

```python
from movie_search import HybridSearcher, RAGPipeline, GeminiProvider

# Hybrid Search
hybrid = HybridSearcher()
results = hybrid.search("superhero saving the city", alpha=0.7, top_n=5)
for res in results:
    print(f"{res.title} (Score: {res.score:.3f})")

# RAG Pipeline
rag = RAGPipeline()
response = rag.query("Find me a good animated movie suitable for family night")
print(response.answer)
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
