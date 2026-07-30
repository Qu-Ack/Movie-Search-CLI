# Movie Search CLI

A toy command-line search program built on a movie dataset. The goal is to learn more about search and recommendation techniques based on user queries, starting with keyword search and extending into semantic and hybrid ranking.

The project currently supports:

- TF-IDF keyword search with an inverted index
- Sentence-transformer embeddings for semantic search
- In-memory ChromaDB cosine similarity lookup
- Hybrid search that combines semantic similarity and TF-IDF scores

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management

## Start On A New Machine

Clone the repository and enter the project directory:

```bash
git clone <repo-url>
cd search
```

Install dependencies:

```bash
uv sync
```

Download the NLTK data used by the tokenizer:

```bash
uv run python -m nltk.downloader punkt punkt_tab stopwords
```

Build the local search caches:

```bash
uv run python main.py build
```

The build command creates the TF-IDF index and movie embeddings under `cache/`. The first semantic build may take several minutes because it downloads and runs the sentence-transformer model.

## Usage

Run keyword search:

```bash
uv run python main.py search -q "space adventure"
```

Run hybrid search:

```bash
uv run python main.py hybrid_search -q "space adventure" --top-n 5 --alpha 0.7
```

`alpha` controls how much weight goes to semantic similarity:

- `1.0` means semantic-only ranking
- `0.0` means TF-IDF-only ranking
- `0.7` favors semantic similarity while still using keyword matches

## Data

The movie data lives in `data/movies.json`. Each movie has an `id`, `title`, and `description`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
