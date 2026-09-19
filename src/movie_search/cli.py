import argparse
import os
import sys

from movie_search.config import (
    DEFAULT_ALPHA,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TOP_N,
    GOLDEN_DATASET_FILE,
    MOVIES_FILE,
)
from movie_search.evaluation import evaluate
from movie_search.indexing.inverted_index import InvertedIndex
from movie_search.indexing.semantic_index import SemanticIndex
from movie_search.rag.pipeline import RAGPipeline
from movie_search.rag.providers import get_llm_provider
from movie_search.search.hybrid import HybridSearcher
from movie_search.search.keyword import KeywordSearcher


def handle_build(args: argparse.Namespace) -> None:
    movies_path = args.movies or MOVIES_FILE
    print(f"Building search indexes from {movies_path}...")

    print("-> Building inverted index (TF-IDF)...")
    inverted_index = InvertedIndex()
    inverted_index.build(movies_path=movies_path)
    print("   Inverted index built successfully.")

    print("-> Building semantic index (Sentence Transformers + ChromaDB)...")
    semantic_index = SemanticIndex()
    semantic_index.build(movies_path=movies_path)
    print("   Semantic index built successfully.")

    print("\nAll search indexes successfully built and cached.")


def handle_search(args: argparse.Namespace) -> None:
    try:
        searcher = KeywordSearcher()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run `python main.py build` first to create the index.")
        sys.exit(1)

    results = searcher.search(args.query, top_k=args.top_k)
    if not results:
        print("No movies found.")
        return

    print(f"\nTop {len(results)} keyword matches for '{args.query}':\n")
    for position, res in enumerate(results, start=1):
        print(f"{position}. {res.title} (Score: {res.score:.3f})")
        print(f"   {res.description}\n")


def handle_hybrid_search(args: argparse.Namespace) -> None:
    try:
        searcher = HybridSearcher()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run `python main.py build` first to create the index.")
        sys.exit(1)

    results = searcher.search(args.query, alpha=args.alpha, top_n=args.top_n)
    if not results:
        print("No movies found.")
        return

    print(
        f"\nTop {len(results)} hybrid matches for '{args.query}' "
        f"(alpha={args.alpha}):\n"
    )
    for position, res in enumerate(results, start=1):
        print(
            f"{position}. {res.title} (Hybrid: {res.score:.3f}, "
            f"Semantic: {res.semantic_score:.3f}, Keyword: {res.keyword_score:.3f})"
        )
        print(f"   {res.description}\n")


def handle_rag(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    provider_name = args.provider

    if provider_name == "gemini" and not api_key:
        print("\n[Error] Gemini API key not found.")
        print("Please provide your API key in one of the following ways:")
        print("  1. Pass the --api-key argument: `python main.py rag -q '...' --api-key 'YOUR_KEY'`")
        print("  2. Set the GEMINI_API_KEY environment variable:")
        print("     PowerShell: $env:GEMINI_API_KEY=\"YOUR_KEY\"")
        print("     Bash:       export GEMINI_API_KEY=\"YOUR_KEY\"")
        print("  3. Add GEMINI_API_KEY=YOUR_KEY to a `.env` file in the project root.\n")
        sys.exit(1)

    try:
        retriever = HybridSearcher()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run `python main.py build` first to create the index.")
        sys.exit(1)

    try:
        llm_provider = get_llm_provider(
            provider_name=provider_name,
            api_key=api_key,
            model_name=args.model,
        )
    except Exception as e:
        print(f"[Error initializing LLM provider]: {e}")
        sys.exit(1)

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_provider=llm_provider,
        alpha=args.alpha,
        top_n=args.top_n,
    )

    print(f"\nRetrieving relevant movies with Hybrid Search (alpha={args.alpha})...")
    print(f"Generating answer with {args.model} ({provider_name})...\n")

    response = pipeline.query(args.query, alpha=args.alpha, top_n=args.top_n)

    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(response.answer)
    print("=" * 60)

    if args.show_sources:
        print("\nRetrieved Movie Sources Used:")
        for idx, src in enumerate(response.sources, start=1):
            print(
                f"  [{idx}] {src.title} "
                f"(Hybrid Score: {src.score:.3f} | "
                f"Semantic: {src.semantic_score:.3f} | "
                f"Keyword: {src.keyword_score:.3f})"
            )


def handle_evaluate(args: argparse.Namespace) -> None:
    try:
        evaluate(
            dataset_path=args.dataset,
            top_n=args.top_n,
            alpha=args.alpha,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have run `python main.py build` and the dataset file exists.")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="movie-search",
        description="Movie Search & RAG CLI: Hybrid keyword + semantic retrieval and LLM generation.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # build
    build_p = subparsers.add_parser("build", help="Build keyword and semantic search indexes")
    build_p.add_argument("--movies", default=str(MOVIES_FILE), help="Path to movies.json")

    # search (keyword)
    search_p = subparsers.add_parser("search", help="Search movies using keyword (TF-IDF) matching")
    search_p.add_argument("-q", "--query", type=str, required=True, help="Search query")
    search_p.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    # hybrid_search
    hybrid_p = subparsers.add_parser(
        "hybrid_search",
        help="Search movies using combined keyword and semantic ranking",
    )
    hybrid_p.add_argument("-q", "--query", type=str, required=True, help="Search query")
    hybrid_p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Semantic weight from 0.0 (keyword only) to 1.0 (semantic only). Default: 0.7",
    )
    hybrid_p.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of results to return. Default: 5",
    )

    # rag
    rag_p = subparsers.add_parser(
        "rag",
        help="Answer questions about movies using Hybrid Retrieval-Augmented Generation (RAG)",
    )
    rag_p.add_argument("-q", "--query", type=str, required=True, help="Question / query for the RAG system")
    rag_p.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=["gemini", "mock"],
        help="LLM provider (default: gemini, or mock for testing)",
    )
    rag_p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_GEMINI_MODEL,
        help=f"Model name to use (default: {DEFAULT_GEMINI_MODEL})",
    )
    rag_p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (defaults to GEMINI_API_KEY from environment or .env)",
    )
    rag_p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Semantic weight for hybrid retrieval. Default: 0.7",
    )
    rag_p.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of top movies to feed to the LLM context. Default: 5",
    )
    rag_p.add_argument(
        "--show-sources",
        action="store_true",
        default=True,
        help="Display the list of retrieved movies used as context (default: True)",
    )
    rag_p.add_argument(
        "--no-sources",
        dest="show_sources",
        action="store_false",
        help="Do not display the list of retrieved movie sources",
    )

    # evaluate
    eval_p = subparsers.add_parser(
        "evaluate",
        help="Evaluate keyword and hybrid search against the golden dataset",
    )
    eval_p.add_argument(
        "--dataset",
        default=str(GOLDEN_DATASET_FILE),
        help="Path to golden dataset json file",
    )
    eval_p.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of results to evaluate. Default: 5",
    )
    eval_p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Semantic weight from 0.0 to 1.0. Default: 0.7",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    match args.command:
        case "build":
            handle_build(args)
        case "search":
            handle_search(args)
        case "hybrid_search":
            handle_hybrid_search(args)
        case "rag":
            handle_rag(args)
        case "evaluate":
            handle_evaluate(args)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
