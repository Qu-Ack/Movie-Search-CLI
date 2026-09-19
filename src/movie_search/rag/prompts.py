from movie_search.models import SearchResult

SYSTEM_PROMPT = """You are an expert movie assistant with access to a rich movie database.
Your goal is to answer the user's query accurately, helpfully, and concisely, using the retrieved movie context provided below.

Guidelines:
1. Ground your answer in the provided movie context. Highlight relevant plot points, themes, and characters mentioned in the synopses.
2. Clearly reference and cite movie titles when recommending or describing them (e.g. "[1] Inception").
3. If the retrieved context contains limited or partial matches for the query, mention the closest available matches and clearly explain any limitations.
4. Provide a structured, engaging answer.
"""


def build_context_string(sources: list[SearchResult]) -> str:
    """Format a list of SearchResults into a clean numbered context string."""
    if not sources:
        return "No relevant movie records found in the database."

    context_blocks = []
    for idx, movie in enumerate(sources, start=1):
        block = (
            f"[{idx}] Title: {movie.title}\n"
            f"Relevance Score: {movie.score:.3f} "
            f"(Keyword: {movie.keyword_score:.3f}, Semantic: {movie.semantic_score:.3f})\n"
            f"Synopsis: {movie.description.strip()}"
        )
        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)


def build_rag_prompt(query: str, sources: list[SearchResult]) -> str:
    """Build the complete user prompt containing context and query."""
    context = build_context_string(sources)
    return (
        f"Retrieved Movie Context:\n"
        f"{context}\n\n"
        f"========================\n"
        f"User Query: {query}\n\n"
        f"Answer:"
    )
