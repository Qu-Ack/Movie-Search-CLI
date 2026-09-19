from typing import Optional

from movie_search.config import DEFAULT_ALPHA, DEFAULT_TOP_N
from movie_search.models import RAGResponse
from movie_search.rag.prompts import SYSTEM_PROMPT, build_rag_prompt
from movie_search.rag.providers import BaseLLMProvider, GeminiProvider, get_llm_provider
from movie_search.search.hybrid import HybridSearcher


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    Orchestrates hybrid retrieval of relevant movies and contextual answer generation via LLM.
    """

    def __init__(
        self,
        retriever: Optional[HybridSearcher] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        alpha: float = DEFAULT_ALPHA,
        top_n: int = DEFAULT_TOP_N,
    ):
        self.retriever = retriever if retriever is not None else HybridSearcher()
        self.llm_provider = llm_provider if llm_provider is not None else GeminiProvider()
        self.alpha = alpha
        self.top_n = top_n

    def query(
        self,
        user_query: str,
        alpha: Optional[float] = None,
        top_n: Optional[int] = None,
    ) -> RAGResponse:
        current_alpha = alpha if alpha is not None else self.alpha
        current_top_n = top_n if top_n is not None else self.top_n

        # 1. Retrieve candidates using hybrid search (combining keyword and semantic scores)
        sources = self.retriever.search(
            query=user_query,
            alpha=current_alpha,
            top_n=current_top_n,
        )

        # 2. Build augmented prompt
        prompt = build_rag_prompt(query=user_query, sources=sources)

        # 3. Generate answer via LLM
        answer = self.llm_provider.generate(
            prompt=prompt,
            system_instruction=SYSTEM_PROMPT,
        )

        model_name = getattr(self.llm_provider, "model_name", "unknown")

        return RAGResponse(
            query=user_query,
            answer=answer,
            sources=sources,
            model=model_name,
        )
