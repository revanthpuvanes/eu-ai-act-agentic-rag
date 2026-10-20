"""Hybrid retrieval with optional full-article context expansion."""

from src.retrieval.context_expansion import ArticleContextExpander
from src.retrieval.hybrid import HybridRetriever


class ExpandedHybridRetriever:
    def __init__(
        self,
        base_retriever,
        expander,
        expand_top_n=1,
    ):
        self.base_retriever = base_retriever
        self.expander = expander
        self.expand_top_n = expand_top_n

    @classmethod
    def from_jsonl(
        cls,
        expand_top_n=1,
        max_article_chars=15000,
        max_context_chars=22000,
    ):
        base_retriever = HybridRetriever.from_jsonl()

        expander = ArticleContextExpander(
            max_article_chars=max_article_chars,
            max_context_chars=max_context_chars,
        )

        return cls(
            base_retriever=base_retriever,
            expander=expander,
            expand_top_n=expand_top_n,
        )

    def search(self, query, k=5):
        chunks = self.base_retriever.search(query, k=k)

        return self.expander.expand(
            chunks,
            expand_top_n=self.expand_top_n,
        )
