from src.retrieval.hybrid import HybridRetriever
from src.retrieval.legal_router import detect_relevant_articles


class RoutedHybridRetriever:
    def __init__(self, hybrid):
        self.hybrid = hybrid

        # Use the same corpus as the hybrid retriever.
        self.chunks_by_article = {}

        for chunk in hybrid.bm25.chunks:
            article = chunk["article_number"]
            self.chunks_by_article.setdefault(article, []).append(chunk)

    @classmethod
    def from_jsonl(cls, path="data/processed/chunks.jsonl"):
        return cls(HybridRetriever.from_jsonl(path))

    def search(self, query, k=5):
        if k < 1:
            raise ValueError("k must be positive")

        results = self.hybrid.search(query, k=k)
        target_articles = detect_relevant_articles(query)

        if not target_articles:
            return results

        selected = list(results)
        existing_articles = {
            item["article_number"]
            for item in selected
        }

        # Select the highest-scoring BM25 chunk within each target article.
        candidate_pool = self.hybrid.bm25.search(
            query,
            k=len(self.hybrid.bm25.chunks),
        )

        missing_targets = sorted(
            article
            for article in target_articles
            if article not in existing_articles
        )

        routed_chunks = []

        for article in missing_targets:
            chosen = next(
                (
                    chunk
                    for chunk in candidate_pool
                    if chunk["article_number"] == article
                ),
                None,
            )

            if chosen is not None:
                routed_chunks.append({
                    **chosen,
                    "score": None,
                    "retrieval_source": "legal_router",
                })

        protected_articles = target_articles

        for routed_chunk in routed_chunks:
            if len(selected) >= k:
                removable_index = next(
                    (
                        i
                        for i in range(len(selected) - 1, -1, -1)
                        if selected[i]["article_number"]
                        not in protected_articles
                    ),
                    None,
                )

                if removable_index is None:
                    break

                selected.pop(removable_index)

            selected.append(routed_chunk)

        return selected
