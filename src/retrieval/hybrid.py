from src.retrieval.bm25 import BM25Retriever
from src.retrieval.semantic import SemanticRetriever


class HybridRetriever:
    def __init__(
        self,
        bm25,
        semantic,
        rrf_k=60,
        candidate_k=50,
        max_chunks_per_article=1,
    ):
        if rrf_k < 1 or candidate_k < 1:
            raise ValueError("RRF parameters must be positive")

        self.bm25 = bm25
        self.semantic = semantic
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.max_chunks_per_article = max_chunks_per_article

    @classmethod
    def from_jsonl(cls, path="data/processed/chunks.jsonl"):
        return cls(
            bm25=BM25Retriever.from_jsonl(path),
            semantic=SemanticRetriever.from_jsonl(path),
        )

    def search(self, query, k=5):
        if k < 1:
            raise ValueError("k must be positive")

        candidate_k = max(k, self.candidate_k)

        bm25_results = self.bm25.search(query, k=candidate_k)
        semantic_results = self.semantic.search(
            query,
            k=candidate_k,
        )

        scores = {}
        chunks = {}

        for results in (bm25_results, semantic_results):
            for rank, chunk in enumerate(results, start=1):
                chunk_id = chunk["chunk_id"]

                scores[chunk_id] = (
                    scores.get(chunk_id, 0.0)
                    + 1.0 / (self.rrf_k + rank)
                )

                chunks[chunk_id] = chunk

        ranked_ids = sorted(
            scores,
            key=lambda chunk_id: (
                -scores[chunk_id],
                chunk_id,
            ),
        )

        selected = []
        article_counts = {}

        for chunk_id in ranked_ids:
            chunk = chunks[chunk_id]
            article = chunk["article_number"]

            if self.max_chunks_per_article is not None:
                if (
                    article_counts.get(article, 0)
                    >= self.max_chunks_per_article
                ):
                    continue

            selected.append({
                **chunk,
                "score": scores[chunk_id],
            })

            article_counts[article] = (
                article_counts.get(article, 0) + 1
            )

            if len(selected) == k:
                break

        return selected
