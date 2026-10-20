import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.index = BM25Okapi(
            [
                tokenize(
                    f"{chunk['title']} {chunk['text']}"
                )
                for chunk in chunks
            ]
        )

    @classmethod
    def from_jsonl(cls, path="data/processed/chunks.jsonl"):
        chunks = [
            json.loads(line)
            for line in Path(path).read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if not chunks:
            raise ValueError("Chunk dataset is empty")
        return cls(chunks)

    def search(self, query, k=5, max_chunks_per_article=None):
        if k < 1:
            raise ValueError("k must be positive")

        if (
            max_chunks_per_article is not None
            and max_chunks_per_article < 1
        ):
            raise ValueError(
                "max_chunks_per_article must be positive"
            )

        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(
            range(len(scores)),
            key=lambda i: (-scores[i], i),
        )

        results = []
        article_counts = {}

        for i in ranked:
            chunk = self.chunks[i]
            article = chunk["article_number"]

            if max_chunks_per_article is not None:
                if article_counts.get(article, 0) >= max_chunks_per_article:
                    continue

            results.append({
                **chunk,
                "score": float(scores[i]),
            })

            article_counts[article] = (
                article_counts.get(article, 0) + 1
            )

            if len(results) == k:
                break

        return results
