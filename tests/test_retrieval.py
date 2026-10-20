import unittest

from src.retrieval.chunk_articles import chunk_text, build_chunks
from src.retrieval.bm25 import BM25Retriever


class ChunkingTests(unittest.TestCase):
    def test_chunk_offsets_reconstruct_source(self):
        text = "abcdefghij" * 100
        chunks = list(chunk_text(text, size=120, overlap=20))

        self.assertGreater(len(chunks), 1)

        for chunk in chunks:
            self.assertEqual(
                chunk["text"],
                text[chunk["start_char"]:chunk["end_char"]],
            )

    def test_article_metadata(self):
        article = {
            "article_number": 5,
            "article_id": "Article 5",
            "title": "Prohibited AI practices",
            "text": "Example legal provision.",
            "source_url": "https://example.org",
            "source_sha256": "abc",
        }

        chunk = next(iter(build_chunks([article])))
        self.assertEqual(chunk["article_number"], 5)
        self.assertEqual(chunk["start_char"], 0)


class RetrievalTests(unittest.TestCase):
    def test_relevant_chunk_ranks_first(self):
        chunks = [
            {
                "title": "Penalties",
                "text": "Administrative fines and sanctions",
                "chunk_id": "1",
                "article_number": 99,
            },
            {
                "title": "Transparency",
                "text": "Disclosure obligations for AI systems",
                "chunk_id": "2",
                "article_number": 50,
            },
            {
                "title": "Risk management",
                "text": "Risk management system requirements",
                "chunk_id": "3",
                "article_number": 9,
            },
        ]

        retriever = BM25Retriever(chunks)
        results = retriever.search("administrative fines", k=2)

        self.assertEqual(results[0]["chunk_id"], "1")


if __name__ == "__main__":
    unittest.main()
