import unittest
from unittest.mock import patch

from src.retrieval.routed_hybrid import RoutedHybridRetriever


class FakeBM25:
    def __init__(self):
        self.chunks = [
            {"chunk_id": str(i), "article_number": i, "text": str(i)}
            for i in range(1, 7)
        ]

    def search(self, query, k=5):
        return [
            {**chunk, "score": 1.0}
            for chunk in self.chunks[:k]
        ]


class FakeHybrid:
    def __init__(self):
        self.bm25 = FakeBM25()

    def search(self, query, k=5):
        return [
            {"chunk_id": str(i), "article_number": i, "score": 1.0}
            for i in range(1, k + 1)
        ]


class RoutedHybridTests(unittest.TestCase):
    def test_multiple_targets_are_preserved(self):
        retriever = RoutedHybridRetriever(FakeHybrid())

        with patch(
            "src.retrieval.routed_hybrid.detect_relevant_articles",
            return_value={5, 6},
        ):
            results = retriever.search("test", k=3)

        articles = {
            result["article_number"]
            for result in results
        }

        self.assertEqual(len(results), 3)
        self.assertIn(5, articles)
        self.assertIn(6, articles)


if __name__ == "__main__":
    unittest.main()
