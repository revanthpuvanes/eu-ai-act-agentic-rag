import json
import tempfile
import unittest
from pathlib import Path

from src.retrieval.context_expansion import ArticleContextExpander


class ContextExpansionTests(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "articles.jsonl"

        articles = [
            {
                "article_number": 5,
                "title": "Prohibited AI practices",
                "text": "A" * 5000,
            },
            {
                "article_number": 1,
                "title": "Subject matter",
                "text": "B" * 2000,
            },
        ]

        with self.path.open("w", encoding="utf-8") as file:
            for article in articles:
                file.write(json.dumps(article) + "\n")

        self.chunks = [
            {
                "article_number": 5,
                "text": "A" * 1200,
                "start_char": 0,
                "end_char": 1200,
            },
            {
                "article_number": 1,
                "text": "B" * 1000,
                "start_char": 0,
                "end_char": 1000,
            },
        ]

    def tearDown(self):
        self.tempdir.cleanup()

    def test_expands_top_article(self):
        expander = ArticleContextExpander(
            self.path,
            max_article_chars=6000,
            max_context_chars=8000,
        )

        result = expander.expand(self.chunks)

        self.assertEqual(len(result[0]["text"]), 5000)
        self.assertTrue(result[0]["context_expanded"])
        self.assertEqual(len(result[1]["text"]), 1000)

    def test_respects_context_budget(self):
        expander = ArticleContextExpander(
            self.path,
            max_article_chars=6000,
            max_context_chars=2500,
        )

        result = expander.expand(self.chunks)

        self.assertEqual(len(result[0]["text"]), 1200)
        self.assertFalse(result[0]["context_expanded"])
        self.assertLessEqual(
            sum(len(item["text"]) for item in result),
            2500,
        )

    def test_original_chunks_not_mutated(self):
        expander = ArticleContextExpander(
            self.path,
            max_article_chars=6000,
            max_context_chars=8000,
        )

        expander.expand(self.chunks)

        self.assertEqual(len(self.chunks[0]["text"]), 1200)

    def test_expand_zero_preserves_chunks(self):
        expander = ArticleContextExpander(
            self.path,
            max_article_chars=6000,
            max_context_chars=8000,
        )

        result = expander.expand(self.chunks, expand_top_n=0)

        self.assertEqual(len(result[0]["text"]), 1200)
        self.assertFalse(result[0]["context_expanded"])


if __name__ == "__main__":
    unittest.main()
