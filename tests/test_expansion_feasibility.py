import json
import tempfile
import unittest
from pathlib import Path

from src.agents.llm_evidence_agent import LLMEvidenceAgent
from src.retrieval.context_expansion import ArticleContextExpander


class ExpansionFeasibilityTests(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

        articles_path = Path(self.tempdir.name) / "articles.jsonl"

        articles = [
            {
                "article_number": 3,
                "title": "Definitions",
                "text": "A" * 17000,
            },
            {
                "article_number": 5,
                "title": "Prohibited AI practices",
                "text": "B" * 11000,
            },
        ]

        articles_path.write_text(
            "\n".join(json.dumps(article) for article in articles)
            + "\n",
            encoding="utf-8",
        )

        self.expander = ArticleContextExpander(
            articles_path=articles_path,
            max_article_chars=15000,
            max_context_chars=22000,
        )

        self.agent = LLMEvidenceAgent(
            retriever=None,
            expander=self.expander,
            client=None,
            model="test-model",
        )

        self.chunks = [
            {
                "article_number": 3,
                "title": "Definitions",
                "text": "A" * 1200,
                "start_char": 0,
                "end_char": 1200,
            },
            {
                "article_number": 5,
                "title": "Prohibited AI practices",
                "text": "B" * 1200,
                "start_char": 0,
                "end_char": 1200,
            },
        ]

    def tearDown(self):
        self.tempdir.cleanup()

    def test_large_article_cannot_expand(self):
        evidence = self.agent.build_evidence_metadata(
            self.chunks
        )

        article3 = next(
            item for item in evidence
            if item["article_number"] == 3
        )

        self.assertFalse(article3["can_expand"])
        self.assertEqual(
            article3["expansion_block_reason"],
            "article_exceeds_expansion_limit",
        )

    def test_small_article_can_expand(self):
        evidence = self.agent.build_evidence_metadata(
            self.chunks
        )

        article5 = next(
            item for item in evidence
            if item["article_number"] == 5
        )

        self.assertTrue(article5["can_expand"])

    def test_infeasible_expansion_falls_back_to_answer(self):
        result = self.agent._validate_assessment(
            {
                "action": "expand",
                "reason": "Need more evidence",
                "article_number": 3,
                "search_query": None,
            },
            self.chunks,
        )

        self.assertEqual(result["action"], "answer")
        self.assertIsNone(result["article_number"])
        self.assertIn(
            "Expansion unavailable",
            result["reason"],
        )


if __name__ == "__main__":
    unittest.main()
