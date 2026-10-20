import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.agents.evidence_agent import EvidenceAwareAgent
from src.retrieval.context_expansion import ArticleContextExpander


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, k=5):
        return self.chunks[:k]


class FakeCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({
                            "answer": "Answer supported by Article 5.",
                            "cited_articles": [5],
                            "abstained": False,
                        })
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=30,
                total_tokens=130,
            ),
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=FakeCompletions()
        )


class EvidenceAgentTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "articles.jsonl"

        article = {
            "article_number": 5,
            "title": "Prohibited practices",
            "text": "A" * 6000,
        }

        self.path.write_text(
            json.dumps(article) + "\n",
            encoding="utf-8",
        )

        self.chunks = [{
            "article_number": 5,
            "title": "Prohibited practices",
            "text": "A" * 1200,
            "start_char": 0,
            "end_char": 1200,
            "source_url": "https://example.org",
        }]

        self.agent = EvidenceAwareAgent(
            retriever=FakeRetriever(self.chunks),
            expander=ArticleContextExpander(
                articles_path=self.path,
                max_article_chars=15000,
                max_context_chars=22000,
            ),
            client=FakeClient(),
            model="fake-model",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_broad_question_triggers_expansion(self):
        result = self.agent.answer(
            "Which AI practices are prohibited?"
        )

        self.assertTrue(result["needs_expansion"])
        self.assertTrue(result["expansion_performed"])
        self.assertEqual(result["cited_articles"], [5])

    def test_narrow_question_skips_expansion(self):
        result = self.agent.answer(
            "Define a prohibited practice."
        )

        self.assertFalse(result["needs_expansion"])
        self.assertFalse(result["expansion_performed"])

    def test_graph_returns_usage(self):
        result = self.agent.answer(
            "Which AI practices are prohibited?"
        )

        self.assertEqual(result["total_tokens"], 130)
        self.assertEqual(result["provider"], "mistral")


if __name__ == "__main__":
    unittest.main()
