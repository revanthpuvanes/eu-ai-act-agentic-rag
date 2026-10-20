import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.agents.llm_evidence_agent import (
    LLMEvidenceAgent,
)
from src.retrieval.context_expansion import (
    ArticleContextExpander,
)


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def search(self, query, k=5):
        self.calls.append(query)
        return self.chunks[:k]


class FakeCompletions:
    def __init__(self, action):
        self.action = action
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1

        if self.calls == 1:
            content = {
                "action": self.action,
                "reason": "Test decision",
                "article_number": (
                    5 if self.action == "expand" else None
                ),
                "search_query": (
                    "biometric identification"
                    if self.action == "search"
                    else None
                ),
            }
        else:
            content = {
                "answer": "Supported by Article 5.",
                "cited_articles": [5],
                "abstained": False,
            }

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(content)
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
    def __init__(self, action):
        self.completions = FakeCompletions(action)
        self.chat = SimpleNamespace(
            completions=self.completions
        )


class LLMEvidenceAgentTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

        path = (
            Path(self.tempdir.name) / "articles.jsonl"
        )

        path.write_text(
            json.dumps({
                "article_number": 5,
                "title": "Prohibited practices",
                "text": "A" * 6000,
            }) + "\n",
            encoding="utf-8",
        )

        self.expander = ArticleContextExpander(
            articles_path=path,
        )

        self.chunks = [{
            "article_number": 5,
            "title": "Prohibited practices",
            "text": "A" * 1200,
            "start_char": 0,
            "end_char": 1200,
            "source_url": "https://example.org",
        }]

    def tearDown(self):
        self.tempdir.cleanup()

    def build(self, action):
        client = FakeClient(action)
        retriever = FakeRetriever(self.chunks)

        agent = LLMEvidenceAgent(
            retriever=retriever,
            expander=self.expander,
            client=client,
            model="fake-model",
        )

        return agent, client, retriever

    def test_answer_action(self):
        agent, client, _ = self.build("answer")

        result = agent.answer(
            "What does Article 5 prohibit?"
        )

        self.assertEqual(
            result["action_taken"], "answer"
        )
        self.assertEqual(
            client.completions.calls, 2
        )
        self.assertEqual(result["total_tokens"], 260)

    def test_expand_action(self):
        agent, client, _ = self.build("expand")

        result = agent.answer(
            "Which practices are prohibited?"
        )

        self.assertEqual(
            result["action_taken"], "expand"
        )
        self.assertEqual(
            client.completions.calls, 2
        )

    def test_search_action(self):
        agent, client, retriever = self.build(
            "search"
        )

        result = agent.answer(
            "What are the biometric rules?"
        )

        self.assertEqual(
            result["action_taken"], "search"
        )
        self.assertEqual(
            len(retriever.calls), 2
        )

    def test_abstain_action(self):
        agent, client, _ = self.build(
            "abstain"
        )

        result = agent.answer(
            "What is the weather tomorrow?"
        )

        self.assertTrue(result["abstained"])
        self.assertEqual(
            client.completions.calls, 1
        )

    def test_rejects_unknown_article(self):
        agent, _, _ = self.build("answer")

        with self.assertRaises(ValueError):
            agent._validate_assessment(
                {
                    "action": "expand",
                    "reason": "Need more context",
                    "article_number": 999,
                    "search_query": None,
                },
                self.chunks,
            )


if __name__ == "__main__":
    unittest.main()
