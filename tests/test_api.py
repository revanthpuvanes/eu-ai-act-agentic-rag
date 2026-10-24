import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import app


class FakeGraph:
    def invoke(self, initial_state):
        return self.final_state(initial_state["question"])

    def stream(self, initial_state, stream_mode="updates"):
        assert stream_mode == "updates"

        yield {
            "retrieve": {
                "retrieval_seconds": 0.01,
            }
        }

        yield {
            "assess": {
                "assessment": {
                    "action": "answer",
                    "reason": "Evidence is sufficient",
                },
                "assessor_input_tokens": 20,
                "assessor_output_tokens": 10,
            }
        }

        yield {
            "answer": {
                **self.final_state(initial_state["question"]),
            }
        }

    @staticmethod
    def final_state(question):
        return {
            "question": question,
            "answer": "Article 5 covers prohibited AI practices.",
            "cited_articles": [5],
            "abstained": False,
            "final_chunks": [
                {"article_number": 5},
            ],
            "assessment": {
                "action": "answer",
                "reason": "Evidence is sufficient",
            },
            "action_taken": "answer",
            "assessor_input_tokens": 20,
            "assessor_output_tokens": 10,
            "generation_input_tokens": 100,
            "generation_output_tokens": 50,
            "retrieval_seconds": 0.01,
            "assessment_seconds": 0.02,
            "evidence_action_seconds": 0,
            "generation_seconds": 0.03,
        }


class FakeAgent:
    model = "test-model"
    graph = FakeGraph()


class APITests(unittest.TestCase):
    def setUp(self):
        self.agent_patch = patch(
            "src.api.app.build_agent",
            return_value=FakeAgent(),
        )
        self.agent_patch.start()

        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.agent_patch.stop()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_ask(self):
        response = self.client.post(
            "/ask",
            json={"question": "What does Article 5 prohibit?"},
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["cited_articles"], [5])
        self.assertEqual(data["action_taken"], "answer")
        self.assertEqual(data["total_tokens"], 180)

    def test_short_question_rejected(self):
        response = self.client.post(
            "/ask",
            json={"question": "Hello"},
        )
        self.assertEqual(response.status_code, 422)

    def test_stream(self):
        response = self.client.post(
            "/ask/stream",
            json={"question": "What does Article 5 prohibit?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: started", response.text)
        self.assertIn("event: node_completed", response.text)
        self.assertIn("event: completed", response.text)
        self.assertIn('"cited_articles": [5]', response.text)


if __name__ == "__main__":
    unittest.main()
