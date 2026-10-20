import unittest
from contextlib import contextmanager
from unittest.mock import patch

from src.observability.langfuse_tracing import (
    trace_agent_run,
)


class FakeAgent:
    def __init__(self):
        self.calls = 0

    def run(self, question):
        self.calls += 1

        return {
            "question": question,
            "answer": "The definition is in Article 3.",
            "cited_articles": [3],
            "retrieved_articles": [3, 2],
            "abstained": False,
            "assessment": {
                "action": "answer",
                "reason": "Evidence is sufficient",
            },
            "action_taken": "answer",
            "total_tokens": 100,
            "total_seconds": 0.5,
            "model": "test-model",
            "provider": "test-provider",
        }


class FakeSpan:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeLangfuse:
    def __init__(self):
        self.span = FakeSpan()
        self.started = []

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        self.started.append(kwargs)
        yield self.span


class LangfuseTracingTests(unittest.TestCase):

    def test_disabled_tracing(self):
        agent = FakeAgent()

        result = trace_agent_run(
            agent,
            "Define an AI system",
            langfuse_client=None,
            enabled=False,
        )

        self.assertEqual(agent.calls, 1)
        self.assertEqual(result["cited_articles"], [3])

    def test_enabled_tracing(self):
        agent = FakeAgent()
        client = FakeLangfuse()

        with patch(
            "langfuse.propagate_attributes",
            return_value=contextmanager(
                lambda: (yield)
            )(),
        ):
            result = trace_agent_run(
                agent,
                "Define an AI system",
                client,
                question_id="Q002",
            )

        self.assertEqual(agent.calls, 1)
        self.assertEqual(
            client.started[0]["name"],
            "eu-ai-act-llm-agent",
        )

        recorded = client.span.updates[0]

        self.assertEqual(
            recorded["metadata"]["assessment_action"],
            "answer",
        )

        self.assertEqual(
            recorded["metadata"]["total_tokens"],
            100,
        )

        self.assertEqual(
            recorded["output"]["cited_articles"],
            [3],
        )


if __name__ == "__main__":
    unittest.main()
