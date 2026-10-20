import unittest
from contextlib import contextmanager
from unittest.mock import patch

from src.agents.llm_evidence_agent import LLMEvidenceAgent
from src.observability.traced_agent import TracedLLMEvidenceAgent


class FakeObservation:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeLangfuse:
    def __init__(self):
        self.started = []
        self.observations = []

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        self.started.append(kwargs)
        observation = FakeObservation()
        self.observations.append(observation)
        yield observation


class TracedAgentTests(unittest.TestCase):

    def test_retrieve_calls_original_once(self):
        client = FakeLangfuse()

        agent = object.__new__(TracedLLMEvidenceAgent)
        agent.langfuse_client = client

        chunks = [
            {
                "article_number": 3,
                "text": "Definition of an AI system",
            }
        ]

        with patch.object(
            LLMEvidenceAgent,
            "retrieve",
            return_value={"chunks": chunks},
        ) as original:
            result = agent.retrieve({"question": "Define AI"})

        original.assert_called_once_with(
            {"question": "Define AI"}
        )

        self.assertEqual(result["chunks"], chunks)
        self.assertEqual(
            client.started[0]["name"],
            "retrieve",
        )
        self.assertEqual(
            client.started[0]["as_type"],
            "retriever",
        )

    def test_assess_calls_original_once(self):
        client = FakeLangfuse()

        agent = object.__new__(TracedLLMEvidenceAgent)
        agent.langfuse_client = client

        assessment = {
            "action": "answer",
            "reason": "Sufficient evidence",
        }

        with patch.object(
            LLMEvidenceAgent,
            "assess",
            return_value={"assessment": assessment},
        ) as original:
            result = agent.assess({"question": "Define AI"})

        original.assert_called_once()

        self.assertEqual(
            result["assessment"]["action"],
            "answer",
        )

        self.assertEqual(
            client.observations[0].updates[0]["metadata"]["action"],
            "answer",
        )


if __name__ == "__main__":
    unittest.main()
