import unittest

from src.agents.llm_evidence_agent import LLMEvidenceAgent


class TestInfeasibleExpansionFallback(unittest.TestCase):

    def test_infeasible_expansion_falls_back_to_answer(self):
        # Construct only the attributes required by
        # _validate_assessment, without making API calls.
        agent = object.__new__(LLMEvidenceAgent)

        class FakeExpander:
            articles = {
                3: {
                    "article_number": 3,
                    "text": "Long article",
                }
            }

        agent.expander = FakeExpander()

        agent.build_evidence_metadata = lambda chunks: [
            {
                "article_number": 3,
                "can_expand": False,
                "expansion_block_reason": (
                    "article_exceeds_expansion_limit"
                ),
            }
        ]

        assessment = agent._validate_assessment(
            {
                "action": "expand",
                "reason": "Need more evidence",
                "article_number": 3,
                "search_query": None,
            },
            [
                {
                    "article_number": 3,
                    "text": "Retrieved excerpt",
                }
            ],
        )

        self.assertEqual(
            assessment["action"],
            "answer",
        )

        self.assertIsNone(
            assessment["article_number"],
        )

        self.assertIn(
            "Expansion unavailable",
            assessment["reason"],
        )


if __name__ == "__main__":
    unittest.main()
