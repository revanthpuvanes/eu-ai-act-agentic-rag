import unittest
from unittest.mock import patch

from scripts.compare_generation_strategies import (
    run_strategy,
    summarize_strategy,
)


class PairedGenerationTests(unittest.TestCase):

    def setUp(self):
        self.question = {
            "id": "Q_TEST",
            "question": "Which practices are prohibited?",
            "category": "single_hop",
            "split": "dev",
            "article_numbers": [5],
        }

    @patch(
        "scripts.compare_generation_strategies.answer_question"
    )
    def test_successful_generation(self, mock_answer):
        mock_answer.return_value = {
            "answer": "Certain practices are prohibited.",
            "cited_articles": [5],
            "retrieved_articles": [5, 1],
            "abstained": False,
            "retrieval_seconds": 0.1,
            "generation_seconds": 0.5,
            "total_seconds": 0.6,
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "provider": "mistral",
        }

        result = run_strategy(
            question=self.question,
            strategy_name="naive",
            retriever=object(),
            client=object(),
            model="fake-model",
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["citation_f1"], 1.0)
        self.assertTrue(result["abstention_correct"])
        self.assertIsNone(result["estimated_cost_usd"])

    @patch(
        "scripts.compare_generation_strategies.answer_question"
    )
    def test_failure_is_recorded(self, mock_answer):
        mock_answer.side_effect = RuntimeError("Simulated 429")

        result = run_strategy(
            question=self.question,
            strategy_name="expanded",
            retriever=object(),
            client=object(),
            model="fake-model",
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Simulated 429", result["error"])

    def test_summary_counts_failures(self):
        results = [
            {
                "status": "ok",
                "expected_abstention": False,
                "citation_precision": 1.0,
                "citation_recall": 1.0,
                "citation_f1": 1.0,
                "abstention_correct": True,
                "total_seconds": 1.5,
                "input_tokens": 100,
                "output_tokens": 50,
                "estimated_cost_usd": None,
            },
            {
                "status": "error",
                "error": "Simulated failure",
            },
        ]

        summary = summarize_strategy(results)

        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(summary["succeeded"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertIsNone(
            summary["estimated_total_cost_usd"]
        )


if __name__ == "__main__":
    unittest.main()
