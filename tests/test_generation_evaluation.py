import unittest

from src.evaluation.evaluate_generation import (
    citation_metrics,
    estimate_cost_usd,
    evaluate_one,
    summarize,
)


class FakeRetriever:
    pass


class GenerationEvaluationTests(unittest.TestCase):

    def test_perfect_citations(self):
        result = citation_metrics([5, 6], [5, 6])

        self.assertEqual(result["citation_precision"], 1.0)
        self.assertEqual(result["citation_recall"], 1.0)
        self.assertEqual(result["citation_f1"], 1.0)

    def test_partial_citations(self):
        result = citation_metrics([5, 9], [5, 6])

        self.assertEqual(result["citation_precision"], 0.5)
        self.assertEqual(result["citation_recall"], 0.5)
        self.assertEqual(result["citation_f1"], 0.5)

    def test_no_citations(self):
        result = citation_metrics([], [5])

        self.assertEqual(result["citation_precision"], 0.0)
        self.assertEqual(result["citation_recall"], 0.0)

    def test_unanswerable_citation_metrics_not_applicable(self):
        result = citation_metrics([], [])

        self.assertIsNone(result["citation_precision"])
        self.assertIsNone(result["citation_recall"])

    def test_cost_calculation(self):
        result = estimate_cost_usd(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            input_price_per_million=1.0,
            output_price_per_million=3.0,
        )

        self.assertEqual(result, 4.0)

    def test_missing_token_usage(self):
        result = estimate_cost_usd(
            input_tokens=None,
            output_tokens=100,
            input_price_per_million=1.0,
            output_price_per_million=3.0,
        )

        self.assertIsNone(result)

    def test_api_failure_is_recorded(self):
        question = {
            "id": "Q_TEST",
            "category": "single_hop",
            "split": "dev",
            "question": "Test question?",
            "article_numbers": [5],
        }

        # Passing None as the client should fail inside the
        # actual generation function, without making an API call.
        result = evaluate_one(
            question=question,
            retriever=FakeRetriever(),
            client=None,
            model="fake-model",
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)

    def test_summary_counts_failures(self):
        results = [
            {
                "status": "error",
                "error": "Rate limit",
            },
            {
                "status": "ok",
                "expected_abstention": False,
                "citation_precision": 1.0,
                "citation_recall": 0.5,
                "citation_f1": 0.6667,
                "abstention_correct": True,
                "total_seconds": 2.0,
                "input_tokens": 100,
                "output_tokens": 50,
                "estimated_cost_usd": 0.001,
            },
        ]

        summary = summarize(results)

        self.assertEqual(summary["questions_attempted"], 2)
        self.assertEqual(summary["questions_succeeded"], 1)
        self.assertEqual(summary["questions_failed"], 1)
        self.assertEqual(summary["mean_citation_recall"], 0.5)
        self.assertFalse(summary["semantic_answer_quality_evaluated"])


if __name__ == "__main__":
    unittest.main()
