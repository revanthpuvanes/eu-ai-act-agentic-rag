import unittest

from src.retrieval.legal_router import detect_relevant_articles


class LegalRouterTests(unittest.TestCase):
    def test_application_question(self):
        self.assertIn(
            113,
            detect_relevant_articles(
                "When does the EU AI Act apply?"
            ),
        )

    def test_entry_into_force(self):
        self.assertIn(
            113,
            detect_relevant_articles(
                "What is the entry into force date?"
            ),
        )

    def test_unrelated_question(self):
        self.assertEqual(
            detect_relevant_articles(
                "What is a high-risk AI system?"
            ),
            set(),
        )


if __name__ == "__main__":
    unittest.main()
