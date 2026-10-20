"""Offline regression tests for prompt-injection defense boundaries."""

import json
import unittest
from pathlib import Path

from src.generation.naive_rag import (
    SYSTEM_PROMPT,
    format_context,
    validate_response,
)


ATTACK_PATH = Path(
    "data/security/prompt_injections.jsonl"
)


def load_attacks():
    with ATTACK_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def make_chunk(payload):
    return {
        "article_number": 3,
        "title": "Definitions",
        "text": (
            "An AI system is a machine-based system "
            "designed to operate with varying levels "
            "of autonomy.\n\n"
            + payload
        ),
        "source_url": (
            "https://eur-lex.europa.eu/"
        ),
        "start_char": 0,
        "end_char": 1000,
    }


class PromptInjectionDefenseTests(unittest.TestCase):

    def test_attack_fixtures_are_unique(self):
        attacks = load_attacks()

        self.assertGreaterEqual(len(attacks), 6)

        identifiers = [
            attack["id"]
            for attack in attacks
        ]

        self.assertEqual(
            len(identifiers),
            len(set(identifiers)),
        )

    def test_context_preserves_untrusted_payload(self):
        """Confirm attack text reaches the evidence boundary.

        This is NOT a claim that the model resists the attack.
        """
        for attack in load_attacks():
            with self.subTest(attack=attack["id"]):
                context = format_context([
                    make_chunk(attack["payload"])
                ])

                self.assertIn(
                    attack["payload"],
                    context,
                )

    def test_system_prompt_requires_evidence_grounding(self):
        prompt = SYSTEM_PROMPT.lower()

        self.assertTrue(
            any(
                word in prompt
                for word in (
                    "context",
                    "evidence",
                    "retrieved",
                )
            )
        )

        self.assertTrue(
            any(
                word in prompt
                for word in (
                    "abstain",
                    "unsupported",
                    "cannot",
                )
            )
        )

    def test_unsupported_citation_is_rejected(self):
        """A retrieved Article 3 cannot authorize citing Article 999."""
        malicious_response = {
            "answer": "False legal claim [Article 999].",
            "cited_articles": [999],
            "abstained": False,
        }

        with self.assertRaises(Exception):
            validate_response(
                malicious_response,
                [3],
            )


if __name__ == "__main__":
    unittest.main()
