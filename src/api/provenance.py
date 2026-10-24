"""Format evidence actually supplied to the agent's generation stage."""

from typing import Any


def format_evidence(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = []

    for index, chunk in enumerate(chunks):
        text = chunk.get("text", "")

        if not isinstance(text, str) or not text.strip():
            continue

        evidence.append(
            {
                "id": index,
                "article_number": int(chunk["article_number"]),
                "text": text,
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
                "source_url": chunk.get("source_url"),
            }
        )

    return evidence
