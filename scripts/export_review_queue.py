
import json
from collections import Counter
from pathlib import Path

QUESTIONS_PATH = Path("data/eval/questions.jsonl")
OUTPUT_PATH = Path("data/eval/review_queue.md")

def main():
    questions = [
        json.loads(line)
        for line in QUESTIONS_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    pending = [
        q for q in questions
        if q["verification_status"] == "pending"
    ]

    lines = [
        "# EU AI Act Benchmark — Human Review Queue",
        "",
        f"Questions awaiting review: {len(pending)}",
        "",
        "Review each answer for legal accuracy, completeness, "
        "citation relevance, and unsupported claims.",
        "",
        "Do not mark an answer verified until its content "
        "has been checked against the official corpus.",
        "",
    ]

    for q in pending:
        lines.extend([
            "---",
            "",
            f"## {q['id']} — {q['category']}",
            "",
            f"**Split:** {q['split']}",
            "",
            f"**Question:** {q['question']}",
            "",
            "### Draft answer",
            "",
            q["expected_answer"],
            "",
            "### Supporting evidence",
            "",
        ])

        passages = q["supporting_passages"]

        if not passages:
            lines.extend([
                "No supporting passages. "
                "Check whether abstention is appropriate.",
                "",
            ])

        for index, passage in enumerate(passages, 1):
            lines.extend([
                f"#### Evidence {index} — "
                f"Article {passage['article_number']}",
                "",
                passage["text"],
                "",
            ])

        lines.extend([
            "### Review decision",
            "",
            "- [ ] Answer is legally accurate",
            "- [ ] Answer is complete",
            "- [ ] Evidence supports the answer",
            "- [ ] No unsupported claims",
            "",
            "**Decision:** PENDING",
            "",
            "**Reviewer notes:**",
            "",
        ])

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("=== REVIEW QUEUE EXPORTED ===")
    print("Pending questions:", len(pending))
    print("Categories:", dict(Counter(
        q["category"] for q in pending
    )))
    print("Output:", OUTPUT_PATH)
    print("Dataset unchanged.")

if __name__ == "__main__":
    main()
