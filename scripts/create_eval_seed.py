
import json
from pathlib import Path

CORPUS = Path("data/processed/articles.jsonl")
OUTPUT = Path("data/eval/questions.jsonl")

articles = {}

with CORPUS.open(encoding="utf-8") as file:
    for line in file:
        article = json.loads(line)
        articles[article["article_number"]] = article

candidates = [
    {
        "id": "Q001",
        "question": "What is the subject matter of the EU AI Act?",
        "category": "single_hop",
        "article_numbers": [1],
    },
    {
        "id": "Q002",
        "question": "How does the EU AI Act define an AI system?",
        "category": "single_hop",
        "article_numbers": [3],
    },
    {
        "id": "Q003",
        "question": "Which AI practices are prohibited under the EU AI Act?",
        "category": "single_hop",
        "article_numbers": [5],
    },
    {
        "id": "Q004",
        "question": "What obligations apply to providers of high-risk AI systems?",
        "category": "single_hop",
        "article_numbers": [16],
    },
    {
        "id": "Q005",
        "question": "What transparency obligations apply to certain AI systems?",
        "category": "single_hop",
        "article_numbers": [50],
    },
]

if OUTPUT.exists() and OUTPUT.stat().st_size > 0:
    raise SystemExit(
        "Evaluation file already contains data. "
        "Refusing to overwrite it."
    )

with OUTPUT.open("w", encoding="utf-8") as file:
    for candidate in candidates:
        for number in candidate["article_numbers"]:
            if number not in articles:
                raise ValueError(
                    f"Missing Article {number}"
                )

        record = {
            **candidate,
            "expected_answer": None,
            "supporting_passages": [],
            "verification_status": "pending",
            "split": None,
        }

        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

print(f"Created {len(candidates)} candidate questions")
print(f"Saved to {OUTPUT}")
