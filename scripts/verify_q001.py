
import json
from pathlib import Path

EVAL_PATH = Path("data/eval/questions.jsonl")
CORPUS_PATH = Path("data/processed/articles.jsonl")

articles = [
    json.loads(line)
    for line in CORPUS_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

article_1 = next(
    article for article in articles
    if article["article_number"] == 1
)

questions = [
    json.loads(line)
    for line in EVAL_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

expected_answer = (
    "The EU AI Act establishes harmonised rules for "
    "placing AI systems on the market, putting them "
    "into service, and using them in the Union. "
    "It promotes human-centric and trustworthy AI "
    "while protecting health, safety, fundamental "
    "rights, democracy, the rule of law, and the "
    "environment. It covers prohibited AI practices, "
    "high-risk AI systems, transparency requirements, "
    "general-purpose AI models, governance, enforcement, "
    "and measures supporting innovation."
)

matches = [
    q for q in questions if q["id"] == "Q001"
]

if len(matches) != 1:
    raise ValueError("Expected exactly one Q001 record")

if article_1["title"] != "Subject matter":
    raise ValueError("Unexpected Article 1 title")

for question in questions:
    if question["id"] == "Q001":
        question["expected_answer"] = expected_answer

        question["supporting_passages"] = [
            {
                "article_id": "Article 1",
                "text": article_1["text"],
            }
        ]

        question["verification_status"] = "verified"

with EVAL_PATH.open("w", encoding="utf-8") as file:
    for question in questions:
        file.write(
            json.dumps(question, ensure_ascii=False)
            + "\n"
        )

print("Q001 verified and saved.")
print("Evidence: Article 1")
print("Evidence length:", len(article_1["text"]))
