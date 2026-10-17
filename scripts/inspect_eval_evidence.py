
import argparse
import json
from pathlib import Path

CORPUS = Path("data/processed/articles.jsonl")
EVAL = Path("data/eval/questions.jsonl")


def load_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    parser = argparse.ArgumentParser(
        description="Inspect evidence for an evaluation question."
    )
    parser.add_argument(
        "question_id",
        help="Question ID, e.g. Q002",
    )
    args = parser.parse_args()

    questions = load_jsonl(EVAL)
    articles = {
        article["article_number"]: article
        for article in load_jsonl(CORPUS)
    }

    matches = [
        question
        for question in questions
        if question["id"] == args.question_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one question with ID "
            f"{args.question_id}; found {len(matches)}"
        )

    question = matches[0]

    print("=" * 60)
    print("QUESTION:", question["id"])
    print("TEXT:", question["question"])
    print("CATEGORY:", question["category"])
    print("STATUS:", question["verification_status"])
    print("=" * 60)

    for number in question["article_numbers"]:
        if number not in articles:
            raise ValueError(f"Article {number} not found")

        article = articles[number]

        print(f"\nARTICLE {number}: {article['title']}")
        print("-" * 60)
        print(article["text"])
        print("-" * 60)


if __name__ == "__main__":
    main()
