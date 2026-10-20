"""Prepare evidence-grounded human review of RAG answers."""

import argparse
import csv
import json
from pathlib import Path


def load_jsonl(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        required=True,
        help="Four-way benchmark JSONL file",
    )

    parser.add_argument(
        "--questions",
        default="data/eval/questions.jsonl",
    )

    args = parser.parse_args()

    results = load_jsonl(args.results)
    questions = load_jsonl(args.questions)

    question_lookup = {
        question.get("id", question.get("question_id")): question
        for question in questions
    }

    output_path = Path(args.results).with_suffix(
        ".quality_review.csv"
    )

    fields = [
        "question_id",
        "category",
        "strategy",
        "question",
        "reference_answer",
        "gold_articles",
        "gold_supporting_passages",
        "generated_answer",
        "cited_articles",
        "action_taken",
        "completeness_0_to_3",
        "legal_accuracy_0_to_3",
        "missing_information",
        "incorrect_information",
        "reviewer_notes",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for result in results:
            if result.get("status") != "ok":
                continue

            question_id = result["question_id"]

            if question_id not in question_lookup:
                raise ValueError(
                    f"Unknown question ID: {question_id}"
                )

            gold = question_lookup[question_id]

            row = {
                "question_id": question_id,
                "category": gold.get("category"),
                "strategy": result["strategy"],
                "question": gold["question"],
                "reference_answer": gold.get(
                    "reference_answer",
                    gold.get("gold_answer", ""),
                ),
                "gold_articles": json.dumps(
                    gold.get("article_numbers", []),
                    ensure_ascii=False,
                ),
                "gold_supporting_passages": json.dumps(
                    gold.get("supporting_passages", []),
                    ensure_ascii=False,
                ),
                "generated_answer": result.get(
                    "answer", ""
                ),
                "cited_articles": json.dumps(
                    result.get("cited_articles", []),
                    ensure_ascii=False,
                ),
                "action_taken": result.get(
                    "action_taken", ""
                ),
                "completeness_0_to_3": "",
                "legal_accuracy_0_to_3": "",
                "missing_information": "",
                "incorrect_information": "",
                "reviewer_notes": "",
            }

            writer.writerow(row)

    print(f"Created: {output_path}")
    print(f"Review rows: {sum(r.get('status') == 'ok' for r in results)}")
    print("Legal correctness has NOT been scored.")


if __name__ == "__main__":
    main()
