"""Interactive human review of four-way RAG benchmark answers.

No LLM calls. Uses existing results and gold evidence.
Review scores are saved incrementally to JSONL.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


STRATEGIES = (
    "naive",
    "expanded",
    "deterministic_langgraph",
    "llm_langgraph",
)


def load_jsonl(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def load_questions(path):
    questions = load_jsonl(path)

    return {
        question.get("id", question.get("question_id")): question
        for question in questions
    }


def review_key(question_id, strategy):
    return f"{question_id}::{strategy}"


def load_existing_reviews(path):
    if not path.exists():
        return {}

    reviews = load_jsonl(path)

    return {
        review_key(
            row["question_id"],
            row["strategy"],
        ): row
        for row in reviews
    }


def ask_score(label):
    while True:
        value = input(
            f"{label} (0-3, or s to skip): "
        ).strip().lower()

        if value == "s":
            return None

        if value in {"0", "1", "2", "3"}:
            return int(value)

        print("Enter 0, 1, 2, 3, or s.")


def display_gold(question):
    print("\n" + "=" * 90)

    print(
        "QUESTION:",
        question.get("id", question.get("question_id")),
    )

    print(question["question"])

    print(
        "\nCATEGORY:",
        question.get("category"),
    )

    print(
        "\nGOLD ARTICLES:",
        question.get("article_numbers", []),
    )

    reference = question.get(
        "reference_answer",
        question.get("gold_answer"),
    )

    if reference:
        print("\nREFERENCE ANSWER:")
        print(reference)

    passages = question.get(
        "supporting_passages", []
    )

    print(
        "\nGOLD SUPPORTING PASSAGES:",
        len(passages),
    )

    for index, passage in enumerate(
        passages,
        start=1,
    ):
        print(f"\n--- Evidence {index} ---")

        if isinstance(passage, dict):
            print(
                json.dumps(
                    passage,
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print(passage)


def display_result(row):
    print("\n" + "-" * 90)

    print(
        "STRATEGY:",
        row["strategy"],
    )

    print(
        "CITATIONS:",
        row.get("cited_articles"),
    )

    print(
        "TOKENS:",
        row.get("total_tokens"),
    )

    print(
        "LATENCY:",
        row.get("total_seconds"),
    )

    print(
        "ACTION:",
        row.get("action_taken"),
    )

    print("\nGENERATED ANSWER:")
    print(row.get("answer", ""))


def summarize(results, reviews):
    grouped = defaultdict(list)

    for row in results:
        if row.get("status") != "ok":
            continue

        key = review_key(
            row["question_id"],
            row["strategy"],
        )

        review = reviews.get(key)

        if review is None:
            continue

        grouped[row["strategy"]].append({
            **row,
            **review,
        })

    print("\n" + "=" * 90)
    print("HUMAN REVIEW SUMMARY")
    print("=" * 90)

    print(
        f"{'Strategy':<27}"
        f"{'Reviewed':>10}"
        f"{'Complete':>12}"
        f"{'Accuracy':>12}"
        f"{'Avg tokens':>14}"
    )

    for strategy in STRATEGIES:
        rows = grouped[strategy]

        if not rows:
            print(
                f"{strategy:<27}"
                f"{0:>10}"
                f"{'-':>12}"
                f"{'-':>12}"
                f"{'-':>14}"
            )
            continue

        completeness = sum(
            row["completeness"]
            for row in rows
        ) / len(rows)

        accuracy = sum(
            row["legal_accuracy"]
            for row in rows
        ) / len(rows)

        tokens = sum(
            row.get("total_tokens") or 0
            for row in rows
        ) / len(rows)

        print(
            f"{strategy:<27}"
            f"{len(rows):>10}"
            f"{completeness:>12.2f}"
            f"{accuracy:>12.2f}"
            f"{tokens:>14.1f}"
        )

    print(
        "\nNOTE: Human review scores are subjective."
    )

    print(
        "Scores should be checked against the verified"
        " supporting passages and full legal text."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        required=True,
    )

    parser.add_argument(
        "--questions",
        default="data/eval/questions.jsonl",
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
    )

    args = parser.parse_args()

    results = load_jsonl(
        args.results
    )

    questions = load_questions(
        args.questions
    )

    review_path = Path(
        str(Path(args.results).with_suffix(""))
        + ".human_reviews.jsonl"
    )

    reviews = load_existing_reviews(
        review_path
    )

    if args.summary_only:
        summarize(
            results,
            reviews,
        )
        return

    by_question = defaultdict(list)

    for row in results:
        if row.get("status") == "ok":
            by_question[
                row["question_id"]
            ].append(row)

    for question_id, rows in by_question.items():

        if question_id not in questions:
            raise ValueError(
                f"Missing gold question: {question_id}"
            )

        pending = [
            row
            for row in rows
            if review_key(
                question_id,
                row["strategy"],
            ) not in reviews
        ]

        if not pending:
            continue

        display_gold(
            questions[question_id]
        )

        for row in pending:
            display_result(row)

            completeness = ask_score(
                "Completeness"
            )

            if completeness is None:
                continue

            accuracy = ask_score(
                "Legal accuracy"
            )

            if accuracy is None:
                continue

            missing = input(
                "Missing information (optional): "
            ).strip()

            incorrect = input(
                "Incorrect information (optional): "
            ).strip()

            notes = input(
                "Reviewer notes (optional): "
            ).strip()

            review = {
                "question_id": question_id,
                "strategy": row["strategy"],
                "completeness": completeness,
                "legal_accuracy": accuracy,
                "missing_information": missing,
                "incorrect_information": incorrect,
                "reviewer_notes": notes,
            }

            with review_path.open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(
                        review,
                        ensure_ascii=False,
                    ) + "\n"
                )

                handle.flush()

            reviews[
                review_key(
                    question_id,
                    row["strategy"],
                )
            ] = review

            print("Saved review.")

    summarize(
        results,
        reviews,
    )

    print(
        "\nReview file:",
        review_path,
    )


if __name__ == "__main__":
    main()
