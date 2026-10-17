
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
    parser = argparse.ArgumentParser()
    parser.add_argument("question_id")
    args = parser.parse_args()

    questions = load_jsonl(EVAL)
    articles = {
        article["article_number"]: article
        for article in load_jsonl(CORPUS)
    }

    matches = [
        q for q in questions
        if q["id"] == args.question_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {args.question_id}"
        )

    if args.question_id == "Q002":
        article = articles[3]
        body = article["text"]

        start_marker = "(1) ‘AI system’ means"
        end_marker = "(2) ‘risk’ means"

        start = body.find(start_marker)
        end = body.find(end_marker)

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "Could not locate exact AI system definition"
            )

        evidence = body[start:end].strip()

        expected_answer = (
            "Under Article 3(1), an AI system is a "
            "machine-based system designed to operate "
            "with varying levels of autonomy and which "
            "may exhibit adaptiveness after deployment. "
            "For explicit or implicit objectives, it "
            "infers from its inputs how to generate "
            "outputs such as predictions, content, "
            "recommendations, or decisions that can "
            "influence physical or virtual environments."
        )

        for question in questions:
            if question["id"] == "Q002":
                question["expected_answer"] = expected_answer
                question["supporting_passages"] = [
                    {
                        "article_id": "Article 3",
                        "text": evidence,
                    }
                ]
                question["verification_status"] = "verified"

    else:
        raise ValueError(
            "No verified evidence configuration yet "
            f"for {args.question_id}"
        )

    # Validate before writing.
    ids = [q["id"] for q in questions]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation question IDs")

    for question in questions:
        if question["verification_status"] == "verified":
            if not question["expected_answer"]:
                raise ValueError(
                    f"{question['id']}: missing answer"
                )

            for passage in question["supporting_passages"]:
                if not any(
                    passage["text"] in article["text"]
                    and passage["article_id"] == article["article_id"]
                    for article in articles.values()
                ):
                    raise ValueError(
                        f"{question['id']}: evidence not found "
                        "in source corpus"
                    )

    with EVAL.open("w", encoding="utf-8") as file:
        for question in questions:
            file.write(
                json.dumps(question, ensure_ascii=False)
                + "\n"
            )

    print(f"{args.question_id} verified and saved.")
    print("Evidence article: Article 3")
    print("Evidence length:", len(evidence))
    print("Total questions:", len(questions))
    print(
        "Verified questions:",
        sum(
            q["verification_status"] == "verified"
            for q in questions
        ),
    )


if __name__ == "__main__":
    main()
