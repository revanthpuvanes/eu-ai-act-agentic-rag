
import argparse
import json
from pathlib import Path

CORPUS = Path("data/processed/articles.jsonl")
EVAL = Path("data/eval/questions.jsonl")
ANSWERS_DIR = Path("data/eval/answers")


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

    answer_path = ANSWERS_DIR / f"{args.question_id}.json"

    if not answer_path.exists():
        raise FileNotFoundError(answer_path)

    answer_data = json.loads(
        answer_path.read_text(encoding="utf-8")
    )

    if answer_data["id"] != args.question_id:
        raise ValueError("Question ID mismatch")

    articles = {
        a["article_number"]: a
        for a in load_jsonl(CORPUS)
    }

    questions = load_jsonl(EVAL)

    matches = [
        q for q in questions
        if q["id"] == args.question_id
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one matching question"
        )

    question = matches[0]

    if question["article_numbers"] != answer_data["article_numbers"]:
        raise ValueError("Article references do not match")

    if len(answer_data["article_numbers"]) != 1:
        raise ValueError(
            "This importer currently supports one article"
        )

    article_number = answer_data["article_numbers"][0]
    article = articles[article_number]
    text = article["text"]

    start_marker = answer_data["evidence_markers"]["start"]
    end_marker = answer_data["evidence_markers"]["end"]

    if text.count(start_marker) != 1:
        raise ValueError("Start marker not unique")

    if text.count(end_marker) != 1:
        raise ValueError("End marker not unique")

    start = text.index(start_marker)
    end = text.index(end_marker) + len(end_marker)

    if end <= start:
        raise ValueError("Invalid evidence boundaries")

    evidence = text[start:end]

    if not answer_data["expected_answer"].strip():
        raise ValueError("Expected answer is empty")

    question["expected_answer"] = answer_data["expected_answer"]
    question["supporting_passages"] = [
        {
            "article_id": article["article_id"],
            "text": evidence,
        }
    ]
    question["verification_status"] = "verified"

    with EVAL.open("w", encoding="utf-8") as file:
        for item in questions:
            file.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    print(f"{args.question_id} imported successfully.")
    print("Evidence article:", article["article_id"])
    print("Evidence length:", len(evidence))
    print(
        "Verified questions:",
        sum(
            q["verification_status"] == "verified"
            for q in questions
        ),
    )


if __name__ == "__main__":
    main()
