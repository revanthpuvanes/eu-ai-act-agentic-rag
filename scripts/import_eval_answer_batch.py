
import argparse
import json
from pathlib import Path

QUESTIONS_PATH = Path("data/eval/questions.jsonl")
ARTICLES_PATH = Path("data/processed/articles.jsonl")


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_file")
    args = parser.parse_args()

    batch_path = Path(args.batch_file)

    questions = load_jsonl(QUESTIONS_PATH)
    articles = {
        int(article["article_number"]): article
        for article in load_jsonl(ARTICLES_PATH)
    }

    question_lookup = {q["id"]: q for q in questions}
    answers = json.loads(batch_path.read_text(encoding="utf-8"))

    if not isinstance(answers, list) or not answers:
        raise ValueError("Batch must be a non-empty JSON array")

    seen = set()
    updates = {}

    # Validate the entire batch before changing questions.jsonl.
    for answer in answers:
        qid = answer["id"]

        if qid in seen:
            raise ValueError(f"Duplicate answer ID: {qid}")
        seen.add(qid)

        if qid not in question_lookup:
            raise ValueError(f"Unknown question: {qid}")

        question = question_lookup[qid]

        if question["verification_status"] == "verified":
            raise ValueError(f"Already verified: {qid}")

        expected_answer = answer["expected_answer"].strip()

        if not expected_answer:
            raise ValueError(f"Empty answer: {qid}")

        evidence = answer["evidence"]

        if not isinstance(evidence, list):
            raise ValueError(f"Evidence must be a list: {qid}")
        is_unanswerable = (
            question["category"] == "unanswerable"
            and question["article_numbers"] == []
        )
        if not evidence and not is_unanswerable:
            raise ValueError(f"Missing evidence: {qid}")
        if evidence and is_unanswerable:
            raise ValueError(
                f"{qid}: Unanswerable questions must not have "
                "fabricated evidence"
            )

        supporting_passages = []

        for item in evidence:
            article_number = int(item["article_number"])
            start = item["start"]
            end = item["end"]

            if article_number not in question["article_numbers"]:
                raise ValueError(
                    f"{qid}: Article {article_number} "
                    "is not referenced by the question"
                )

            if article_number not in articles:
                raise ValueError(
                    f"{qid}: Missing Article {article_number}"
                )

            article_text = articles[article_number]["text"]

            if not start or not end:
                raise ValueError(f"{qid}: Empty evidence marker")

            if article_text.count(start) != 1:
                raise ValueError(
                    f"{qid}: Start marker is not unique "
                    f"in Article {article_number}"
                )

            if article_text.count(end) != 1:
                raise ValueError(
                    f"{qid}: End marker is not unique "
                    f"in Article {article_number}"
                )

            start_index = article_text.index(start)
            end_index = article_text.index(end) + len(end)

            if start_index >= end_index:
                raise ValueError(
                    f"{qid}: Evidence markers are out of order"
                )

            passage = article_text[start_index:end_index]

            supporting_passages.append({
                "article_number": article_number,
                "text": passage,
            })

        updates[qid] = {
            "expected_answer": expected_answer,
            "supporting_passages": supporting_passages,
        }

    for question in questions:
        if question["id"] in updates:
            question.update(updates[question["id"]])
            # Evidence matching is not legal verification.
            question["verification_status"] = "pending"

    with QUESTIONS_PATH.open("w", encoding="utf-8") as file:
        for question in questions:
            file.write(json.dumps(question, ensure_ascii=False) + "\n")

    print("=== BATCH IMPORT COMPLETE ===")
    print("Answers imported:", len(updates))
    print("Evidence markers validated:", sum(
        len(answer["evidence"]) for answer in answers
    ))
    print("Verification status: pending (awaiting review)")


if __name__ == "__main__":
    main()
