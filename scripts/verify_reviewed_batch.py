
import argparse
import hashlib
import json
import shutil
from pathlib import Path

QUESTIONS_PATH = Path("data/eval/questions.jsonl")
ARTICLES_PATH = Path("data/processed/articles.jsonl")
REVIEW_DIR = Path("data/eval/reviews")


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def answer_digest(question):
    payload = {
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "article_numbers": question["article_numbers"],
        "supporting_passages": question["supporting_passages"],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ids",
        nargs="+",
        required=True,
        help="Question IDs approved after review",
    )
    parser.add_argument(
        "--reviewer",
        required=True,
        help="Reviewer identifier",
    )
    args = parser.parse_args()

    ids = args.ids

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs")

    questions = load_jsonl(QUESTIONS_PATH)
    articles = {
        int(a["article_number"]): a["text"]
        for a in load_jsonl(ARTICLES_PATH)
    }
    lookup = {q["id"]: q for q in questions}

    for qid in ids:
        if qid not in lookup:
            raise ValueError(f"Unknown question: {qid}")

        q = lookup[qid]

        if q["verification_status"] != "pending":
            raise ValueError(f"{qid} is not pending")

        if not q["expected_answer"].strip():
            raise ValueError(f"{qid} has no answer")

        if q["category"] == "unanswerable":
            if q["article_numbers"] != []:
                raise ValueError(
                    f"{qid}: unanswerable question must have no article references"
                )
            if q["supporting_passages"] != []:
                raise ValueError(
                    f"{qid}: unanswerable question must have no supporting passages"
                )
        elif not q["supporting_passages"]:
            raise ValueError(f"{qid} has no evidence")

        for passage in q["supporting_passages"]:
            number = passage["article_number"]
            text = passage["text"]

            if number not in q["article_numbers"]:
                raise ValueError(
                    f"{qid}: Article {number} not referenced"
                )

            if number not in articles:
                raise ValueError(
                    f"{qid}: Article {number} missing"
                )

            if text not in articles[number]:
                raise ValueError(
                    f"{qid}: evidence does not match corpus"
                )

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    review_file = REVIEW_DIR / (
        f"verified_{ids[0]}_{ids[-1]}.json"
    )

    if review_file.exists():
        raise FileExistsError(
            f"Review record already exists: {review_file}"
        )

    review_record = {
        "reviewer": args.reviewer,
        "review_scope": "source-grounded answer review",
        "questions": [
            {
                "id": qid,
                "answer_sha256": answer_digest(lookup[qid]),
                "decision": "verified",
            }
            for qid in ids
        ],
    }

    backup = QUESTIONS_PATH.with_name(
        f"questions.before_verification_{ids[0]}_{ids[-1]}.jsonl"
    )

    if backup.exists():
        raise FileExistsError(
            f"Backup already exists: {backup}"
        )

    shutil.copy2(QUESTIONS_PATH, backup)

    for qid in ids:
        lookup[qid]["verification_status"] = "verified"

    with QUESTIONS_PATH.open("w", encoding="utf-8") as file:
        for q in questions:
            file.write(
                json.dumps(q, ensure_ascii=False) + "\n"
            )

    review_file.write_text(
        json.dumps(
            review_record,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("=== BATCH VERIFICATION COMPLETE ===")
    print("Verified questions:", len(ids))
    print("Review record:", review_file)
    print("Backup:", backup)
    print("Question splits unchanged.")


if __name__ == "__main__":
    main()
