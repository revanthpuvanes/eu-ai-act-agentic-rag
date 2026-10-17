
import json
from collections import Counter
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
    questions = load_jsonl(QUESTIONS_PATH)
    articles = {
        int(article["article_number"]): article["text"]
        for article in load_jsonl(ARTICLES_PATH)
    }

    errors = []
    warnings = []

    expected_ids = {f"Q{i:03d}" for i in range(1, 61)}
    actual_ids = [q["id"] for q in questions]

    if len(questions) != 60:
        errors.append(f"Expected 60 questions, found {len(questions)}")

    if set(actual_ids) != expected_ids:
        errors.append("Question IDs do not match Q001-Q060")

    if len(actual_ids) != len(set(actual_ids)):
        errors.append("Duplicate question IDs found")

    expected_categories = {
        "single_hop": 20,
        "multi_hop": 15,
        "exceptions": 10,
        "definitions": 5,
        "unanswerable": 5,
        "temporal": 5,
    }

    categories = Counter(q["category"] for q in questions)
    splits = Counter(q["split"] for q in questions)
    statuses = Counter(q["verification_status"] for q in questions)

    if dict(categories) != expected_categories:
        errors.append(f"Unexpected category counts: {dict(categories)}")

    if splits != Counter({"dev": 20, "test": 40}):
        errors.append(f"Unexpected split counts: {dict(splits)}")

    evidence_count = 0

    for q in questions:
        qid = q["id"]
        answer = q.get("expected_answer", "")
        passages = q.get("supporting_passages", [])
        referenced = q.get("article_numbers", [])

        if not isinstance(answer, str) or not answer.strip():
            errors.append(f"{qid}: missing expected answer")

        if not isinstance(passages, list):
            errors.append(f"{qid}: supporting_passages is not a list")
            continue

        if q["category"] == "unanswerable":
            if referenced:
                errors.append(
                    f"{qid}: unanswerable question references articles"
                )
            if passages:
                errors.append(
                    f"{qid}: unanswerable question has evidence passages"
                )
        elif not passages:
            errors.append(f"{qid}: missing evidence passages")

        for passage in passages:
            evidence_count += 1
            article_number = passage.get("article_number")
            text = passage.get("text", "")

            if article_number not in referenced:
                errors.append(
                    f"{qid}: evidence Article {article_number} "
                    "not referenced by question"
                )
                continue

            if article_number not in articles:
                errors.append(
                    f"{qid}: Article {article_number} missing from corpus"
                )
                continue

            if not text or text not in articles[article_number]:
                errors.append(
                    f"{qid}: evidence passage not found in "
                    f"Article {article_number}"
                )

        if q["verification_status"] == "pending":
            warnings.append(f"{qid}: awaits substantive human review")

    print("=== EU AI ACT DATASET AUDIT ===")
    print("Questions:", len(questions))
    print("Categories:", dict(sorted(categories.items())))
    print("Splits:", dict(sorted(splits.items())))
    print("Verification statuses:", dict(sorted(statuses.items())))
    print("Evidence passages:", evidence_count)
    print("Structural errors:", len(errors))
    print("Review warnings:", len(warnings))

    if errors:
        print("\nERRORS:")
        for error in errors:
            print(" -", error)

    if warnings:
        print("\nReview pending for:", len(warnings), "questions")

    if errors:
        raise SystemExit(1)

    print("\nPASS: All structural checks passed.")
    print("NOTE: Legal answer accuracy still requires review.")

if __name__ == "__main__":
    main()
