
import argparse
import json
from pathlib import Path

CORPUS = Path("data/processed/articles.jsonl")
EVAL = Path("data/eval/questions.jsonl")


def load_jsonl(path):
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_file")
    args = parser.parse_args()

    batch_path = Path(args.batch_file)

    if not batch_path.exists():
        raise FileNotFoundError(batch_path)

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    existing = load_jsonl(EVAL)
    articles = load_jsonl(CORPUS)

    existing_ids = {q["id"] for q in existing}
    article_numbers = {a["article_number"] for a in articles}

    if not isinstance(batch, list) or not batch:
        raise ValueError("Batch must be a non-empty list")

    batch_ids = [q["id"] for q in batch]

    if len(batch_ids) != len(set(batch_ids)):
        raise ValueError("Duplicate IDs within batch")

    duplicates = existing_ids.intersection(batch_ids)

    if duplicates:
        raise ValueError(f"Questions already exist: {sorted(duplicates)}")

    new_questions = []

    for item in batch:
        if not item["question"].strip():
            raise ValueError(f"Empty question: {item['id']}")

        missing_articles = (
            set(item["article_numbers"]) - article_numbers
        )

        if missing_articles:
            raise ValueError(
                f"{item['id']} references missing articles: "
                f"{sorted(missing_articles)}"
            )

        new_questions.append({
            "id": item["id"],
            "question": item["question"],
            "category": item["category"],
            "article_numbers": item["article_numbers"],
            "expected_answer": "",
            "supporting_passages": [],
            "verification_status": "pending",
            "split": None
        })

    all_questions = existing + new_questions

    with EVAL.open("w", encoding="utf-8") as file:
        for question in all_questions:
            file.write(
                json.dumps(question, ensure_ascii=False) + "\n"
            )

    print("Batch imported successfully.")
    print("New questions:", len(new_questions))
    print("Total questions:", len(all_questions))
    print(
        "Verified questions:",
        sum(
            q["verification_status"] == "verified"
            for q in all_questions
        )
    )
    print(
        "Pending questions:",
        sum(
            q["verification_status"] == "pending"
            for q in all_questions
        )
    )


if __name__ == "__main__":
    main()
