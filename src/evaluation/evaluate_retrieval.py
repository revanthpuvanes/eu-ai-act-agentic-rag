import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.retrieval.bm25 import BM25Retriever


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def evaluate(retriever, questions, ks=(5, 10)):
    results = defaultdict(list)
    details = []

    for question in questions:
        gold = set(question["article_numbers"])

        # Abstention quality needs an answer-generation evaluator.
        if not gold:
            details.append({
                "id": question["id"],
                "category": question["category"],
                "status": "unanswerable_excluded",
            })
            continue

        top_k = max(ks)
        retrieved = retriever.search(
            question["question"],
            k=top_k,
        )

        row = {
            "id": question["id"],
            "category": question["category"],
            "gold_articles": sorted(gold),
            "retrieved_articles": [
                item["article_number"]
                for item in retrieved
            ],
        }

        for k in ks:
            found = {
                item["article_number"]
                for item in retrieved[:k]
            }

            recall = len(gold & found) / len(gold)
            complete = int(gold <= found)

            results[f"article_recall@{k}"].append(recall)
            results[f"complete_evidence@{k}"].append(complete)

            row[f"article_recall@{k}"] = recall
            row[f"complete_evidence@{k}"] = complete

        details.append(row)

    summary = {
        metric: round(sum(values) / len(values), 4)
        for metric, values in results.items()
        if values
    }

    summary["answerable_questions_evaluated"] = sum(
        bool(q["article_numbers"])
        for q in questions
    )
    summary["unanswerable_questions_excluded"] = sum(
        not q["article_numbers"]
        for q in questions
    )

    return summary, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        default="data/eval/questions.jsonl",
    )
    parser.add_argument(
        "--chunks",
        default="data/processed/chunks.jsonl",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
    )
    parser.add_argument(
        "--output",
        default="evals/results/bm25_dev.json",
    )

    args = parser.parse_args()

    questions = [
        q
        for q in load_jsonl(args.questions)
        if q["split"] == args.split
    ]

    if not questions:
        raise ValueError("No questions found for split")

    retriever = BM25Retriever.from_jsonl(args.chunks)

    summary, details = evaluate(retriever, questions)

    result = {
        "retriever": "bm25",
        "split": args.split,
        "chunk_count": len(retriever.chunks),
        "summary": summary,
        "questions": details,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== BM25 RETRIEVAL EVALUATION ===")
    print("Split:", args.split)
    print("Questions:", len(questions))
    print("Chunks:", len(retriever.chunks))

    for metric, value in summary.items():
        print(f"{metric}: {value}")

    print("\nLowest Recall@5 questions:")

    ranked = sorted(
        [
            row for row in details
            if "article_recall@5" in row
        ],
        key=lambda row: row["article_recall@5"],
    )

    for row in ranked[:5]:
        print(
            row["id"],
            "gold=", row["gold_articles"],
            "retrieved=", row["retrieved_articles"][:5],
            "recall=", row["article_recall@5"],
        )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
