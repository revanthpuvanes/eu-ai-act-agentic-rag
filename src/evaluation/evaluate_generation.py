"""Evaluate generated RAG answers against the reviewed development dataset.

Article citations are checked structurally. This is NOT a semantic
correctness or legal-completeness evaluator.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.generation.naive_rag import answer_question
from src.retrieval.hybrid import HybridRetriever


DATASET_PATH = Path("data/eval/questions.jsonl")
RESULTS_DIR = Path("evals/results")


def load_questions(path=DATASET_PATH, split="dev"):
    questions = []

    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)

            if item.get("split") == split:
                questions.append(item)

    return questions


def citation_metrics(predicted, expected):
    """Compare cited article numbers with gold article numbers."""
    predicted = set(predicted)
    expected = set(expected)

    if not expected:
        # Unanswerable questions are evaluated through abstention instead.
        return {
            "citation_precision": None,
            "citation_recall": None,
            "citation_f1": None,
        }

    true_positives = len(predicted & expected)

    precision = (
        true_positives / len(predicted)
        if predicted
        else 0.0
    )

    recall = true_positives / len(expected)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "citation_precision": round(precision, 4),
        "citation_recall": round(recall, 4),
        "citation_f1": round(f1, 4),
    }


def estimate_cost_usd(
    input_tokens,
    output_tokens,
    input_price_per_million,
    output_price_per_million,
):
    """Use caller-provided prices; never silently assume live pricing."""
    if input_tokens is None or output_tokens is None:
        return None

    return round(
        (
            input_tokens * input_price_per_million
            + output_tokens * output_price_per_million
        ) / 1_000_000,
        8,
    )


def evaluate_one(
    question,
    retriever,
    client,
    model,
    input_price_per_million=0.0,
    output_price_per_million=0.0,
):
    """Evaluate one question using the existing naive RAG pipeline."""
    gold_articles = question.get("article_numbers", [])
    is_unanswerable = question["category"] == "unanswerable"

    start = time.perf_counter()

    try:
        result = answer_question(
            question=question["question"],
            retriever=retriever,
            client=client,
            model=model,
            k=5,
        )

        metrics = citation_metrics(
            result["cited_articles"],
            gold_articles,
        )

        abstained = bool(result["abstained"])

        return {
            "id": question["id"],
            "category": question["category"],
            "split": question["split"],
            "question": question["question"],
            "gold_articles": gold_articles,
            "gold_answer": question.get("answer"),
            "answer": result["answer"],
            "cited_articles": result["cited_articles"],
            "retrieved_articles": result["retrieved_articles"],
            "abstained": abstained,
            "expected_abstention": is_unanswerable,
            "abstention_correct": abstained == is_unanswerable,
            **metrics,
            "retrieval_seconds": result["retrieval_seconds"],
            "generation_seconds": result["generation_seconds"],
            "total_seconds": result["total_seconds"],
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "total_tokens": result.get("total_tokens"),
            "estimated_cost_usd": estimate_cost_usd(
                result.get("input_tokens"),
                result.get("output_tokens"),
                input_price_per_million,
                output_price_per_million,
            ),
            "model": model,
            "provider": result.get("provider", "unknown"),
            "status": "ok",
            "error": None,
            "wall_seconds": round(time.perf_counter() - start, 3),
        }

    except Exception as error:
        return {
            "id": question["id"],
            "category": question["category"],
            "split": question["split"],
            "question": question["question"],
            "gold_articles": gold_articles,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "wall_seconds": round(time.perf_counter() - start, 3),
        }


def average(values):
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 4) if values else None


def summarize(results):
    successful = [
        item for item in results
        if item["status"] == "ok"
    ]

    answerable = [
        item for item in successful
        if not item["expected_abstention"]
    ]

    return {
        "questions_attempted": len(results),
        "questions_succeeded": len(successful),
        "questions_failed": len(results) - len(successful),
        "answerable_questions_succeeded": len(answerable),
        "mean_citation_precision": average([
            item["citation_precision"] for item in answerable
        ]),
        "mean_citation_recall": average([
            item["citation_recall"] for item in answerable
        ]),
        "mean_citation_f1": average([
            item["citation_f1"] for item in answerable
        ]),
        "abstention_accuracy": average([
            float(item["abstention_correct"])
            for item in successful
        ]),
        "mean_total_seconds": average([
            item["total_seconds"] for item in successful
        ]),
        "total_input_tokens": sum(
            item["input_tokens"] or 0
            for item in successful
        ),
        "total_output_tokens": sum(
            item["output_tokens"] or 0
            for item in successful
        ),
        "estimated_total_cost_usd": round(sum(
            item["estimated_cost_usd"] or 0
            for item in successful
        ), 6),
        "semantic_answer_quality_evaluated": False,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum development questions to run.",
    )

    parser.add_argument(
        "--input-price",
        type=float,
        default=None,
        help="USD per million input tokens.",
    )

    parser.add_argument(
        "--output-price",
        type=float,
        default=None,
        help="USD per million output tokens.",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    if (args.input_price is None) != (args.output_price is None):
        parser.error(
            "Supply both --input-price and --output-price, or neither."
        )

    if args.input_price is not None and (
        args.input_price < 0 or args.output_price < 0
    ):
        parser.error("Token prices must not be negative")

    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise SystemExit("MISTRAL_API_KEY missing from .env")

    model = os.getenv("MISTRAL_MODEL", "mistral-small-2603")

    questions = load_questions(split="dev")[:args.limit]

    print(f"Loading hybrid retriever for {len(questions)} questions...")
    retriever = HybridRetriever.from_jsonl()

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
        timeout=60.0,
        max_retries=1,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    output_path = RESULTS_DIR / (
        f"naive_rag_dev_{timestamp}.jsonl"
    )

    results = []

    with output_path.open("w", encoding="utf-8") as file:
        for index, question in enumerate(questions, start=1):
            print(
                f"[{index}/{len(questions)}] "
                f"{question['id']} — {question['category']}",
                flush=True,
            )

            result = evaluate_one(
                question,
                retriever,
                client,
                model,
                input_price_per_million=args.input_price or 0.0,
                output_price_per_million=args.output_price or 0.0,
            )

            results.append(result)

            file.write(json.dumps(result, ensure_ascii=False) + "\n")
            file.flush()

            if result["status"] == "ok":
                print(
                    f"  OK | {result['total_seconds']}s | "
                    f"citations={result['cited_articles']}"
                )
            else:
                print(
                    f"  ERROR | {result['error']}"
                )

    summary = summarize(results)
    summary["model"] = model
    summary["dataset_split"] = "dev"
    summary["pricing_provided"] = args.input_price is not None

    if not summary["pricing_provided"]:
        summary["estimated_total_cost_usd"] = None

        for result in results:
            if result["status"] == "ok":
                result["estimated_cost_usd"] = None

        # Rewrite JSONL so unknown pricing is never represented as $0.
        with output_path.open("w", encoding="utf-8") as file:
            for result in results:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary_path = output_path.with_suffix(".summary.json")

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== NAIVE RAG DEV SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
