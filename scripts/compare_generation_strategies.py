"""Paired naive-vs-expanded RAG evaluation on the development split.

Both strategies use:
- The same question
- The same model and generation prompt
- The same underlying hybrid retriever
- The same retrieved article ranking

Only the context expansion strategy differs.

Citation metrics do not measure semantic answer correctness.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.evaluation.evaluate_generation import (
    citation_metrics,
    estimate_cost_usd,
    load_questions,
)
from src.generation.naive_rag import answer_question
from src.retrieval.context_expansion import ArticleContextExpander
from src.retrieval.expanded_hybrid import ExpandedHybridRetriever
from src.retrieval.hybrid import HybridRetriever


OUTPUT_DIR = Path("evals/results")


def run_strategy(
    question,
    strategy_name,
    retriever,
    client,
    model,
    input_price=None,
    output_price=None,
):
    """Execute one strategy and return a structured result."""
    started = time.perf_counter()

    gold_articles = question["article_numbers"]
    expected_abstention = question["category"] == "unanswerable"

    try:
        response = answer_question(
            question=question["question"],
            retriever=retriever,
            client=client,
            model=model,
            k=5,
        )

        citations = citation_metrics(
            response["cited_articles"],
            gold_articles,
        )

        estimated_cost = None

        if input_price is not None and output_price is not None:
            estimated_cost = estimate_cost_usd(
                response.get("input_tokens"),
                response.get("output_tokens"),
                input_price,
                output_price,
            )

        return {
            "question_id": question["id"],
            "question": question["question"],
            "category": question["category"],
            "strategy": strategy_name,
            "status": "ok",
            "answer": response["answer"],
            "gold_articles": gold_articles,
            "cited_articles": response["cited_articles"],
            "retrieved_articles": response["retrieved_articles"],
            "abstained": response["abstained"],
            "expected_abstention": expected_abstention,
            "abstention_correct": (
                bool(response["abstained"]) == expected_abstention
            ),
            **citations,
            "retrieval_seconds": response["retrieval_seconds"],
            "generation_seconds": response["generation_seconds"],
            "total_seconds": response["total_seconds"],
            "input_tokens": response.get("input_tokens"),
            "output_tokens": response.get("output_tokens"),
            "total_tokens": response.get("total_tokens"),
            "estimated_cost_usd": estimated_cost,
            "model": model,
            "provider": response.get("provider"),
            "error": None,
            "wall_seconds": round(time.perf_counter() - started, 3),
        }

    except Exception as error:
        return {
            "question_id": question["id"],
            "question": question["question"],
            "category": question["category"],
            "strategy": strategy_name,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "wall_seconds": round(time.perf_counter() - started, 3),
        }


def mean(values):
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 4) if values else None


def summarize_strategy(results):
    successful = [
        item for item in results if item["status"] == "ok"
    ]

    answerable = [
        item for item in successful
        if not item["expected_abstention"]
    ]

    costs = [
        item["estimated_cost_usd"]
        for item in successful
        if item["estimated_cost_usd"] is not None
    ]

    return {
        "attempted": len(results),
        "succeeded": len(successful),
        "failed": len(results) - len(successful),
        "mean_citation_precision": mean([
            item["citation_precision"] for item in answerable
        ]),
        "mean_citation_recall": mean([
            item["citation_recall"] for item in answerable
        ]),
        "mean_citation_f1": mean([
            item["citation_f1"] for item in answerable
        ]),
        "abstention_accuracy": mean([
            float(item["abstention_correct"])
            for item in successful
        ]),
        "mean_latency_seconds": mean([
            item["total_seconds"] for item in successful
        ]),
        "total_input_tokens": sum(
            item["input_tokens"] or 0 for item in successful
        ),
        "total_output_tokens": sum(
            item["output_tokens"] or 0 for item in successful
        ),
        "estimated_total_cost_usd": (
            round(sum(costs), 6)
            if len(costs) == len(successful) and successful
            else None
        ),
        "semantic_correctness_measured": False,
        "legal_completeness_measured": False,
    }


def build_retrievers():
    """Share one hybrid retriever between both strategies."""
    base = HybridRetriever.from_jsonl()

    expanded = ExpandedHybridRetriever(
        base_retriever=base,
        expander=ArticleContextExpander(
            max_article_chars=15000,
            max_context_chars=22000,
        ),
        expand_top_n=1,
    )

    return {
        "naive": base,
        "expanded": expanded,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Number of dev questions; default is 2.",
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

    if args.limit < 1 or args.limit > 20:
        parser.error("--limit must be between 1 and 20")

    if (args.input_price is None) != (args.output_price is None):
        parser.error("Provide both token prices or neither")

    if args.input_price is not None and (
        args.input_price < 0 or args.output_price < 0
    ):
        parser.error("Token prices cannot be negative")

    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise SystemExit("MISTRAL_API_KEY missing from .env")

    model = os.getenv("MISTRAL_MODEL", "mistral-small-2603")

    questions = load_questions(split="dev")[:args.limit]
    retrievers = build_retrievers()

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
        timeout=60,
        max_retries=1,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    output_path = (
        OUTPUT_DIR / f"paired_rag_dev_{timestamp}.jsonl"
    )

    results = []

    with output_path.open("w", encoding="utf-8") as file:
        for index, question in enumerate(questions, start=1):
            print(
                f"\n[{index}/{len(questions)}] "
                f"{question['id']}: {question['question']}",
                flush=True,
            )

            for strategy_name in ("naive", "expanded"):
                print(
                    f"  Running {strategy_name}...",
                    flush=True,
                )

                result = run_strategy(
                    question=question,
                    strategy_name=strategy_name,
                    retriever=retrievers[strategy_name],
                    client=client,
                    model=model,
                    input_price=args.input_price,
                    output_price=args.output_price,
                )

                results.append(result)
                file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                file.flush()

                if result["status"] == "ok":
                    print(
                        f"    OK | "
                        f"{result['total_seconds']}s | "
                        f"{result['total_tokens']} tokens"
                    )
                else:
                    print(f"    ERROR | {result['error']}")

    summary = {
        "model": model,
        "split": "dev",
        "questions": len(questions),
        "naive": summarize_strategy([
            item for item in results
            if item["strategy"] == "naive"
        ]),
        "expanded": summarize_strategy([
            item for item in results
            if item["strategy"] == "expanded"
        ]),
    }

    summary_path = output_path.with_suffix(".summary.json")

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== PAIRED RAG SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nResults:", output_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
