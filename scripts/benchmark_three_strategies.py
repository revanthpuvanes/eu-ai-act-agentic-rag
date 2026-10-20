"""Compare naive, expanded, and LangGraph RAG on dev questions.

Records every API result immediately to JSONL.
A complete paired question requires all three strategies to succeed.

This benchmark measures citations, abstention, latency, and tokens.
It does not automatically measure legal answer correctness.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.agents.evidence_agent import EvidenceAwareAgent
from src.evaluation.evaluate_generation import (
    citation_metrics,
    load_questions,
)
from src.generation.naive_rag import answer_question
from src.retrieval.context_expansion import ArticleContextExpander
from src.retrieval.expanded_hybrid import ExpandedHybridRetriever
from src.retrieval.hybrid import HybridRetriever


RESULTS_DIR = Path("evals/results")
STRATEGIES = ("naive", "expanded", "langgraph")


def average(values):
    values = [x for x in values if x is not None]
    return round(sum(values) / len(values), 4) if values else None


def create_systems(client, model):
    base = HybridRetriever.from_jsonl()

    expanded = ExpandedHybridRetriever(
        base_retriever=base,
        expander=ArticleContextExpander(
            max_article_chars=15000,
            max_context_chars=22000,
        ),
        expand_top_n=1,
    )

    agent = EvidenceAwareAgent(
        retriever=base,
        expander=ArticleContextExpander(
            max_article_chars=15000,
            max_context_chars=22000,
        ),
        client=client,
        model=model,
    )

    return {
        "naive": base,
        "expanded": expanded,
        "langgraph": agent,
    }


def evaluate(question, strategy, systems, client, model):
    started = time.perf_counter()

    record = {
        "question_id": question["id"],
        "question": question["question"],
        "category": question["category"],
        "strategy": strategy,
        "gold_articles": question["article_numbers"],
        "status": "error",
    }

    try:
        if strategy == "langgraph":
            response = systems["langgraph"].answer(
                question["question"]
            )
        else:
            response = answer_question(
                question=question["question"],
                retriever=systems[strategy],
                client=client,
                model=model,
                k=5,
            )

        gold = question["article_numbers"]
        expected_abstention = (
            question["category"] == "unanswerable"
        )

        metrics = citation_metrics(
            response["cited_articles"],
            gold,
        )

        record.update({
            "status": "ok",
            "answer": response["answer"],
            "cited_articles": response["cited_articles"],
            "retrieved_articles": response["retrieved_articles"],
            "abstained": response["abstained"],
            "expected_abstention": expected_abstention,
            "abstention_correct": (
                bool(response["abstained"])
                == expected_abstention
            ),
            "citation_precision": metrics["citation_precision"],
            "citation_recall": metrics["citation_recall"],
            "citation_f1": metrics["citation_f1"],
            "retrieval_seconds": response.get(
                "retrieval_seconds"
            ),
            "generation_seconds": response.get(
                "generation_seconds"
            ),
            "total_seconds": response.get("total_seconds"),
            "input_tokens": response.get("input_tokens"),
            "output_tokens": response.get("output_tokens"),
            "total_tokens": response.get("total_tokens"),
            "needs_expansion": response.get(
                "needs_expansion"
            ),
            "expansion_performed": response.get(
                "expansion_performed"
            ),
            "assessment_reason": response.get(
                "assessment_reason"
            ),
            "model": response.get("model", model),
            "provider": response.get("provider"),
        })

    except Exception as error:
        record["error"] = (
            f"{type(error).__name__}: {error}"
        )

    record["wall_seconds"] = round(
        time.perf_counter() - started,
        4,
    )

    return record


def summarize(rows):
    """Use only questions completed by all three strategies."""
    successful = {
        (row["question_id"], row["strategy"]): row
        for row in rows
        if row["status"] == "ok"
    }

    question_ids = sorted({
        row["question_id"] for row in rows
    })

    paired_ids = [
        qid for qid in question_ids
        if all(
            (qid, strategy) in successful
            for strategy in STRATEGIES
        )
    ]

    summary = {
        "attempted_questions": len(question_ids),
        "complete_paired_questions": len(paired_ids),
        "excluded_incomplete_questions": [
            qid for qid in question_ids
            if qid not in paired_ids
        ],
        "semantic_correctness_measured": False,
        "legal_completeness_measured": False,
        "strategies": {},
    }

    for strategy in STRATEGIES:
        selected = [
            successful[(qid, strategy)]
            for qid in paired_ids
        ]

        answerable = [
            row for row in selected
            if not row["expected_abstention"]
        ]

        total_tokens = sum(
            row["total_tokens"] or 0
            for row in selected
        )

        summary["strategies"][strategy] = {
            "paired_samples": len(selected),
            "mean_citation_f1": average([
                row["citation_f1"]
                for row in answerable
            ]),
            "mean_citation_precision": average([
                row["citation_precision"]
                for row in answerable
            ]),
            "mean_citation_recall": average([
                row["citation_recall"]
                for row in answerable
            ]),
            "abstention_accuracy": average([
                float(row["abstention_correct"])
                for row in selected
            ]),
            "mean_latency_seconds": average([
                row["total_seconds"]
                for row in selected
            ]),
            "mean_input_tokens": average([
                row["input_tokens"]
                for row in selected
            ]),
            "mean_total_tokens": average([
                row["total_tokens"]
                for row in selected
            ]),
            "total_tokens": total_tokens,
            "expansion_count": (
                sum(
                    bool(row["expansion_performed"])
                    for row in selected
                )
                if strategy == "langgraph"
                else None
            ),
        }

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of development questions, default 3.",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based starting index in the dev split.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between API calls.",
    )

    args = parser.parse_args()

    if args.limit < 1 or args.start < 0:
        parser.error("--limit must be positive and --start >= 0")

    if args.delay < 0:
        parser.error("--delay must be nonnegative")

    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("MISTRAL_API_KEY missing")

    model = os.getenv(
        "MISTRAL_MODEL",
        "mistral-small-2603",
    )

    all_questions = load_questions(split="dev")

    questions = all_questions[
        args.start:args.start + args.limit
    ]

    if not questions:
        raise SystemExit("No questions in selected range")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
        timeout=60,
        max_retries=1,
    )

    systems = create_systems(client, model)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    result_path = (
        RESULTS_DIR / f"three_way_dev_{timestamp}.jsonl"
    )

    rows = []

    with result_path.open("w", encoding="utf-8") as file:
        for index, question in enumerate(questions, 1):
            print(
                f"\n[{index}/{len(questions)}] "
                f"{question['id']}: {question['question']}",
                flush=True,
            )

            for strategy in STRATEGIES:
                print(
                    f"  {strategy}: ",
                    end="",
                    flush=True,
                )

                row = evaluate(
                    question,
                    strategy,
                    systems,
                    client,
                    model,
                )

                rows.append(row)

                file.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    ) + "\n"
                )
                file.flush()

                if row["status"] == "ok":
                    print(
                        f"OK | "
                        f"{row['total_tokens']} tokens | "
                        f"{row['total_seconds']}s | "
                        f"citation F1={row['citation_f1']}"
                    )
                else:
                    print(f"ERROR | {row['error']}")

                time.sleep(args.delay)

    summary = summarize(rows)

    summary_path = result_path.with_suffix(
        ".summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== THREE-WAY BENCHMARK ===")
    print(json.dumps(summary, indent=2))
    print("\nResults:", result_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
