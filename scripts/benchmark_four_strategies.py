"""Four-way EU AI Act RAG generation benchmark.

Strategies:
- naive
- expanded
- deterministic_langgraph
- llm_langgraph

Uses the development split by default.
Does not automatically evaluate legal answer correctness.
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.agents.evidence_agent import EvidenceAwareAgent
from src.agents.llm_evidence_agent import LLMEvidenceAgent
from src.evaluation.evaluate_generation import (
    citation_metrics,
    load_questions,
)
from src.generation.naive_rag import answer_question
from src.retrieval.context_expansion import ArticleContextExpander
from src.retrieval.expanded_hybrid import ExpandedHybridRetriever
from src.retrieval.hybrid import HybridRetriever


STRATEGIES = (
    "naive",
    "expanded",
    "deterministic_langgraph",
    "llm_langgraph",
)


def create_systems(client, model):
    base = HybridRetriever.from_jsonl()

    expander = ArticleContextExpander(
        max_article_chars=15000,
        max_context_chars=22000,
    )

    expanded = ExpandedHybridRetriever(
        base_retriever=base,
        expander=expander,
    )

    deterministic = EvidenceAwareAgent(
        retriever=base,
        expander=expander,
        client=client,
        model=model,
    )

    llm = LLMEvidenceAgent(
        retriever=base,
        expander=expander,
        client=client,
        model=model,
    )

    return {
        "naive": base,
        "expanded": expanded,
        "deterministic_langgraph": deterministic,
        "llm_langgraph": llm,
    }


def question_identifier(question):
    return question.get("id", question.get("question_id"))


def expected_articles(question):
    return [
        int(number)
        for number in question.get("article_numbers", [])
    ]


def evaluate_one(question, strategy, systems, client, model):
    question_id = question_identifier(question)
    question_text = question["question"]
    gold_articles = expected_articles(question)

    row = {
        "question_id": question_id,
        "question": question_text,
        "category": question.get("category"),
        "strategy": strategy,
        "gold_articles": gold_articles,
        "status": "error",
        "error": None,
    }

    started = time.perf_counter()

    try:
        if strategy in {
            "deterministic_langgraph",
            "llm_langgraph",
        }:
            result = systems[strategy].answer(question_text)
        else:
            result = answer_question(
                question=question_text,
                retriever=systems[strategy],
                client=client,
                model=model,
                k=5,
            )

        cited = [
            int(number)
            for number in result.get("cited_articles", [])
        ]

        metrics = citation_metrics(
            cited,
            gold_articles,
        )

        assessment = result.get("assessment") or {}

        row.update({
            "status": "ok",
            "answer": result.get("answer", ""),
            "cited_articles": cited,
            "retrieved_articles": result.get(
                "retrieved_articles", []
            ),
            "abstained": bool(
                result.get("abstained", False)
            ),
            "citation_precision": metrics.get(
                "citation_precision"
            ),
            "citation_recall": metrics.get(
                "citation_recall"
            ),
            "citation_f1": metrics.get(
                "citation_f1"
            ),
            "total_seconds": result.get(
                "total_seconds",
                time.perf_counter() - started,
            ),
            "input_tokens": result.get(
                "input_tokens", 0
            ),
            "output_tokens": result.get(
                "output_tokens", 0
            ),
            "total_tokens": result.get(
                "total_tokens", 0
            ),
            "action_taken": result.get(
                "action_taken"
            ),
            "assessment_action": assessment.get(
                "action"
            ),
            "assessment_reason": assessment.get(
                "reason"
            ),
            "assessment_article": assessment.get(
                "article_number"
            ),
            "assessment_search_query": assessment.get(
                "search_query"
            ),
            "assessor_input_tokens": result.get(
                "assessor_input_tokens", 0
            ),
            "assessor_output_tokens": result.get(
                "assessor_output_tokens", 0
            ),
            "generation_input_tokens": result.get(
                "generation_input_tokens"
            ),
            "generation_output_tokens": result.get(
                "generation_output_tokens"
            ),
            "expansion_performed": result.get(
                "expansion_performed"
            ),
            "assessment_reason_deterministic": result.get(
                "assessment_reason"
            ) if strategy == "deterministic_langgraph"
            else None,
        })

    except Exception as exc:
        row["error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        row["total_seconds"] = (
            time.perf_counter() - started
        )

    return row


def mean(values):
    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return round(
        sum(values) / len(values),
        4,
    )


def summarize(rows):
    grouped = defaultdict(dict)

    for row in rows:
        grouped[row["question_id"]][
            row["strategy"]
        ] = row

    paired_ids = []

    for question_id, strategies in grouped.items():
        if all(
            name in strategies
            and strategies[name]["status"] == "ok"
            for name in STRATEGIES
        ):
            paired_ids.append(question_id)

    summary = {
        "complete_paired_questions": len(paired_ids),
        "strategies": {},
        "semantic_answer_quality_evaluated": False,
        "human_review_required": True,
    }

    for strategy in STRATEGIES:
        selected = [
            grouped[question_id][strategy]
            for question_id in paired_ids
        ]

        answerable = [
            row
            for row in selected
            if row["gold_articles"]
        ]

        summary["strategies"][strategy] = {
            "paired_samples": len(selected),
            "citation_f1": mean([
                row["citation_f1"]
                for row in answerable
            ]),
            "citation_precision": mean([
                row["citation_precision"]
                for row in answerable
            ]),
            "citation_recall": mean([
                row["citation_recall"]
                for row in answerable
            ]),
            "average_latency_seconds": mean([
                row["total_seconds"]
                for row in selected
            ]),
            "average_total_tokens": mean([
                row["total_tokens"]
                for row in selected
            ]),
            "total_tokens": sum(
                row["total_tokens"] or 0
                for row in selected
            ),
            "abstention_accuracy": mean([
                int(
                    row["abstained"]
                    == (not row["gold_articles"])
                )
                for row in selected
            ]),
            "llm_assessor_actions": {
                action: sum(
                    row.get("assessment_action") == action
                    for row in selected
                )
                for action in (
                    "answer",
                    "expand",
                    "search",
                    "abstain",
                )
            },
        }

    return summary


def write_review_sheet(rows, output_path):
    fields = [
        "question_id",
        "category",
        "strategy",
        "question",
        "gold_articles",
        "cited_articles",
        "answer",
        "action_taken",
        "assessment_action",
        "assessment_reason",
        "completeness_score_0_to_3",
        "legal_accuracy_score_0_to_3",
        "missing_requirements",
        "incorrect_claims",
        "reviewer_notes",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            if row["status"] != "ok":
                continue

            writer.writerow({
                field: (
                    json.dumps(
                        row[field],
                        ensure_ascii=False,
                    )
                    if field in {
                        "gold_articles",
                        "cited_articles",
                    }
                    else row.get(field, "")
                )
                for field in fields
            })


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    if args.limit < 1:
        raise SystemExit("--limit must be positive")

    if args.start < 0:
        raise SystemExit("--start cannot be negative")

    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise SystemExit("MISTRAL_API_KEY missing")

    model = os.getenv(
        "MISTRAL_MODEL",
        "mistral-small-2603",
    )

    questions = load_questions(split="dev")

    selected = questions[
        args.start:args.start + args.limit
    ]

    if not selected:
        raise SystemExit("No questions selected")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
        timeout=60,
        max_retries=1,
    )

    systems = create_systems(client, model)

    output_dir = Path("evals/results")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    prefix = output_dir / (
        f"four_way_dev_{timestamp}"
    )

    jsonl_path = prefix.with_suffix(".jsonl")
    summary_path = Path(
        str(prefix) + ".summary.json"
    )
    review_path = Path(
        str(prefix) + ".review.csv"
    )

    rows = []

    with jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for question in selected:
            question_id = question_identifier(
                question
            )

            print(
                f"\n=== {question_id} ===",
                flush=True,
            )

            for strategy in STRATEGIES:
                print(
                    f"Running {strategy}...",
                    flush=True,
                )

                row = evaluate_one(
                    question,
                    strategy,
                    systems,
                    client,
                    model,
                )

                rows.append(row)

                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    ) + "\n"
                )

                handle.flush()

                print(
                    f"  status={row['status']}"
                    f" tokens={row.get('total_tokens')}"
                    f" latency={row.get('total_seconds')}"
                    f" action={row.get('action_taken')}",
                    flush=True,
                )

                if row["status"] == "error":
                    print(
                        f"  error={row['error']}",
                        flush=True,
                    )

                time.sleep(
                    max(0, args.delay)
                )

    summary = summarize(rows)

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_review_sheet(
        rows,
        review_path,
    )

    print("\n=== SUMMARY ===")

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print("\nFiles:")
    print(jsonl_path)
    print(summary_path)
    print(review_path)


if __name__ == "__main__":
    main()
