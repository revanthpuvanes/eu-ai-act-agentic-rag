"""Report benchmark reliability and performance.

Reads saved JSONL results. Makes zero API calls.

Paired metrics come from the original benchmark summarizer.
All-attempt metrics include failed executions.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.benchmark_four_strategies import (
    STRATEGIES,
    summarize,
)


def load_rows(path):
    rows = []

    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

    return rows


def average(values):
    values = [
        value for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]

    return (
        round(sum(values) / len(values), 4)
        if values else None
    )


def error_category(row):
    error = str(row.get("error") or "")

    if "was not present in retrieved evidence" in error:
        return "unsupported_citation"

    if "Expansion unavailable" in error:
        return "infeasible_expansion"

    if row.get("status") != "ok":
        return "other_error"

    return None


def build_report(rows):
    # Prevent accidentally treating duplicate runs as
    # additional independent questions.
    seen = set()

    for row in rows:
        key = (
            row["question_id"],
            row["strategy"],
        )

        if key in seen:
            raise ValueError(
                f"Duplicate result for {key}. "
                "Use one benchmark run per report."
            )

        seen.add(key)

    question_ids = sorted({
        row["question_id"]
        for row in rows
    })

    paired_summary = summarize(rows)

    report = {
        "question_count": len(question_ids),
        "expected_executions": (
            len(question_ids) * len(STRATEGIES)
        ),
        "recorded_executions": len(rows),
        "complete_paired_questions": (
            paired_summary["complete_paired_questions"]
        ),
        "semantic_answer_quality_evaluated": False,
        "strategies": {},
        "paired_metrics": paired_summary["strategies"],
    }

    for strategy in STRATEGIES:
        subset = [
            row for row in rows
            if row["strategy"] == strategy
        ]

        successes = [
            row for row in subset
            if row.get("status") == "ok"
        ]

        failures = [
            row for row in subset
            if row.get("status") != "ok"
        ]

        errors = Counter(
            error_category(row)
            for row in failures
        )

        actions = Counter(
            row.get("action_taken")
            or row.get("assessment_action")
            for row in successes
            if (
                row.get("action_taken")
                or row.get("assessment_action")
            )
        )

        report["strategies"][strategy] = {
            "attempts": len(subset),
            "successes": len(successes),
            "failures": len(failures),
            "success_rate": (
                round(
                    len(successes) / len(subset),
                    4,
                )
                if subset else None
            ),
            "error_categories": dict(errors),
            "average_latency_successes": average([
                row.get("total_seconds")
                for row in successes
            ]),
            "average_tokens_successes": average([
                row.get("total_tokens")
                for row in successes
            ]),
            "total_reported_tokens_successes": sum(
                row.get("total_tokens") or 0
                for row in successes
            ),
            "executed_actions": dict(actions),
        }

    report["failed_executions"] = [
        {
            "question_id": row["question_id"],
            "strategy": row["strategy"],
            "category": error_category(row),
            "error": row.get("error"),
        }
        for row in rows
        if row.get("status") != "ok"
    ]

    return report


def print_report(report):
    print("\n" + "=" * 76)
    print("FINAL DEVELOPMENT BENCHMARK")
    print("=" * 76)

    print("Questions:", report["question_count"])
    print(
        "Recorded executions:",
        report["recorded_executions"],
        "/",
        report["expected_executions"],
    )
    print(
        "Complete paired questions:",
        report["complete_paired_questions"],
    )

    print("\nRELIABILITY — ALL ATTEMPTED QUESTIONS")
    print("-" * 76)

    print(
        f"{'Strategy':<27}"
        f"{'Success':>12}"
        f"{'Rate':>10}"
        f"{'Avg tokens':>14}"
    )

    for strategy, stats in report["strategies"].items():
        rate = stats["success_rate"]

        print(
            f"{strategy:<27}"
            f"{stats['successes']:>5}/{stats['attempts']:<6}"
            f"{rate:>9.1%}"
            f"{stats['average_tokens_successes']:>14.1f}"
        )

    print("\nCITATION QUALITY — COMPLETE PAIRED QUESTIONS")
    print("-" * 76)

    print(
        f"{'Strategy':<27}"
        f"{'Citation F1':>14}"
        f"{'Latency':>12}"
    )

    for strategy, stats in report["paired_metrics"].items():
        print(
            f"{strategy:<27}"
            f"{stats['citation_f1']:>14.4f}"
            f"{stats['average_latency_seconds']:>11.3f}s"
        )

    print("\nFAILURES")
    print("-" * 76)

    for failure in report["failed_executions"]:
        print(
            failure["question_id"],
            failure["strategy"],
            failure["category"],
        )

    print("\nLLM AGENT ACTIONS — ALL SUCCESSFUL EXECUTIONS")
    print("-" * 76)

    actions = report["strategies"][
        "llm_langgraph"
    ]["executed_actions"]

    for action, count in sorted(actions.items()):
        print(f"{action}: {count}")

    print(
        "\nNOTE: Citation F1 measures citation overlap, "
        "not semantic legal correctness."
    )

    print(
        "Latency/token averages exclude failed executions. "
        "Tokens spent on failed calls may not be captured."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        required=True,
    )

    args = parser.parse_args()

    rows = load_rows(args.results)
    report = build_report(rows)

    output = Path(args.results).with_suffix(
        ".final_report.json"
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print_report(report)
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
