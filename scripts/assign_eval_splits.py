
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

EVAL = Path("data/eval/questions.jsonl")

SEED = 42

DEV_TARGETS = {
    "single_hop": 7,
    "multi_hop": 5,
    "exceptions": 3,
    "definitions": 2,
    "unanswerable": 1,
    "temporal": 2,
}

STARTER_IDS = {f"Q{i:03d}" for i in range(1, 6)}


def main():
    questions = [
        json.loads(line)
        for line in EVAL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(questions) != 60:
        raise ValueError("Expected exactly 60 questions")

    if len({q["id"] for q in questions}) != 60:
        raise ValueError("Duplicate question IDs")

    by_category = defaultdict(list)

    for question in questions:
        by_category[question["category"]].append(question)

    if set(by_category) != set(DEV_TARGETS):
        raise ValueError("Unexpected category")

    rng = random.Random(SEED)

    for category, group in by_category.items():
        required_dev = DEV_TARGETS[category]

        # Preserve the five previously reviewed questions in dev.
        fixed_dev = [
            q for q in group
            if q["id"] in STARTER_IDS
        ]

        candidates = [
            q for q in group
            if q["id"] not in STARTER_IDS
        ]

        needed = required_dev - len(fixed_dev)

        if needed < 0 or needed > len(candidates):
            raise ValueError(f"Invalid allocation for {category}")

        rng.shuffle(candidates)

        selected_dev = {
            q["id"] for q in fixed_dev + candidates[:needed]
        }

        for question in group:
            question["split"] = (
                "dev" if question["id"] in selected_dev else "test"
            )

    counts = Counter(q["split"] for q in questions)

    if counts != {"dev": 20, "test": 40}:
        raise ValueError(f"Unexpected split counts: {counts}")

    # Validate every category's development allocation.
    for category, target in DEV_TARGETS.items():
        actual = sum(
            q["category"] == category and q["split"] == "dev"
            for q in questions
        )
        if actual != target:
            raise ValueError(f"Incorrect dev count for {category}")

    # Only save after all checks pass.
    with EVAL.open("w", encoding="utf-8") as file:
        for question in questions:
            file.write(json.dumps(question, ensure_ascii=False) + "\n")

    print("=== SPLIT ASSIGNMENT COMPLETE ===")
    print("Total questions:", len(questions))
    print("Development:", counts["dev"])
    print("Test:", counts["test"])
    print("Random seed:", SEED)
    print()
    print("Category breakdown:")

    for category in DEV_TARGETS:
        dev = sum(
            q["category"] == category and q["split"] == "dev"
            for q in questions
        )
        test = sum(
            q["category"] == category and q["split"] == "test"
            for q in questions
        )
        print(f"{category}: dev={dev}, test={test}")


if __name__ == "__main__":
    main()
