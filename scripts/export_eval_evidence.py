
import argparse
import json
from pathlib import Path

CORPUS = Path("data/processed/articles.jsonl")
EVAL = Path("data/eval/questions.jsonl")
OUTPUT_DIR = Path("data/eval/evidence_packs")


def load_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("category")
    args = parser.parse_args()

    articles = {
        article["article_number"]: article
        for article in load_jsonl(CORPUS)
    }

    questions = [
        question
        for question in load_jsonl(EVAL)
        if question["category"] == args.category
        and question["verification_status"] == "pending"
    ]

    if not questions:
        raise ValueError(
            f"No pending questions found for category: {args.category}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"{args.category}.md"

    lines = [
        f"# Evidence Pack: {args.category}",
        "",
        f"Questions: {len(questions)}",
        ""
    ]

    for question in questions:
        lines.extend([
            "---",
            "",
            f"## {question['id']}",
            "",
            f"**Question:** {question['question']}",
            "",
            f"**Split:** {question['split']}",
            "",
        ])

        if not question["article_numbers"]:
            lines.extend([
                "**Referenced articles:** None",
                "",
                "This question is designed to test whether the "
                "system recognises missing information in the corpus.",
                ""
            ])
            continue

        for number in question["article_numbers"]:
            article = articles[number]

            lines.extend([
                f"### Article {number}: {article['title']}",
                "",
                article["text"],
                ""
            ])

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("Evidence pack created:", output_path)
    print("Questions included:", len(questions))
    print("File size:", output_path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
