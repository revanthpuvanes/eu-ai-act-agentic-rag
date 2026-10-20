import argparse
import json
from pathlib import Path


def chunk_text(text: str, size: int = 1200, overlap: int = 200):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")

    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        yield {
            "start_char": start,
            "end_char": end,
            "text": text[start:end],
        }

        if end == len(text):
            break

        start = end - overlap


def build_chunks(articles, size=1200, overlap=200):
    for article in articles:
        for index, part in enumerate(
            chunk_text(article["text"], size, overlap)
        ):
            yield {
                "chunk_id": f"article_{article['article_number']}_chunk_{index:03d}",
                "article_number": article["article_number"],
                "article_id": article["article_id"],
                "title": article["title"],
                "text": part["text"],
                "start_char": part["start_char"],
                "end_char": part["end_char"],
                "source_url": article["source_url"],
                "source_sha256": article["source_sha256"],
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/articles.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
    )
    parser.add_argument("--size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    articles = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    chunks = list(build_chunks(articles, args.size, args.overlap))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Articles: {len(articles)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
