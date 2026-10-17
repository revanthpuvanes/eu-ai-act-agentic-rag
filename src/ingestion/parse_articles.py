
import json
import re
import hashlib
from pathlib import Path
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Suppress the XHTML parsing warning
warnings.filterwarnings(
    "ignore",
    category=XMLParsedAsHTMLWarning
)

SOURCE = Path("data/raw/eu_ai_act.html")
OUTPUT = Path("data/processed/articles.jsonl")

SOURCE_URL = (
    "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng"
)


def normalize_text(text):
    """Normalize whitespace without changing legal wording."""
    return " ".join(text.split())


def parse_articles():
    """Extract structured articles from the EU AI Act."""

    raw_content = SOURCE.read_bytes()
    source_hash = hashlib.sha256(raw_content).hexdigest()

    soup = BeautifulSoup(raw_content, "lxml")
    articles = []

    for number in range(1, 114):
        article_div = soup.find(
            "div",
            id=f"art_{number}",
            class_="eli-subdivision",
        )

        if article_div is None:
            raise ValueError(
                f"Article {number} container not found"
            )

        heading = article_div.find(
            "p",
            class_="oj-ti-art",
        )

        if heading is None:
            raise ValueError(
                f"Article {number} heading not found"
            )

        heading_text = normalize_text(
            heading.get_text(" ", strip=True)
        )

        if not re.fullmatch(
            rf"Article {number}",
            heading_text,
        ):
            raise ValueError(
                f"Unexpected heading: {heading_text}"
            )

        title_div = article_div.find(
            "div",
            class_="eli-title",
            recursive=False,
        )

        if title_div is None:
            raise ValueError(
                f"Article {number} title not found"
            )

        title = normalize_text(
            title_div.get_text(" ", strip=True)
        )
        if number == 1 and title == "Subject matter`":
            title = "Subject matter"
        body_parts = []

        for child in article_div.find_all(
            recursive=False
        ):
            if child == heading or child == title_div:
                continue

            text = normalize_text(
                child.get_text(" ", strip=True)
            )

            if text:
                body_parts.append(text)

        body = "\n".join(body_parts)

        if not body:
            raise ValueError(
                f"Article {number} has empty body"
            )

        record = {
            "document_id": "EU-2024-1689",
            "article_id": f"Article {number}",
            "article_number": number,
            "title": title,
            "text": body,
            "source_url": SOURCE_URL,
            "source_sha256": source_hash,
        }

        articles.append(record)

    return articles


def main():
    articles = parse_articles()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("w", encoding="utf-8") as file:
        for article in articles:
            file.write(
                json.dumps(article, ensure_ascii=False)
                + "\n"
            )

    print("=== PARSER RESULTS ===")
    print("Articles extracted:", len(articles))
    print("Output:", OUTPUT.resolve())

    for article in articles[:3]:
        print(
            article["article_id"],
            "|",
            article["title"],
            "| Characters:",
            len(article["text"]),
        )


if __name__ == "__main__":
    main()
