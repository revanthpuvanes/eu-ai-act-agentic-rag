
import json
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Suppress the XHTML parsing warning.
warnings.filterwarnings(
    "ignore",
    category=XMLParsedAsHTMLWarning,
)

RAW = Path("data/raw/eu_ai_act.html")
PROCESSED = Path("data/processed/articles.jsonl")


def normalize(text):
    """Normalize whitespace without changing legal wording."""
    return " ".join(text.split())


def normalize_title(title, article_number):
    """
    Apply documented corrections to known source anomalies.

    The downloaded EUR-Lex HTML contains an unexpected
    trailing backtick in the title of Article 1.
    """
    title = normalize(title)

    if article_number == 1 and title == "Subject matter`":
        return "Subject matter"

    return title


def main():
    soup = BeautifulSoup(RAW.read_bytes(), "lxml")

    articles = [
        json.loads(line)
        for line in PROCESSED.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    errors = []

    for article in articles:
        number = article["article_number"]

        container = soup.find(
            "div",
            id=f"art_{number}",
            class_="eli-subdivision",
        )

        if container is None:
            errors.append(
                f"Article {number}: missing container"
            )
            continue

        heading = container.find(
            "p",
            class_="oj-ti-art",
            recursive=False,
        )

        title = container.find(
            "div",
            class_="eli-title",
            recursive=False,
        )

        if heading is None or title is None:
            errors.append(
                f"Article {number}: missing heading/title"
            )
            continue

        # Reconstruct the body from the original HTML.
        body_parts = []

        for child in container.find_all(recursive=False):
            if child is heading or child is title:
                continue

            text = normalize(
                child.get_text(" ", strip=True)
            )

            if text:
                body_parts.append(text)

        expected_body = "\n".join(body_parts)

        if article["text"] != expected_body:
            errors.append(
                f"Article {number}: extracted body mismatch"
            )

        # Compare titles after applying the documented
        # Article 1 source-text correction.
        expected_title = normalize_title(
            title.get_text(" ", strip=True),
            number,
        )

        actual_title = normalize(article["title"])

        if actual_title != expected_title:
            errors.append(
                f"Article {number}: title mismatch "
                f"(expected {expected_title!r}, "
                f"got {actual_title!r})"
            )

        nested_headings = container.select("p.oj-ti-art")

        if len(nested_headings) != 1:
            errors.append(
                f"Article {number}: "
                f"{len(nested_headings)} article headings"
            )

    print("=== CORPUS INTEGRITY REPORT ===")
    print("Articles checked:", len(articles))
    print("Errors:", len(errors))

    if errors:
        for error in errors[:20]:
            print("ERROR:", error)

        raise SystemExit(1)

    print("All structural checks passed.")


if __name__ == "__main__":
    main()
