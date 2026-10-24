"""Read-only evidence access to the locally ingested EU AI Act corpus."""

import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/evidence", tags=["evidence"])

ARTICLES_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "articles.jsonl"
)


@lru_cache(maxsize=1)
def load_articles() -> dict[int, dict]:
    """Load and index the official regulation articles once per process."""
    articles = {}

    with ARTICLES_PATH.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            article = json.loads(line)
            number = int(article["article_number"])

            articles[number] = {
                "article_number": number,
                "article_id": article.get("article_id"),
                "title": article["title"],
                "text": article["text"],
                "source_url": article.get("source_url"),
                "source_sha256": article.get("source_sha256"),
            }

    return articles


@router.get("/articles/{article_number}")
def get_article(article_number: int):
    """Return an article from the ingested legal corpus."""
    if not 1 <= article_number <= 113:
        raise HTTPException(status_code=404, detail="Article not found")

    article = load_articles().get(article_number)

    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")

    return article
