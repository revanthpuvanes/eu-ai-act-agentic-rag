"""Expand selected retrieval chunks to full articles within a context budget."""

import json
from pathlib import Path


class ArticleContextExpander:
    def __init__(
        self,
        articles_path="data/processed/articles.jsonl",
        max_article_chars=15000,
        max_context_chars=22000,
    ):
        self.max_article_chars = max_article_chars
        self.max_context_chars = max_context_chars
        self.articles = {}

        with Path(articles_path).open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                article = json.loads(line)
                self.articles[int(article["article_number"])] = article

    def expand(self, chunks, expand_top_n=1):
        """Expand top-ranked articles without exceeding the context budget.

        This returns copies and never mutates the original retrieval results.
        """
        if expand_top_n < 0:
            raise ValueError("expand_top_n cannot be negative")

        expanded = []
        used_chars = 0
        expanded_articles = set()

        for index, chunk in enumerate(chunks):
            item = dict(chunk)
            article_number = int(item["article_number"])
            original_text = item["text"]
            candidate_text = original_text

            if (
                index < expand_top_n
                and article_number not in expanded_articles
                and article_number in self.articles
            ):
                full_text = self.articles[article_number]["text"]

                if len(full_text) <= self.max_article_chars:
                    candidate_text = full_text

            remaining = self.max_context_chars - used_chars

            if len(candidate_text) > remaining:
                # Prefer the original retrieval chunk if the full article
                # does not fit.
                candidate_text = original_text

            if len(candidate_text) > remaining:
                break

            item["text"] = candidate_text
            item["context_expanded"] = (
                candidate_text != original_text
            )

            if item["context_expanded"]:
                item["start_char"] = 0
                item["end_char"] = len(candidate_text)

            expanded.append(item)
            used_chars += len(candidate_text)
            expanded_articles.add(article_number)

        return expanded
