"""Compare baseline and expanded evidence for a single RAG question."""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.generation.naive_rag import answer_question
from src.retrieval.context_expansion import ArticleContextExpander
from src.retrieval.expanded_hybrid import ExpandedHybridRetriever
from src.retrieval.hybrid import HybridRetriever


QUESTION = "Which AI practices are prohibited under the EU AI Act?"


def main():
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise SystemExit("MISTRAL_API_KEY missing from .env")

    model = os.getenv("MISTRAL_MODEL", "mistral-small-2603")

    # Shared underlying retriever: same ranking, different context.
    base_retriever = HybridRetriever.from_jsonl()

    expanded_retriever = ExpandedHybridRetriever(
        base_retriever=base_retriever,
        expander=ArticleContextExpander(),
        expand_top_n=1,
    )

    baseline_chunks = base_retriever.search(QUESTION, k=5)
    expanded_chunks = expanded_retriever.search(QUESTION, k=5)

    print("\n=== CONTEXT COMPARISON ===")
    print(
        "Baseline characters:",
        sum(len(c["text"]) for c in baseline_chunks),
    )
    print(
        "Expanded characters:",
        sum(len(c["text"]) for c in expanded_chunks),
    )
    print(
        "Article order:",
        [c["article_number"] for c in expanded_chunks],
    )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
        timeout=60,
        max_retries=1,
    )

    print("\nCalling Mistral with expanded context...")

    result = answer_question(
        question=QUESTION,
        retriever=expanded_retriever,
        client=client,
        model=model,
        k=5,
    )

    print("\n=== EXPANDED RAG RESULT ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
