"""Run one Langfuse-traced EU AI Act agent question."""

import json
import os

from dotenv import load_dotenv
from langfuse import get_client
from openai import OpenAI

from src.observability.traced_agent import TracedLLMEvidenceAgent
from src.observability.langfuse_tracing import trace_agent_run
from src.retrieval.context_expansion import ArticleContextExpander
from src.retrieval.hybrid import HybridRetriever


def main():
    load_dotenv()

    client = OpenAI(
        api_key=os.environ["MISTRAL_API_KEY"],
        base_url="https://api.mistral.ai/v1",
        timeout=60,
        max_retries=1,
    )

    model = os.getenv(
        "MISTRAL_MODEL",
        "mistral-small-2603",
    )

    retriever = HybridRetriever.from_jsonl(
        "data/processed/chunks.jsonl"
    )

    expander = ArticleContextExpander(
        articles_path="data/processed/articles.jsonl",
        max_article_chars=15000,
        max_context_chars=22000,
    )

    langfuse = get_client()

    agent = TracedLLMEvidenceAgent(
        langfuse_client=langfuse,
        retriever=retriever,
        expander=expander,
        client=client,
        model=model,
    )

    result = trace_agent_run(
        agent,
        "How does the EU AI Act define an AI system?",
        langfuse,
        question_id="Q002",
        execute=agent.answer,
    )

    langfuse.flush()

    print(
        json.dumps(
            {
                "question": result["question"],
                "answer": result["answer"],
                "assessment": result["assessment"],
                "action_taken": result["action_taken"],
                "cited_articles": result["cited_articles"],
                "total_tokens": result["total_tokens"],
                "total_seconds": result["total_seconds"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nTrace submitted to Langfuse.")


if __name__ == "__main__":
    main()
