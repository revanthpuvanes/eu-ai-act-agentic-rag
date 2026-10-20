"""LangGraph RAG agent with bounded, evidence-aware context expansion.

The initial policy is deterministic and intended as a testable baseline.
It does not establish that evidence is legally or semantically sufficient.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from src.generation.naive_rag import (
    SYSTEM_PROMPT,
    format_context,
    validate_response,
)
from src.retrieval.context_expansion import ArticleContextExpander
from src.retrieval.hybrid import HybridRetriever


BROAD_QUESTION_PATTERN = re.compile(
    r"\b("
    r"which .*practices|"
    r"what .*obligations|"
    r"list|enumerate|"
    r"all .*requirements|"
    r"what .*prohibitions|"
    r"what .*exceptions"
    r")\b",
    flags=re.IGNORECASE,
)


class AgentState(TypedDict, total=False):
    question: str
    chunks: list[dict[str, Any]]
    expanded_chunks: list[dict[str, Any]]
    needs_expansion: bool
    assessment_reason: str
    answer: str
    cited_articles: list[int]
    abstained: bool
    retrieved_articles: list[int]
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    retrieval_seconds: float
    assessment_seconds: float
    expansion_seconds: float
    generation_seconds: float
    total_seconds: float
    expansion_performed: bool
    model: str
    provider: str


class EvidenceAwareAgent:
    def __init__(
        self,
        retriever,
        expander,
        client,
        model="mistral-small-2603",
        k=5,
        expansion_threshold=0.5,
    ):
        self.retriever = retriever
        self.expander = expander
        self.client = client
        self.model = model
        self.k = k
        self.expansion_threshold = expansion_threshold
        self.graph = self._build_graph()

    def retrieve(self, state: AgentState) -> dict:
        started = time.perf_counter()

        chunks = self.retriever.search(
            state["question"],
            k=self.k,
        )

        return {
            "chunks": chunks,
            "retrieved_articles": [
                int(chunk["article_number"])
                for chunk in chunks
            ],
            "retrieval_seconds": round(
                time.perf_counter() - started, 3
            ),
        }

    def assess_evidence(self, state: AgentState) -> dict:
        started = time.perf_counter()

        chunks = state.get("chunks", [])

        if not chunks:
            return {
                "needs_expansion": False,
                "assessment_reason": "No evidence retrieved",
                "assessment_seconds": round(
                    time.perf_counter() - started, 3
                ),
            }

        question = state["question"]
        broad_question = bool(
            BROAD_QUESTION_PATTERN.search(question)
        )

        top_chunk = chunks[0]
        article_number = int(top_chunk["article_number"])

        article = self.expander.articles.get(article_number)

        if article is None:
            return {
                "needs_expansion": False,
                "assessment_reason": "Top article unavailable",
                "assessment_seconds": round(
                    time.perf_counter() - started, 3
                ),
            }

        article_length = len(article["text"])
        chunk_length = len(top_chunk["text"])

        coverage = (
            chunk_length / article_length
            if article_length
            else 1.0
        )

        needs_expansion = (
            broad_question
            and coverage < self.expansion_threshold
            and article_length <= self.expander.max_article_chars
        )

        reason = (
            f"broad_question={broad_question}; "
            f"top_article={article_number}; "
            f"coverage={coverage:.3f}; "
            f"expand={needs_expansion}"
        )

        return {
            "needs_expansion": needs_expansion,
            "assessment_reason": reason,
            "assessment_seconds": round(
                time.perf_counter() - started, 3
            ),
        }

    def expand_evidence(self, state: AgentState) -> dict:
        started = time.perf_counter()

        expanded = self.expander.expand(
            state["chunks"],
            expand_top_n=1,
        )

        expansion_performed = any(
            chunk.get("context_expanded", False)
            for chunk in expanded
        )

        return {
            "expanded_chunks": expanded,
            "expansion_performed": expansion_performed,
            "expansion_seconds": round(
                time.perf_counter() - started, 3
            ),
        }

    def generate(self, state: AgentState) -> dict:
        started = time.perf_counter()

        chunks = state.get(
            "expanded_chunks",
            state.get("chunks", []),
        )

        if not chunks:
            return {
                "answer": "Insufficient evidence to answer.",
                "cited_articles": [],
                "abstained": True,
                "generation_seconds": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "model": self.model,
                "provider": "mistral",
            }

        context = format_context(chunks)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{state['question']}\n\n"
                        f"Retrieved legal evidence:\n{context}"
                    ),
                },
            ],
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("Mistral returned an empty response")

        result = json.loads(content)

        # Use the same validation as naive RAG.
        validate_response(result, chunks)

        usage = response.usage

        return {
            "answer": result["answer"],
            "cited_articles": result["cited_articles"],
            "abstained": result["abstained"],
            "generation_seconds": round(
                time.perf_counter() - started, 3
            ),
            "input_tokens": (
                usage.prompt_tokens if usage else None
            ),
            "output_tokens": (
                usage.completion_tokens if usage else None
            ),
            "total_tokens": (
                usage.total_tokens if usage else None
            ),
            "model": self.model,
            "provider": "mistral",
        }

    def route_after_assessment(self, state: AgentState) -> str:
        return (
            "expand"
            if state.get("needs_expansion", False)
            else "generate"
        )

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("retrieve", self.retrieve)
        builder.add_node("assess", self.assess_evidence)
        builder.add_node("expand", self.expand_evidence)
        builder.add_node("generate", self.generate)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "assess")

        builder.add_conditional_edges(
            "assess",
            self.route_after_assessment,
            {
                "expand": "expand",
                "generate": "generate",
            },
        )

        builder.add_edge("expand", "generate")
        builder.add_edge("generate", END)

        return builder.compile()

    def answer(self, question: str) -> dict:
        started = time.perf_counter()

        state = self.graph.invoke({
            "question": question,
            "expansion_performed": False,
        })

        state["total_seconds"] = round(
            time.perf_counter() - started, 3
        )

        return {
            "question": question,
            "answer": state["answer"],
            "cited_articles": state["cited_articles"],
            "abstained": state["abstained"],
            "retrieved_articles": state.get(
                "retrieved_articles", []
            ),
            "needs_expansion": state.get(
                "needs_expansion", False
            ),
            "expansion_performed": state.get(
                "expansion_performed", False
            ),
            "assessment_reason": state.get(
                "assessment_reason", ""
            ),
            "retrieval_seconds": state.get(
                "retrieval_seconds", 0.0
            ),
            "assessment_seconds": state.get(
                "assessment_seconds", 0.0
            ),
            "expansion_seconds": state.get(
                "expansion_seconds", 0.0
            ),
            "generation_seconds": state.get(
                "generation_seconds", 0.0
            ),
            "total_seconds": state["total_seconds"],
            "input_tokens": state.get("input_tokens"),
            "output_tokens": state.get("output_tokens"),
            "total_tokens": state.get("total_tokens"),
            "model": state["model"],
            "provider": state["provider"],
        }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()

    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise SystemExit("MISTRAL_API_KEY missing")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
        timeout=60,
        max_retries=1,
    )

    agent = EvidenceAwareAgent(
        retriever=HybridRetriever.from_jsonl(),
        expander=ArticleContextExpander(),
        client=client,
        model=os.getenv(
            "MISTRAL_MODEL",
            "mistral-small-2603",
        ),
    )

    result = agent.answer(args.question)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
