"""FastAPI service for the live EU AI Act LangGraph agent."""

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from src.agents.llm_evidence_agent import LLMEvidenceAgent
from src.api.provenance import format_evidence
from src.retrieval.context_expansion import ArticleContextExpander
from src.retrieval.hybrid import HybridRetriever


logger = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 1000

MAX_CONCURRENT_REQUESTS = int(
    os.getenv("MAX_CONCURRENT_REQUESTS", "4")
)

_inference_semaphore = threading.BoundedSemaphore(
    MAX_CONCURRENT_REQUESTS
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=10,
        max_length=MAX_QUESTION_LENGTH,
    )


def build_agent():
    """Reuse the exact agent configuration from the benchmark."""
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not configured")

    model = os.getenv("MISTRAL_MODEL", "mistral-small-2603")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
        timeout=60,
        max_retries=1,
    )

    retriever = HybridRetriever.from_jsonl()

    expander = ArticleContextExpander(
        max_article_chars=15000,
        max_context_chars=22000,
    )

    return LLMEvidenceAgent(
        retriever=retriever,
        expander=expander,
        client=client,
        model=model,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = build_agent()
    yield


from src.api.evidence import router as evidence_router
app = FastAPI(
    title="EU AI Act Intelligence Lab",
    description="Live evidence-grounded legal research using LangGraph.",
    version="0.2.0",
    lifespan=lifespan,
)
app.include_router(evidence_router)



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,"
            "http://127.0.0.1:3000,"
            "http://localhost:3001,"
            "http://127.0.0.1:3001",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def format_result(state: dict, question: str, elapsed: float, model: str):
    """Convert final LangGraph state into a public response."""
    assessor_input = state.get("assessor_input_tokens", 0)
    assessor_output = state.get("assessor_output_tokens", 0)
    generation_input = state.get("generation_input_tokens", 0)
    generation_output = state.get("generation_output_tokens", 0)

    return {
        "question": question,
        "answer": state["answer"],
        "cited_articles": state["cited_articles"],
        "abstained": state["abstained"],
        "retrieved_articles": [
            int(chunk["article_number"])
            for chunk in state["final_chunks"]
        ],
        "evidence": format_evidence(state["final_chunks"]),
        "assessment": state.get("assessment"),
        "action_taken": state.get("action_taken"),
        "input_tokens": assessor_input + generation_input,
        "output_tokens": assessor_output + generation_output,
        "total_tokens": (
            assessor_input
            + assessor_output
            + generation_input
            + generation_output
        ),
        "retrieval_seconds": state.get("retrieval_seconds", 0),
        "assessment_seconds": state.get("assessment_seconds", 0),
        "evidence_action_seconds": state.get(
            "evidence_action_seconds", 0
        ),
        "generation_seconds": state.get("generation_seconds", 0),
        "total_seconds": round(elapsed, 3),
        "model": model,
        "provider": "mistral",
        "strategy": "llm_langgraph",
    }


def run_agent(agent, question: str):
    """Execute the graph once and format its final state."""
    started = time.perf_counter()

    state = agent.graph.invoke({"question": question})

    return format_result(
        state=state,
        question=question,
        elapsed=time.perf_counter() - started,
        model=agent.model,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "eu-ai-act-intelligence-lab",
    }


@app.post("/ask")
def ask(payload: AskRequest):
    if not _inference_semaphore.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="The agent is busy. Try again shortly.",
        )

    try:
        return run_agent(app.state.agent, payload.question)

    except Exception:
        logger.exception("Agent request failed")
        raise HTTPException(
            status_code=502,
            detail="The research request could not be completed.",
        )

    finally:
        _inference_semaphore.release()


def sse_event(event_type: str, data: dict[str, Any]):
    """Encode a Server-Sent Event."""
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
    )


@app.post("/ask/stream")
def ask_stream(payload: AskRequest):
    if not _inference_semaphore.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="The agent is busy. Try again shortly.",
        )

    agent = app.state.agent

    def generate():
        started = time.perf_counter()
        final_state = {"question": payload.question}

        try:
            yield sse_event(
                "started",
                {"message": "Live research started"},
            )

            for update in agent.graph.stream(
                {"question": payload.question},
                stream_mode="updates",
            ):
                for node_name, node_state in update.items():
                    if not isinstance(node_state, dict):
                        continue

                    final_state.update(node_state)

                    event_data = {
                        "node": node_name,
                        "elapsed_seconds": round(
                            time.perf_counter() - started, 3
                        ),
                    }

                    if "assessment" in node_state:
                        event_data["assessment"] = node_state["assessment"]

                    if "action_taken" in node_state:
                        event_data["action_taken"] = node_state[
                            "action_taken"
                        ]

                    if "final_chunks" in node_state:
                        event_data["retrieved_articles"] = [
                            int(chunk["article_number"])
                            for chunk in node_state["final_chunks"]
                        ]

                    yield sse_event("node_completed", event_data)

            result = format_result(
                state=final_state,
                question=payload.question,
                elapsed=time.perf_counter() - started,
                model=agent.model,
            )

            yield sse_event("completed", result)

        except Exception:
            logger.exception("Streaming agent request failed")
            yield sse_event(
                "error",
                {
                    "message": (
                        "The research request could not be completed."
                    )
                },
            )

        finally:
            _inference_semaphore.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
