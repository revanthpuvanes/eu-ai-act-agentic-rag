"""Langfuse tracing for EU AI Act agents."""

from typing import Any, Callable


def trace_agent_run(
    agent: Any,
    question: str,
    langfuse_client: Any,
    *,
    question_id: str | None = None,
    enabled: bool = True,
    execute: Callable[[str], dict] | None = None,
) -> dict:
    """Execute an agent once, optionally recording a Langfuse trace."""

    if execute is None:
        execute = getattr(agent, "run", None)

    if execute is None:
        raise TypeError(
            "No execution function supplied. "
            "Pass execute=agent.<actual_method>."
        )

    if not enabled:
        return execute(question)

    from langfuse import propagate_attributes

    with langfuse_client.start_as_current_observation(
        as_type="span",
        name="eu-ai-act-llm-agent",
        input={
            "question": question,
            "question_id": question_id,
        },
    ) as root:
        with propagate_attributes(
            tags=["eu-ai-act", "agentic-rag", "development"],
        ):
            result = execute(question)

        assessment = result.get("assessment") or {}

        root.update(
            output={
                "answer_preview": str(
                    result.get("answer", "")
                )[:500],
                "cited_articles": result.get(
                    "cited_articles", []
                ),
                "abstained": result.get(
                    "abstained", False
                ),
            },
            metadata={
                "question_id": question_id or "",
                "model": result.get("model", ""),
                "provider": result.get("provider", ""),
                "retrieved_articles": result.get(
                    "retrieved_articles", []
                ),
                "assessment_action": assessment.get(
                    "action"
                ),
                "assessment_reason": assessment.get(
                    "reason"
                ),
                "action_taken": result.get(
                    "action_taken"
                ),
                "retrieval_seconds": result.get(
                    "retrieval_seconds"
                ),
                "assessment_seconds": result.get(
                    "assessment_seconds"
                ),
                "evidence_action_seconds": result.get(
                    "evidence_action_seconds"
                ),
                "generation_seconds": result.get(
                    "generation_seconds"
                ),
                "total_seconds": result.get(
                    "total_seconds"
                ),
                "assessor_input_tokens": result.get(
                    "assessor_input_tokens"
                ),
                "assessor_output_tokens": result.get(
                    "assessor_output_tokens"
                ),
                "generation_input_tokens": result.get(
                    "generation_input_tokens"
                ),
                "generation_output_tokens": result.get(
                    "generation_output_tokens"
                ),
                "total_tokens": result.get(
                    "total_tokens"
                ),
            },
        )

        return result
