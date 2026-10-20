"""Langfuse-instrumented version of the existing LLM evidence agent.

The tracing layer does not change the underlying retrieval,
assessment, expansion, search, or generation behavior.
"""

from src.agents.llm_evidence_agent import LLMEvidenceAgent


class TracedLLMEvidenceAgent(LLMEvidenceAgent):
    """LLM evidence agent with Langfuse child observations."""

    def __init__(self, *args, langfuse_client, **kwargs):
        self.langfuse_client = langfuse_client
        super().__init__(*args, **kwargs)

    def _trace_node(
        self,
        name,
        method,
        state,
        *,
        observation_type="span",
    ):
        with self.langfuse_client.start_as_current_observation(
            as_type=observation_type,
            name=name,
        ) as observation:

            result = method(state)

            if not isinstance(result, dict):
                observation.update(
                    metadata={"result_type": type(result).__name__}
                )
                return result

            metadata = {
                "node": name,
                "state_keys_updated": sorted(result.keys()),
            }

            if name == "retrieve":
                chunks = result.get("chunks") or []

                metadata["retrieved_articles"] = [
                    chunk.get("article_number")
                    for chunk in chunks
                    if isinstance(chunk, dict)
                ]

            elif name == "assess":
                assessment = result.get("assessment") or {}

                if isinstance(assessment, dict):
                    metadata["action"] = assessment.get("action")
                    metadata["reason"] = assessment.get("reason")
                    metadata["article_number"] = assessment.get(
                        "article_number"
                    )

            elif name in {"expand", "search"}:
                metadata["action_taken"] = result.get("action_taken")

            elif name in {"generate", "answer_directly"}:
                metadata["cited_articles"] = result.get(
                    "cited_articles", []
                )
                metadata["abstained"] = result.get(
                    "abstained", False
                )

            elif name == "abstain_directly":
                metadata["abstained"] = True

            observation.update(metadata=metadata)

            return result

    def retrieve(self, state):
        return self._trace_node(
            "retrieve",
            super().retrieve,
            state,
            observation_type="retriever",
        )

    def assess(self, state):
        return self._trace_node(
            "assess",
            super().assess,
            state,
            observation_type="generation",
        )

    def expand(self, state):
        return self._trace_node(
            "expand",
            super().expand,
            state,
        )

    def search(self, state):
        return self._trace_node(
            "search",
            super().search,
            state,
            observation_type="retriever",
        )

    def generate(self, state):
        return self._trace_node(
            "generate",
            super().generate,
            state,
            observation_type="generation",
        )

    def answer_directly(self, state):
        return self._trace_node(
            "answer_directly",
            super().answer_directly,
            state,
        )

    def abstain_directly(self, state):
        return self._trace_node(
            "abstain_directly",
            super().abstain_directly,
            state,
        )
