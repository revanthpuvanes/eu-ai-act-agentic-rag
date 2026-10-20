import json
from pathlib import Path

from src.evaluation.evaluate_retrieval import evaluate, load_jsonl
from src.retrieval.bm25 import BM25Retriever


class DiverseRetriever:
    def __init__(self, base, limit):
        self.base = base
        self.limit = limit

    def search(self, query, k=5):
        return self.base.search(
            query,
            k=k,
            max_chunks_per_article=self.limit,
        )


questions = [
    q for q in load_jsonl("data/eval/questions.jsonl")
    if q["split"] == "dev"
]

assert len(questions) == 20

base = BM25Retriever.from_jsonl()

strategies = {
    "standard": base,
    "max_2_chunks_per_article": DiverseRetriever(base, 2),
    "max_1_chunk_per_article": DiverseRetriever(base, 1),
}

output = {}

for name, retriever in strategies.items():
    summary, details = evaluate(
        retriever,
        questions,
        ks=(5, 10),
    )

    output[name] = {
        "summary": summary,
        "questions": details,
    }

    print(f"\n=== {name.upper()} ===")
    for metric, value in summary.items():
        print(f"{metric}: {value}")

path = Path("evals/results/bm25_diversity_dev.json")
path.parent.mkdir(parents=True, exist_ok=True)

path.write_text(
    json.dumps(output, indent=2) + "\n",
    encoding="utf-8",
)

print("\nSaved:", path)
