import json
from pathlib import Path

from src.evaluation.evaluate_retrieval import evaluate, load_jsonl
from src.retrieval.hybrid import HybridRetriever


questions = [
    q
    for q in load_jsonl("data/eval/questions.jsonl")
    if q["split"] == "dev"
]

assert len(questions) == 20

print("Loading BM25 and semantic retrievers...")
retriever = HybridRetriever.from_jsonl()

summary, details = evaluate(
    retriever,
    questions,
    ks=(5, 10),
)

output = {
    "retriever": "hybrid_rrf",
    "split": "dev",
    "configuration": {
        "rrf_k": 60,
        "candidate_k": 50,
        "max_chunks_per_article": 1,
    },
    "summary": summary,
    "questions": details,
}

path = Path("evals/results/hybrid_dev.json")
path.parent.mkdir(parents=True, exist_ok=True)

path.write_text(
    json.dumps(output, indent=2) + "\n",
    encoding="utf-8",
)

print("\n=== HYBRID RRF DEV EVALUATION ===")
for metric, value in summary.items():
    print(f"{metric}: {value}")

print("\nLowest Recall@5 questions:")
ranked = sorted(
    (
        row for row in details
        if "article_recall@5" in row
    ),
    key=lambda row: row["article_recall@5"],
)

for row in ranked[:5]:
    print(
        row["id"],
        "gold=", row["gold_articles"],
        "retrieved=", row["retrieved_articles"][:5],
        "recall=", row["article_recall@5"],
    )

print("\nSaved:", path)
