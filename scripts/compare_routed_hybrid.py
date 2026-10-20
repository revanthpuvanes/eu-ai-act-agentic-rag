import json
from pathlib import Path

from src.evaluation.evaluate_retrieval import evaluate, load_jsonl
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.routed_hybrid import RoutedHybridRetriever


questions = [
    q
    for q in load_jsonl("data/eval/questions.jsonl")
    if q["split"] == "dev"
]

assert len(questions) == 20

print("Loading retrievers...")

base = HybridRetriever.from_jsonl()
routed = RoutedHybridRetriever(base)

output = {}

for name, retriever in {
    "hybrid_original": base,
    "hybrid_with_legal_router": routed,
}.items():
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

path = Path("evals/results/routed_hybrid_dev.json")
path.parent.mkdir(parents=True, exist_ok=True)

path.write_text(
    json.dumps(output, indent=2) + "\n",
    encoding="utf-8",
)

print("\nSaved:", path)

print("\n=== APPLICATION DATE SMOKE TEST ===")

query = "When does the European Union AI Act become applicable?"

for name, retriever in {
    "original": base,
    "routed": routed,
}.items():
    articles = [
        result["article_number"]
        for result in retriever.search(query, k=5)
    ]

    print(name, articles)
