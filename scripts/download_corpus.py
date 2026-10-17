
from pathlib import Path
import requests

URL = "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng"
OUTPUT = Path("data/raw/eu_ai_act.html")


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        URL,
        timeout=30,
        headers={
            "User-Agent": "EU-AI-Act-RAG-Research/0.1"
        },
    )

    response.raise_for_status()

    if not response.content:
        raise ValueError("Downloaded file is empty")

    OUTPUT.write_bytes(response.content)

    print(f"Downloaded {len(response.content):,} bytes")
    print(f"Saved to: {OUTPUT.resolve()}")
    print(f"Content type: {response.headers.get('Content-Type')}")


if __name__ == "__main__":
    main()
