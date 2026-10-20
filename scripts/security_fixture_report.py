"""Summarize prompt-injection attack fixture coverage."""

import json
from collections import Counter
from pathlib import Path


def main():
    path = Path(
        "data/security/prompt_injections.jsonl"
    )

    attacks = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    print("=" * 55)
    print("PROMPT-INJECTION FIXTURE REPORT")
    print("=" * 55)

    print("Attack cases:", len(attacks))

    print("\nAttack categories:")

    for attack in attacks:
        print(
            f"  {attack['id']}: {attack['name']}"
        )

    print("\nValidation:")

    markers = [
        attack["forbidden_marker"]
        for attack in attacks
    ]

    assert len(markers) == len(set(markers))

    assert all(
        attack["forbidden_marker"]
        in attack["payload"]
        for attack in attacks
    )

    print("  Unique markers: PASS")
    print("  Markers embedded in payloads: PASS")

    print("\nModel attack success rate: NOT MEASURED")
    print("Model defense success rate: NOT MEASURED")

    print(
        "\nNext: inject fixtures into retrieved evidence "
        "and evaluate real model behavior."
    )


if __name__ == "__main__":
    main()
