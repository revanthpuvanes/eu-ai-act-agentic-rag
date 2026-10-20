import re


APPLICATION_PATTERNS = [
    r"\bwhen\b.*\b(apply|applies|applicable)\b",
    r"\bwhen\b.*\b(enter|enters)\s+into\s+force\b",
    r"\bapplication\s+date\b",
    r"\bdate\s+of\s+application\b",
    r"\bentry\s+into\s+force\b",
    r"\btransitional\s+provisions?\b",
]


def detect_relevant_articles(query):
    text = query.lower()

    articles = set()

    for pattern in APPLICATION_PATTERNS:
        if re.search(pattern, text):
            articles.add(113)
            break

    return articles
