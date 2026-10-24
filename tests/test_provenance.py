from src.api.provenance import format_evidence


def test_format_evidence_preserves_source_text():
    chunks = [
        {
            "article_number": 16,
            "text": "Providers of high-risk AI systems shall...",
            "start_char": 100,
            "end_char": 145,
            "source_url": "https://example.org/article-16",
        }
    ]

    result = format_evidence(chunks)

    assert len(result) == 1
    assert result[0]["article_number"] == 16
    assert result[0]["text"] == chunks[0]["text"]
    assert result[0]["start_char"] == 100
    assert result[0]["end_char"] == 145


def test_format_evidence_ignores_empty_text():
    chunks = [
        {"article_number": 16, "text": ""},
        {"article_number": 17, "text": "Actual evidence"},
    ]

    result = format_evidence(chunks)

    assert len(result) == 1
    assert result[0]["article_number"] == 17
