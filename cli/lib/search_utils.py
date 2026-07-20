SCORE_PRECISION = 4


def format_search_result(
    doc: dict, score: float, metadata: dict | None = None
) -> dict:
    return {
        "id": doc["id"],
        "title": doc["title"],
        "document": doc.get("description", "")[:100],
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata or {},
    }
