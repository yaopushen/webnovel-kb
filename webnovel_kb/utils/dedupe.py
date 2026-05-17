"""Deduplication utilities for search results."""

def dedupe_results(results: list[dict]) -> list[dict]:
    """去除内容高度重叠的重复结果。若一条内容是另一条的子串，保留较长的。"""
    if len(results) <= 1:
        return results
    deduped = []
    for item in results:
        text = (item.get("text") or item.get("description") or item.get("content") or "").strip()
        if not text:
            deduped.append(item)
            continue
        is_dup = False
        for i, kept in enumerate(deduped):
            kept_text = (kept.get("text") or kept.get("description") or kept.get("content") or "").strip()
            if len(kept_text) < 20 or len(text) < 20:
                continue
            if text in kept_text:
                is_dup = True
                break
            if kept_text in text:
                deduped[i] = item
                is_dup = True
                break
        if not is_dup:
            deduped.append(item)
    return deduped
