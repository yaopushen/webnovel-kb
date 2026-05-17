"""Output formatting utilities for search results."""
from typing import List, Dict, Any


def format_search_results(
    raw_results: List[Dict[str, Any]],
    output_format: str = "compact",
    max_content_length: int = 0,
    dedupe: bool = False,
    dedupe_fn=None
) -> List[Any]:
    """
    格式化搜索结果。
    
    Args:
        raw_results: 原始搜索结果列表
        output_format: 输出格式
            - raw: 完整结构化数据（调试用）
            - compact: "[来源] 内容..." 作家可直接阅读
            - clean: 仅纯文本内容，无来源标记
        max_content_length: 每条内容最大字数(0=不限制)
        dedupe: 是否去重
        dedupe_fn: 去重函数
    
    Returns:
        格式化后的结果列表
    """
    if not raw_results:
        return raw_results
    if len(raw_results) == 1 and "status" in raw_results[0]:
        return raw_results

    items = dedupe_fn(raw_results) if dedupe and dedupe_fn else raw_results

    if output_format == "raw":
        return items

    if output_format == "clean":
        output = []
        for item in items:
            text = item.get("text") or item.get("description") or ""
            if max_content_length > 0 and len(text) > max_content_length:
                text = text[:max_content_length] + "…"
            output.append(text)
        return output

    output = []
    for item in items:
        source = item.get("source", "")
        if not source:
            title = ""
            chapter = ""
            meta = item.get("metadata", {})
            if meta:
                title = meta.get("title", "")
                chapter = meta.get("chapter_title", "")
            if not title:
                title = item.get("source_novel", "") or item.get("novel_title", "")
            source_parts = []
            if title:
                source_parts.append(f"《{title}》")
            if chapter:
                source_parts.append(chapter)
            source = " ".join(source_parts)
        
        text = item.get("text") or item.get("description") or ""
        if max_content_length > 0 and len(text) > max_content_length:
            text = text[:max_content_length] + "…"

        if source:
            output.append(f"[{source}] {text}")
        else:
            output.append(text)
    return output
