"""分析类工具：style_analysis。"""


def _compact_style(result: dict) -> dict:
    """精简风格分析结果，移除大段示例文本。"""
    if "error" in result:
        return result
    # 精简 section_breakdown 中的 sample_text
    for sec in result.get("section_breakdown", []):
        sec.pop("sample_text", None)
    # 精简 sample_passages，只保留 chapter 和 position
    result.pop("sample_passages", None)
    return result


def register_analysis_tools(mcp, kb, safe_async):

    @mcp.tool()
    async def style_analysis(novel_titles: str, output_format: str = "compact") -> dict:
        """Analyze and compare writing styles (sentence length, dialogue ratio, humour, etc) for novels."""
        if isinstance(novel_titles, str):
            titles = [t.strip() for t in novel_titles.split(",") if t.strip()]
        else:
            titles = novel_titles

        if len(titles) == 1:
            result = await safe_async("analyze_style", kb.analyze_style, titles[0])
        else:
            result = await safe_async("compare_styles", kb.compare_styles, titles)

        if output_format == "compact" and isinstance(result, dict):
            return _compact_style(result)
        return result
