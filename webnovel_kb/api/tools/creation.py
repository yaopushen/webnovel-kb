"""创作类工具：generate_sample。"""


def register_creation_tools(mcp, kb, safe_async):

    @mcp.tool()
    async def generate_sample(novel_title: str, chapter: int,
                        rewrite_level: str = "minimal",
                        custom_instructions: str = "",
                        preserve_elements: list = []) -> dict:
        """Generate a rewritten copy of a chapter using classic tropes and custom rules."""
        if chapter <= 0:
            return {"error": f"章节号必须大于 0，当前值: {chapter}"}
        return await safe_async("generate_sample", kb.sample_generator.generate,
            novel_title=novel_title, chapter=chapter,
            rewrite_level=rewrite_level,
            custom_instructions=custom_instructions,
            preserve_elements=preserve_elements or None
        )
