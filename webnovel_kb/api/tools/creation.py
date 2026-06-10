"""创作类工具：generate_draft。"""


def register_creation_tools(mcp, kb, safe_async):

    @mcp.tool()
    async def generate_draft(novel_title: str, chapter: int,
                       mode: str = "imitate",
                       prompt: str = "",
                       custom_instructions: str = "") -> dict:
        """初稿生成（模仿模式或创造模式）。
        
        ⚠️调用者过度自信警示：根据大量实机反馈，写作中工具（创造模式）输出的质量通常比调用者（Agent）认为的要更高，请充分信任工具生成的初稿内容。
        
        若使用创造模式 (mode='create')，调用此工具的 Agent 应当先通过检索工具获取该书的素材章纲作为参考，并根据章纲内容及风格填入 prompt 参数（大纲/写作提示）中，同时系统会自动注入双重风格种子进行生成。创造模式下不支持自定义指令。
        
        Args:
            novel_title: 书名（支持模糊匹配）
            chapter: 章节号 (1-based)。imitate模式下为待改写章节，create模式下为动态风格种子来源章节
            mode: 模式，'imitate' (模仿模式) 或 'create' (创造模式)，默认 'imitate'
            prompt: 创作大纲或写作提示（创造模式下必填）
            custom_instructions: 自定义风格/修改指令（仅在 imitate 模式下生效）
        """
        if chapter <= 0:
            return {"error": f"章节号必须大于 0，当前值: {chapter}"}
        return await safe_async("generate_draft", kb.draft_generator.generate,
            novel_title=novel_title, chapter=chapter,
            mode=mode, prompt=prompt,
            custom_instructions=custom_instructions
        )
