"""知识类工具：add_knowledge。"""


def register_knowledge_tools(mcp, kb, safe_async):

    @mcp.tool()
    async def add_knowledge(content: str, title: str, category: str = "research",
                      tags: list = [], source: str = "",
                      analyze: bool = True) -> dict:
        """Add research/technique/guideline knowledge to the Agent knowledge base."""
        return await safe_async("add_knowledge", kb.knowledge_store.add,
            content=content, title=title, category=category,
            tags=tags, source=source, analyze=analyze
        )
