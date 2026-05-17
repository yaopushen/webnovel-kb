"""WebNovel Knowledge Base - MCP Server Entry Point."""
__version__ = "1.5"

from webnovel_kb.core.knowledge_base import WebNovelKnowledgeBase
from webnovel_kb.api.clients import RemoteEmbeddingFunction, RemoteReranker, RemoteChatClient
from webnovel_kb.api.mcp_tools import MCPTools

__all__ = [
    "WebNovelKnowledgeBase",
    "RemoteEmbeddingFunction",
    "RemoteReranker", 
    "RemoteChatClient",
    "MCPTools",
]
