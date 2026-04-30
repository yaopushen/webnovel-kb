"""MCP Server for WebNovel Knowledge Base."""
import os
import logging

from mcp.server.fastmcp import FastMCP

from webnovel_kb.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_CHAT_BASE_URL,
    LLM_EMBEDDING_MODEL, LLM_RERANK_MODEL,
    LLM_EMBEDDING_DIMENSIONS, LLM_CHAT_MODEL,
)
from webnovel_kb.core.knowledge_base import WebNovelKnowledgeBase
from webnovel_kb.api.mcp_tools import MCPTools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("webnovel-kb")

use_reranker = bool(LLM_RERANK_MODEL)
kb = WebNovelKnowledgeBase(use_reranker=use_reranker)
logger.info(f"WebNovel Knowledge Base initialized: {kb.get_stats()}")

host = os.environ.get("MCP_HOST", "127.0.0.1")
port = int(os.environ.get("MCP_PORT", "8765"))
mcp = FastMCP("webnovel-kb", host=host, port=port)
tools = MCPTools(mcp, kb)


def run():
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport in ("sse", "streamable-http"):
        mcp.run(transport=transport)
    else:
        mcp.run()


if __name__ == "__main__":
    run()
