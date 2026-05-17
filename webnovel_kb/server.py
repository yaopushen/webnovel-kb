"""MCP Server for WebNovel Knowledge Base."""
import os
import sys

from webnovel_kb.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_CHAT_BASE_URL,
    LLM_EMBEDDING_MODEL, LLM_RERANK_MODEL,
    LLM_EMBEDDING_DIMENSIONS, LLM_CHAT_MODEL,
    LOG_LEVEL, LOG_DIR, LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT,
    LOG_CONSOLE_LEVEL, LOG_FILE_LEVEL,
)
from webnovel_kb.utils.logging_config import setup_logging, get_logger

setup_logging(
    level=LOG_LEVEL,
    log_dir=LOG_DIR,
    log_file=LOG_FILE,
    max_bytes=LOG_MAX_BYTES,
    backup_count=LOG_BACKUP_COUNT,
    console_level=LOG_CONSOLE_LEVEL,
    file_level=LOG_FILE_LEVEL,
)

logger = get_logger("server")

try:
    from mcp.server.fastmcp import FastMCP
    from webnovel_kb.core.knowledge_base import WebNovelKnowledgeBase
    from webnovel_kb.api.mcp_tools import MCPTools

    use_reranker = bool(LLM_RERANK_MODEL)
    kb = WebNovelKnowledgeBase(use_reranker=use_reranker)
    logger.info(f"WebNovel Knowledge Base initialized: {kb.get_stats()}")

    host = os.environ.get("MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("MCP_PORT", "8765"))
    mcp = FastMCP("webnovel-kb", host=host, port=port)
    tools = MCPTools(mcp, kb)
except Exception as e:
    logger.critical(f"Failed to initialize server: {e}", exc_info=True)
    sys.exit(1)


def run():
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport in ("sse", "streamable-http"):
        mcp.run(transport=transport)
    else:
        mcp.run()


if __name__ == "__main__":
    run()
