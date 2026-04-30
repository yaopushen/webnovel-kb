import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webnovel_kb.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_CHAT_BASE_URL,
    LLM_EMBEDDING_MODEL,
    LLM_RERANK_MODEL,
    LLM_EMBEDDING_DIMENSIONS,
    LLM_CHAT_MODEL,
)

os.environ.setdefault("LLM_API_KEY", LLM_API_KEY or "")
os.environ.setdefault("LLM_BASE_URL", LLM_BASE_URL or "")
os.environ.setdefault("LLM_CHAT_BASE_URL", LLM_CHAT_BASE_URL or "")
os.environ.setdefault("LLM_EMBEDDING_MODEL", LLM_EMBEDDING_MODEL or "")
os.environ.setdefault("LLM_RERANK_MODEL", LLM_RERANK_MODEL or "")
os.environ.setdefault("LLM_EMBEDDING_DIMENSIONS", str(LLM_EMBEDDING_DIMENSIONS))
os.environ.setdefault("LLM_CHAT_MODEL", LLM_CHAT_MODEL or "")

from webnovel_kb.server import mcp

if __name__ == '__main__':
    transport = os.environ.get('MCP_TRANSPORT', 'stdio')
    host = os.environ.get('MCP_HOST', '127.0.0.1')
    port = int(os.environ.get('MCP_PORT', '8765'))

    if transport == 'sse':
        mcp.run(transport='sse', host=host, port=port)
    elif transport == 'streamable-http':
        mcp.run(transport='streamable-http', host=host, port=port)
    else:
        mcp.run()
