"""MCP Server for WebNovel Knowledge Base."""
import os
import sys

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from webnovel_kb.config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_CHAT_BASE_URL,
    LLM_EMBEDDING_MODEL, LLM_RERANK_MODEL,
    LLM_EMBEDDING_DIMENSIONS, LLM_CHAT_MODEL,
    LOG_LEVEL, LOG_DIR, LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT,
    LOG_CONSOLE_LEVEL, LOG_FILE_LEVEL,
)
from webnovel_kb.utils.logging_config import setup_logging, get_logger
from webnovel_kb.oauth_auth import create_token_verifier, OAUTH_PATHS

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

MCP_API_KEY = os.environ.get("MCP_API_KEY", "")
MCP_OAUTH_ISSUER_URL = os.environ.get("MCP_OAUTH_ISSUER_URL", "")

_token_verifier = None


class BearerAuthMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        if scope["path"] in OAUTH_PATHS:
            await self.app(scope, receive, send)
            return

        if not MCP_API_KEY and not _token_verifier:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()

        if not auth.startswith("Bearer "):
            response = JSONResponse(
                {"error": "Missing Authorization header"}, status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        token = auth[7:]

        if MCP_API_KEY and token == MCP_API_KEY:
            await self.app(scope, receive, send)
            return

        if _token_verifier:
            result = await _token_verifier.verify_token(token)
            if result is not None:
                await self.app(scope, receive, send)
                return

        response = JSONResponse(
            {"error": "Invalid token"}, status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive, send)


try:
    from mcp.server.fastmcp import FastMCP
    from webnovel_kb.core.knowledge_base import WebNovelKnowledgeBase
    from webnovel_kb.api.mcp_tools import MCPTools

    use_reranker = bool(LLM_RERANK_MODEL)
    kb = WebNovelKnowledgeBase(use_reranker=use_reranker)
    logger.info(f"WebNovel Knowledge Base initialized: {kb.get_stats()}")

    host = os.environ.get("MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("MCP_PORT", "8765"))

    if MCP_OAUTH_ISSUER_URL:
        _token_verifier = create_token_verifier()
        logger.info(f"OAuth token verifier ready: issuer={MCP_OAUTH_ISSUER_URL}")

    class _OAuthFastMCP(FastMCP):
        async def run_streamable_http_async(self):
            from starlette.routing import Route
            from webnovel_kb.oauth_auth import oauth_well_known, oauth_authorize, oauth_token
            import uvicorn

            starlette_app = self.streamable_http_app()

            if MCP_OAUTH_ISSUER_URL:
                oauth_routes = [
                    Route("/.well-known/oauth-authorization-server", oauth_well_known, methods=["GET"]),
                    Route("/authorize", oauth_authorize, methods=["GET"]),
                    Route("/token", oauth_token, methods=["POST"]),
                ]
                starlette_app.routes[:0] = oauth_routes
                logger.info("OAuth routes added to app")

            config = uvicorn.Config(
                starlette_app,
                host=self.settings.host,
                port=self.settings.port,
                log_level=self.settings.log_level.lower(),
            )
            server = uvicorn.Server(config)
            await server.serve()

    mcp = _OAuthFastMCP("webnovel-kb", host=host, port=port)
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
