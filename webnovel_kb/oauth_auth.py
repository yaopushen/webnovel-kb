import os
import time
import secrets
import hashlib
import base64
from urllib.parse import urlencode

import jwt
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from webnovel_kb.utils.logging_config import get_logger

logger = get_logger("oauth")

OAUTH_JWT_SECRET = os.environ.get("OAUTH_JWT_SECRET", secrets.token_hex(32)).encode()
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRY = int(os.environ.get("OAUTH_TOKEN_EXPIRY", "86400"))

CLIENT_ID = "mcp-client"

_auth_codes: dict[str, dict] = {}


def _compute_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


async def oauth_well_known(request: Request) -> JSONResponse:
    base = str(request.base_url).rstrip("/")
    return JSONResponse({
        "issuer": base,
        "authorization_endpoint": f"{base}/authorize",
        "token_endpoint": f"{base}/token",
        "token_endpoint_auth_methods_supported": ["none"],
        "code_challenge_methods_supported": ["S256"],
        "scopes_supported": ["mcp:read", "mcp:write"],
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
    })


async def oauth_authorize(request: Request) -> RedirectResponse:
    client_id = request.query_params.get("client_id", "")
    redirect_uri = request.query_params.get("redirect_uri", "")
    code_challenge = request.query_params.get("code_challenge", "")
    state = request.query_params.get("state")

    code = secrets.token_urlsafe(32)
    _auth_codes[code] = {
        "client_id": client_id,
        "code_challenge": code_challenge,
        "expires_at": time.time() + 600,
    }
    logger.info(f"OAuth authorize: client={client_id}")

    params = [("code", code)]
    if state:
        params.append(("state", state))
    return RedirectResponse(url=f"{redirect_uri}?{urlencode(params)}")


async def oauth_token(request: Request) -> JSONResponse:
    body = await request.form()
    grant_type = body.get("grant_type", "")
    code = body.get("code", "")
    code_verifier = body.get("code_verifier", "")
    client_id = body.get("client_id", "")

    if grant_type != "authorization_code":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    stored = _auth_codes.pop(code, None)
    if not stored or stored["expires_at"] < time.time():
        return JSONResponse({"error": "invalid_grant"}, status_code=400)

    if _compute_challenge(code_verifier) != stored["code_challenge"]:
        return JSONResponse({"error": "invalid_grant"}, status_code=400)

    now = int(time.time())
    token_payload = {
        "sub": client_id,
        "client_id": client_id,
        "scope": "mcp:read mcp:write",
        "iat": now,
        "exp": now + TOKEN_EXPIRY,
        "iss": "webnovel-kb-oauth",
    }
    access_token = jwt.encode(token_payload, OAUTH_JWT_SECRET, algorithm=JWT_ALGORITHM)

    logger.info(f"OAuth token issued: client={client_id}")

    return JSONResponse({
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": TOKEN_EXPIRY,
        "scope": "mcp:read mcp:write",
    })


def create_token_verifier():
    from mcp.server.auth.provider import AccessToken, TokenVerifier

    class _SimpleTokenVerifier:
        async def verify_token(self, token: str) -> AccessToken | None:
            try:
                payload = jwt.decode(
                    token, OAUTH_JWT_SECRET, algorithms=[JWT_ALGORITHM],
                    options={"require": ["exp", "client_id"]},
                )
                return AccessToken(
                    token=token,
                    client_id=payload["client_id"],
                    scopes=payload.get("scope", "").split(),
                    expires_at=payload.get("exp"),
                )
            except jwt.InvalidTokenError:
                return None

    return _SimpleTokenVerifier()


OAUTH_PATHS = {
    "/.well-known/oauth-authorization-server": {"GET": oauth_well_known},
    "/authorize": {"GET": oauth_authorize},
    "/token": {"POST": oauth_token},
}


class OAuthRoutingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path = scope["path"]
            if path in OAUTH_PATHS:
                method = scope.get("method", "")
                handler = OAUTH_PATHS[path].get(method)
                if handler:
                    request = Request(scope, receive)
                    response = await handler(request)
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)