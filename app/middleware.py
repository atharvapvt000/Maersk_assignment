"""Pure-ASGI request middleware: request id, access log, and last-resort 500 handling.

Implemented as raw ASGI (not BaseHTTPMiddleware) so that unhandled exceptions are caught
*here*, turned into the standard JSON 500 envelope, and still produce one access-log line with
status=500 — instead of escaping to Starlette's plain-text ServerErrorMiddleware.
"""

from __future__ import annotations

import json
import logging
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.errors import error_body
from app.logging_setup import request_id_var

access_log = logging.getLogger("app.access")
error_log = logging.getLogger("app.error")

REQUEST_ID_HEADER = b"x-request-id"


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(REQUEST_ID_HEADER, b"").decode("latin-1").strip()
        request_id = incoming[:64] if incoming else uuid.uuid4().hex
        token = request_id_var.set(request_id)

        status_code = 500
        response_started = False
        start = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_started = True
                message.setdefault("headers", [])
                message["headers"] = [
                    *message["headers"],
                    (REQUEST_ID_HEADER, request_id.encode("latin-1")),
                ]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            status_code = 500
            error_log.exception("unhandled_exception", extra={"path": scope.get("path")})
            if not response_started:
                body = json.dumps(error_body("INTERNAL_ERROR", "Internal server error")).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 500,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                            (REQUEST_ID_HEADER, request_id.encode("latin-1")),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            route = scope.get("route")
            access_log.info(
                "request",
                extra={
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "route": getattr(route, "path", None),
                    "query": scope.get("query_string", b"").decode("latin-1") or None,
                    "status": status_code,
                    "latency_ms": latency_ms,
                    "client": (scope.get("client") or (None,))[0],
                },
            )
            request_id_var.reset(token)
