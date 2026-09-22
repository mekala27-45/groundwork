"""structlog setup and the one request logging middleware every request
passes through: JSON output, no document content at any level, per the
stack table's own hard rule.

The middleware logs request metadata only, method, path, status code, and
duration, and never reads or logs a request or response body. That is a
stronger guarantee than redacting specific fields after the fact: a body
containing a chunk of an uploaded PDF or a full generated answer never
reaches a logger call in the first place, rather than relying on every
call site to remember to redact it. Anywhere a router does want to log
something content bearing, such as noting a document was ingested, it
goes through groundwork_core.redaction.content_hash explicitly, the same
choke point the rest of this codebase already uses.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import cast

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from groundwork_core.config import get_settings

Logger = structlog.stdlib.BoundLogger


def configure_logging() -> None:
    """Idempotent: safe to call more than once, since the test suite
    builds the app factory repeatedly within one process."""
    settings = get_settings()
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Logger:
    # structlog.get_logger is typed to return Any (its factory is only
    # known at structlog.configure() time); cast to the concrete
    # stdlib.BoundLogger every call site in this codebase actually gets,
    # since configure_logging always wires stdlib.LoggerFactory.
    return cast(Logger, structlog.get_logger(name))


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: object) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._logger = get_logger("groundwork_api.request")

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.monotonic()
        response = await call_next(request)
        self._logger.info(
            "request.completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round((time.monotonic() - started) * 1000, 2),
        )
        return response
