"""The FastAPI app factory. create_app() builds a fresh app each call
(what the test suite uses, one app per test module rather than one shared
mutable global), and the module level app below is what
`uvicorn groundwork_api.app:app` actually serves in the Fly.io container,
per the Dockerfile's own CMD.

CORS is the one piece of configuration that only matters once the static
web app (build order steps 21 to 25) is live on GitHub Pages, a different
origin than this API's own Fly.io domain by construction: a static export
cannot call a different origin without it. Settings.cors_origins, its own
docstring explains, defaults to the local web dev server and is set to
the real Pages origin as a documented, explicit deploy step, never
guessed at here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from groundwork_api.db import dispose_engine
from groundwork_api.logging import RequestLoggingMiddleware, configure_logging, get_logger
from groundwork_api.routers import chunks, conversations, health, workspaces
from groundwork_api.routers import eval as eval_router
from groundwork_core.config import get_settings

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    logger.info("app.startup")
    yield
    await dispose_engine()
    logger.info("app.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="groundwork",
        description=(
            "A multi tenant platform that turns an uploaded PDF into a chatbot that "
            "answers questions grounded only in that document, with verifiable "
            "citations and a measured refusal for anything the document does not "
            "cover. Retrieved document content is data, never instructions."
        ),
        version="0.1.0",
        lifespan=_lifespan,
    )

    origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-Admin-Token"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(health.router)
    app.include_router(workspaces.router)
    app.include_router(conversations.router)
    app.include_router(chunks.router)
    app.include_router(eval_router.router)

    return app


app = create_app()
