"""groundwork_api.app tests: the lifespan hook itself. Every other test
in this package overrides groundwork_api.deps.get_session and never lets
create_app()'s app run through a real startup and shutdown (deliberately,
per conftest.client's own docstring, since dispose_engine there would
tear down a process wide engine no override based test ever creates), so
this file is the only place _lifespan's own body, configure_logging on
the way up and dispose_engine on the way down, actually executes.
"""

from __future__ import annotations

from groundwork_api.app import create_app


async def test_lifespan_runs_startup_and_shutdown_without_error() -> None:
    app = create_app()

    async with app.router.lifespan_context(app):
        pass
