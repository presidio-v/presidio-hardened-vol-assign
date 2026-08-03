"""FastAPI application for the ``pva serve`` demo GUI.

Built on ``presidio-hardened-fastapi``, so locked-down CORS, the security header
set (including ``Content-Security-Policy: default-src 'self'``) and per-IP rate
limiting are applied on construction rather than bolted on here.

The CSP is why the bundled page loads its CSS and JS as separate same-origin
files and carries no inline ``<script>`` or ``style=`` attributes.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from presidio_fastapi import FastAPI
from presidio_fastapi.rate_limit import limiter

from presidio_vol_assign.web.runner import (
    LIMITS,
    RunRejected,
    RunTimeout,
    SolverPool,
    build_request,
    evidence_available,
)
from presidio_vol_assign.web.schemas import RunPayload

log = logging.getLogger("presidio_vol_assign.web")

STATIC_DIR = Path(__file__).parent / "static"

RUN_RATE_LIMIT = "12/minute"
"""Stricter than the family default: each run costs seconds of CPU."""


def _package_version() -> str:
    try:
        return version("presidio-hardened-vol-assign")
    except PackageNotFoundError:  # pragma: no cover - source checkout without install
        return "unknown"


def create_app(*, cors_allow_origins: tuple[str, ...] = ()) -> FastAPI:
    """Build the demo application.

    Args:
        cors_allow_origins: Extra origins allowed to call the API. Empty by
            default — the bundled page is same-origin, so nothing is needed for
            normal use.
    """
    pool = SolverPool()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Warm the worker pool at startup so the first visitor does not pay the
        # scikit-fuzzy / DEAP import cost.
        pool.start()
        log.info("solver pool warmed (%d workers)", LIMITS.max_workers)
        try:
            yield
        finally:
            pool.shutdown()

    app = FastAPI(
        title="presidio-hardened-vol-assign — demo",
        description=(
            "Interactive demo of the multi-objective post-disaster assignment models. "
            "Every run is synthetic and reproducible from (scenario, knobs, seed); "
            "no uploaded data is stored."
        ),
        version=_package_version(),
        cors_allow_origins=cors_allow_origins,
        cors_allow_methods=("GET", "POST"),
        lifespan=lifespan,
    )
    app.state.pool = pool

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Liveness probe."""
        return {"status": "ok", "version": _package_version()}

    @app.get("/api/scenarios")
    def scenarios() -> dict[str, Any]:
        """The presets, the caps they run under, and the solver options."""
        from presidio_vol_assign.web.runner import ALLOWED_SOLVERS
        from presidio_vol_assign.web.scenarios import SCENARIOS

        return {
            "version": _package_version(),
            "scenarios": [s.as_dict() for s in SCENARIOS],
            "solvers": [
                {"id": "nsga2", "label": "NSGA-II"},
                {"id": "nrga", "label": "NRGA"},
                {"id": "both", "label": "Both (compare)"},
            ],
            "allowedSolvers": list(ALLOWED_SOLVERS),
            "limits": {
                "maxUnits": LIMITS.max_units,
                "maxSites": LIMITS.max_sites,
                "maxGenerations": LIMITS.max_generations,
                "maxPopSize": LIMITS.max_pop_size,
                "timeoutSec": LIMITS.timeout_sec,
            },
            "evidenceAvailable": evidence_available(),
        }

    @app.post("/api/run")
    @limiter.limit(RUN_RATE_LIMIT)
    def run(request: Request, payload: RunPayload) -> JSONResponse:
        """Generate a synthetic instance for a scenario and solve it.

        ``request`` is unused in the body but required by the rate limiter,
        which keys on the client address.
        """
        try:
            run_request = build_request(payload.to_request_dict())
        except RunRejected as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
            result = request.app.state.pool.run(run_request)
        except RunTimeout as exc:
            log.warning("run timed out: %s", run_request.as_dict())
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except RunRejected as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - never leak internals to the browser
            log.exception("solver run failed")
            raise HTTPException(status_code=500, detail="solver run failed") from exc

        return JSONResponse(result)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/")
        def index() -> FileResponse:
            """Serve the demo page."""
            return FileResponse(STATIC_DIR / "index.html")

    return app


def app_from_env() -> FastAPI:
    """Factory for ``uvicorn --reload``, which needs an import string.

    Reads extra CORS origins from ``PVA_WEB_CORS_ORIGINS`` (comma-separated),
    since the reloader cannot be handed constructor arguments.
    """
    raw = os.environ.get("PVA_WEB_CORS_ORIGINS", "")
    origins = tuple(origin.strip() for origin in raw.split(",") if origin.strip())
    return create_app(cors_allow_origins=origins)
