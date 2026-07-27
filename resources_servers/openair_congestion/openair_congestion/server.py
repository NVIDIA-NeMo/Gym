# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI server for ``openair_congestion_v1``.

Implements the NeMo Gym REST contract from ``docs/PLAN.md`` §4.2:

- ``GET  /health``  — implemented in M4.1.
- ``POST /reset``   — implemented in M4.2 (delegates to :class:`LiveEnv`).
- ``POST /step``    — implemented in M4.2.
- ``GET  /render``  — implemented in M4.2.
- ``POST /close``   — implemented in M4.2.

The ``app`` symbol is mounted by ``uvicorn``::

    uvicorn openair_congestion.server:app --host 127.0.0.1 --port 9100

Override the kpi-exporter URL via ``KPI_EXPORTER_URL`` (default
``http://localhost:9101/metrics``), the per-step pacing via
``ENV_STEP_DT_S`` (default ``1.0``), the post-reset settle via
``ENV_STEADY_STATE_S`` (default ``1.0``), and the env pool size via
``ENV_POOL_SIZE`` (default ``4``).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from . import ENV_NAME, SCHEMA_VERSION, __version__
from .env import LiveEnv
from .kpi_client import KpiScrapeError
from .schemas import (
    CloseRequest,
    CloseResponse,
    HealthResponse,
    RenderResponse,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)


LOG = logging.getLogger("openair_congestion.server")
_BUILD_REVISION: str | None = None


# --- Build/runtime metadata ------------------------------------------------


def _build_revision() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


_PRIVATE_CLOSE_KEYS = frozenset({"work_dir", "snapshot_path"})


def _public_close_summary(value: Any) -> Any:
    """Remove host-local paths before returning a close receipt over HTTP."""

    if isinstance(value, dict):
        return {key: _public_close_summary(item) for key, item in value.items() if key not in _PRIVATE_CLOSE_KEYS}
    if isinstance(value, list):
        return [_public_close_summary(item) for item in value]
    return value


# --- App factory -----------------------------------------------------------


def create_app(env: LiveEnv | None = None) -> FastAPI:
    global _BUILD_REVISION
    if _BUILD_REVISION is None:
        _BUILD_REVISION = _build_revision()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            closer = getattr(app.state.env, "close_all", None)
            if closer is not None:
                try:
                    closer()
                except Exception as exc:  # pragma: no cover
                    LOG.warning("env shutdown cleanup failed: %s", exc)

    app = FastAPI(
        title="openair_congestion env-server",
        version=__version__,
        lifespan=lifespan,
        description=(
            "NeMo Gym REST environment for openair-rl-gym. "
            "M4.2: /reset, /step, /render, /close are live and scrape the "
            "kpi-exporter for observations. Default live mode logs accepted "
            "actions. ENV_SCENARIO_MODE=t1_runner starts the congestion-gen "
            "T1 runner and applies set_admission_policy as traffic-side "
            "scenario control. FlexRIC RC / OAI telnet RAN control remains "
            "future work."
        ),
    )

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        return HealthResponse(
            ok=True,
            env_name=ENV_NAME,
            schema_version=SCHEMA_VERSION,
            n_episodes_live=app.state.env.n_episodes_live(),
            gpu_status={"required": False},
            build_revision=_BUILD_REVISION,
            scenario_mode=app.state.env.scenario_mode,
        )

    @app.post("/reset", response_model=ResetResponse, tags=["episode"])
    def reset(req: ResetRequest) -> ResetResponse:
        try:
            obs, meta = app.state.env.reset(
                seed=req.seed,
                difficulty=req.difficulty if req.difficulty is not None else 0.5,
                regime_mix=req.regime_mix,
                scenario_id=req.scenario_id,
                tier=req.tier,
                max_steps=req.max_steps,
            )
        except KpiScrapeError as e:
            LOG.warning("KPI scrape failed during reset: %s", e)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "kpi_scrape_error",
                    "message": "KPI source unavailable",
                },
            ) from e
        except RuntimeError as e:
            # Pool exhausted or sampler error.
            LOG.warning("environment unavailable during reset: %s", e)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "env_unavailable",
                    "message": "Environment unavailable",
                },
            ) from e
        except Exception as e:
            LOG.exception("reset failed")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "internal_error",
                    "message": "Internal server error",
                },
            ) from e
        return ResetResponse(episode_id=meta.episode_id, observation=obs, meta=meta)

    @app.post("/step", response_model=StepResponse, tags=["episode"])
    def step(req: StepRequest) -> StepResponse:
        try:
            obs, reward, done, info = app.state.env.step(req.episode_id, req.action)
        except KeyError as e:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "unknown_episode",
                    "episode_id": req.episode_id,
                },
            ) from e
        except KpiScrapeError as e:
            LOG.warning("KPI scrape failed during step: %s", e)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "kpi_scrape_error",
                    "message": "KPI source unavailable",
                },
            ) from e
        except RuntimeError as e:
            LOG.warning("invalid episode state during step: %s", e)
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "step_invalid_state",
                    "message": "Episode is not in a step-able state",
                },
            ) from e
        except Exception as e:
            LOG.exception("step failed")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "internal_error",
                    "message": "Internal server error",
                },
            ) from e
        return StepResponse(observation=obs, reward=reward, done=done, info=info)

    @app.get("/render", response_model=RenderResponse, tags=["episode"])
    def render(
        episode_id: str = Query(..., min_length=1, max_length=64),
        format: str = Query("ascii", pattern="^(ascii|json)$"),
    ) -> RenderResponse:
        try:
            payload = app.state.env.render(episode_id, format=format)
        except KeyError as e:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "unknown_episode",
                    "episode_id": episode_id,
                },
            ) from e
        return RenderResponse(episode_id=episode_id, format=format, payload=payload)

    @app.get("/render/text", include_in_schema=False)
    def render_text(
        episode_id: str = Query(..., min_length=1, max_length=64),
    ) -> PlainTextResponse:
        try:
            payload = app.state.env.render(episode_id, format="ascii")
        except KeyError as e:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "unknown_episode",
                    "episode_id": episode_id,
                },
            ) from e
        return PlainTextResponse(payload)

    @app.post("/close", response_model=CloseResponse, tags=["episode"])
    def close(req: CloseRequest) -> CloseResponse:
        try:
            summary = app.state.env.close(req.episode_id)
        except KeyError as e:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "unknown_episode",
                    "episode_id": req.episode_id,
                },
            ) from e
        public_summary = _public_close_summary(summary)
        return CloseResponse(ok=bool(public_summary.get("ok", True)), summary=public_summary)

    @app.get("/", include_in_schema=False)
    def root() -> JSONResponse:
        return JSONResponse(
            {
                "env_name": ENV_NAME,
                "schema_version": SCHEMA_VERSION,
                "version": __version__,
                "docs": "/docs",
                "openapi": "/openapi.json",
                "health": "/health",
            },
        )

    app.state.env = env if env is not None else LiveEnv()
    return app


app = create_app()


def main() -> int:
    parser = argparse.ArgumentParser(prog="openair-env-server")
    parser.add_argument("--host", default=os.environ.get("ENV_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ENV_SERVER_PORT", "9100")))
    parser.add_argument("--log-level", default=os.environ.get("ENV_SERVER_LOG_LEVEL", "info"))
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        "openair_congestion.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
