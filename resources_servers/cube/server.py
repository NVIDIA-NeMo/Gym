# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo-Gym resources server for CUBE ``Task`` episodes (YAML ``env_domain`` selects the adapter)."""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from openai.types.responses import FunctionToolParam
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymFunctionCallOutput
from resources_servers.cube.adapters import action_schemas_to_openai_tools, observation_to_input_messages
from resources_servers.cube.registry import instantiate_domain
from resources_servers.cube.schemas import (
    CubeAgentVerifyRequest,
    CubeAgentVerifyResponse,
    CubeCloseRequest,
    CubeCloseResponse,
    CubeResourcesServerConfig,
    CubeSeedSessionRequest,
    CubeSeedSessionResponse,
    CubeStepRequest,
    CubeStepResponse,
)


if TYPE_CHECKING:
    from resources_servers.cube.domains.base import CubeEnvironmentBase


logger = logging.getLogger(__name__)


class CubeResourcesServer(SimpleResourcesServer):
    """Hosts CUBE tasks; ``config.env_domain`` selects a :class:`~resources_servers.cube.domains.base.CubeEnvironmentBase` for load + warmup hooks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: CubeResourcesServerConfig
    env_id_to_task: Dict[str, Any] = Field(default_factory=dict)
    env_id_to_total_reward: Dict[str, float] = Field(default_factory=lambda: defaultdict(float))
    _task_configs_list: List[Any] = PrivateAttr(default_factory=list)
    _adapter_state: Any = PrivateAttr(default=None)
    _cube_domain_adapter_cache: Optional["CubeEnvironmentBase"] = PrivateAttr(default=None)

    def _cube_domain_adapter(self) -> CubeEnvironmentBase:
        if self._cube_domain_adapter_cache is None:
            self._cube_domain_adapter_cache = instantiate_domain(self.config.env_domain)
        return self._cube_domain_adapter_cache

    def ensure_task_configs(self) -> None:
        self._cube_domain_adapter().ensure_loaded(self)

    def empty_reset_obs_detail(self) -> str:
        return self._cube_domain_adapter().empty_reset_obs_detail()

    def close_all_open_environments(self) -> None:
        """Call ``task.close()`` for every active env (best-effort). Idempotent.

        Runs on ASGI lifespan shutdown (graceful uvicorn stop) and again at process exit via
        :func:`atexit.register` if anything is still open. Cannot run on SIGKILL / ``kill -9``.
        """
        items = list(self.env_id_to_task.items())
        self.env_id_to_task.clear()
        for env_id, task in items:
            try:
                task.close()
            except Exception:
                logger.exception(
                    "Cube resources server: task.close() failed during shutdown (env_id=%s)",
                    env_id,
                )
        self.env_id_to_total_reward.clear()

    async def _lifespan_shutdown_cleanup_async(self) -> None:
        """Run blocking ``task.close()`` (e.g. QEMU teardown) off the event loop."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.close_all_open_environments)

    def _wrap_lifespan_with_cube_shutdown(self, app: FastAPI) -> None:
        """Chain cube shutdown after the router's existing lifespan (startup hooks / profiling)."""
        prev = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan_with_shutdown(app_: FastAPI):
            async with prev(app_):
                yield
            await self._lifespan_shutdown_cleanup_async()

        app.router.lifespan_context = lifespan_with_shutdown  # type: ignore[assignment]

    def setup_webserver(self) -> FastAPI:
        if self.config.eager_benchmark_init:
            # Run before uvicorn starts so Ctrl+C during long init does not tear down Starlette lifespan
            # (avoids noisy KeyboardInterrupt + nested CancelledError from lifespan receive()).
            logger.info(
                "Cube resources server: eager startup (benchmark load + optional domain warmup; may take a long time)..."
            )
            try:
                self.ensure_task_configs()
                self._cube_domain_adapter().warm_on_startup(self)
            except Exception:
                logger.exception("Cube resources server: benchmark init failed during startup")
                raise
            logger.info("Cube resources server: benchmark ready (%d tasks).", len(self._task_configs_list))
            app = FastAPI()
            self.setup_session_middleware(app)
            app.post("/seed_session")(self.seed_session)
            app.post("/verify")(self.verify)
            app.post("/aggregate_metrics")(self.aggregate_metrics)
        else:
            app = super().setup_webserver()
        app.post("/step")(self.step)
        app.post("/close")(self.close)
        self._wrap_lifespan_with_cube_shutdown(app)
        atexit.register(self.close_all_open_environments)
        return app

    async def seed_session(self, request: Request, body: CubeSeedSessionRequest) -> CubeSeedSessionResponse:
        self.ensure_task_configs()
        if body.task_idx < 0 or body.task_idx >= len(self._task_configs_list):
            raise HTTPException(
                status_code=400,
                detail=f"task_idx {body.task_idx} out of range (0..{len(self._task_configs_list) - 1})",
            )

        env_id = str(uuid.uuid4())
        task_config = self._task_configs_list[body.task_idx]
        task = task_config.make()
        self.env_id_to_task[env_id] = task

        obs, _info = task.reset()
        raw_tools = action_schemas_to_openai_tools(task.action_set)
        tools = [FunctionToolParam(**t) for t in raw_tools]
        nemo_obs = observation_to_input_messages(obs)

        if not nemo_obs:
            raise HTTPException(status_code=500, detail=self.empty_reset_obs_detail())

        return CubeSeedSessionResponse(env_id=env_id, obs=nemo_obs, tools=tools)

    async def step(self, request: Request, body: CubeStepRequest) -> CubeStepResponse:
        from cube.core import Action, Observation

        try:
            task = self.env_id_to_task[body.env_id]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Unknown env_id {body.env_id}") from None

        try:
            actions: List[Action] = []
            for a in body.action:
                try:
                    args = json.loads(a.arguments) if isinstance(a.arguments, str) else dict(a.arguments)
                except json.JSONDecodeError as e:
                    logger.warning("Invalid tool arguments JSON: %s", e)
                    return CubeStepResponse(
                        obs=[
                            NeMoGymEasyInputMessage(
                                role="user",
                                content="Invalid tool call arguments (not valid JSON).",
                                type="message",
                            )
                        ],
                        reward=0.0,
                        done=False,
                    )
                actions.append(Action(id=a.call_id, name=a.name, arguments=args))

            env_out = task.step(actions)
        except Exception:
            logger.exception("CUBE task step failed")
            raise

        self.env_id_to_total_reward[body.env_id] += float(env_out.reward)

        if env_out.error is not None:
            err = env_out.error
            return CubeStepResponse(
                obs=[
                    NeMoGymEasyInputMessage(
                        role="user",
                        content=f"Environment error ({err.error_type}): {err.exception_str}",
                        type="message",
                    )
                ],
                reward=float(env_out.reward),
                done=True,
            )

        nemo_obs: List[NeMoGymEasyInputMessage | NeMoGymFunctionCallOutput] = []
        for content in env_out.obs.contents:
            if content.tool_call_id:
                nemo_obs.append(
                    NeMoGymFunctionCallOutput(
                        call_id=content.tool_call_id,
                        output=content.to_markdown(),
                        type="function_call_output",
                    )
                )
            else:
                sub = Observation(contents=[content])
                nemo_obs.extend(observation_to_input_messages(sub))

        return CubeStepResponse(
            obs=nemo_obs,
            reward=float(env_out.reward),
            done=bool(env_out.done),
        )

    async def verify(self, request: Request, body: CubeAgentVerifyRequest) -> CubeAgentVerifyResponse:
        reward = float(self.env_id_to_total_reward[body.response.env_id])
        return CubeAgentVerifyResponse(**body.model_dump(), reward=reward)

    async def close(self, request: Request, body: CubeCloseRequest) -> CubeCloseResponse:
        task = self.env_id_to_task.pop(body.env_id, None)
        if task is None:
            return CubeCloseResponse(message=f"Unknown env_id {body.env_id}", success=False)
        try:
            task.close()
        except Exception as e:
            return CubeCloseResponse(message=repr(e), success=False)
        return CubeCloseResponse(message="Success", success=True)
