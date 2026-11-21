# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import math
import os
import sys
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Annotated, Any, ClassVar

import anyio
import anyio.to_thread
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BeforeValidator, ConfigDict, Field

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import EpisodeId
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    SimpleResourcesServer,
    lost_session_verification,
    reject_partial_typed_seed,
)
from nemo_gym.verification import VerifierFixture
from resources_servers.visgym._metadata import sanitize_metadata
from resources_servers.visgym._observation import (
    attach_env_info,
    coerce_images,
    observation_to_user_message,
)
from resources_servers.visgym.schemas import (
    VisGymAgentVerifyRequest,
    VisGymAgentVerifyResponse,
    VisGymCloseRequest,
    VisGymCloseResponse,
    VisGymResourcesServerConfig,
    VisGymSeedSessionRequest,
    VisGymSeedSessionResponse,
    VisGymStepRequest,
    VisGymStepResponse,
    VisGymTaskRow,
)
from resources_servers.visgym.task_data import VISGYM_BINARY_SUCCESS_ENV_IDS
from resources_servers.visgym.verifier_fixture import create_server, invoke


logger = logging.getLogger(__name__)


async def _run_in_threadpool_to_completion(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Do not release a native-env lock while its worker thread is still running."""
    worker_task = asyncio.create_task(run_in_threadpool(func, *args, **kwargs))
    try:
        return await asyncio.shield(worker_task)
    except asyncio.CancelledError as cancellation:
        # AnyIO cancellation is level-triggered and would immediately cancel a
        # plain second await; repeated raw Task.cancel() calls can do the same.
        # Keep ownership of the native operation until its worker really exits.
        with anyio.CancelScope(shield=True):
            while not worker_task.done():
                try:
                    await asyncio.shield(worker_task)
                except asyncio.CancelledError:
                    continue
        try:
            worker_task.result()
        except BaseException:
            pass
        raise cancellation


@dataclass
class _SessionOperation:
    """Per-session coordination state, published before the first await."""

    lock: asyncio.Lock = dataclass_field(default_factory=asyncio.Lock)
    seed_finished: asyncio.Event = dataclass_field(default_factory=asyncio.Event)
    close_requested: bool = False
    discard_reward_requested: bool = False


def _ensure_headless_defaults() -> None:
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _ensure_matplotlib_canvas_compatibility() -> None:
    try:
        from matplotlib.backend_bases import FigureCanvasBase
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        return

    if hasattr(FigureCanvasBase, "print_to_buffer"):
        return

    def print_to_buffer(canvas: Any) -> tuple[bytes, tuple[int, int]]:
        return FigureCanvasAgg(canvas.figure).print_to_buffer()

    FigureCanvasBase.print_to_buffer = print_to_buffer  # type: ignore[attr-defined]


# Serializes every mental_rotation_3d_cube figure operation in this process.
# The environment renders through the pyplot state machine (plt.figure, then
# plt.close) and its close() calls plt.close("all") / plt.clf() / plt.cla(),
# all of which mutate matplotlib's *global* figure registry. This server runs
# sessions concurrently in a threadpool, so one session tearing down its
# environment while another is mid-render drops the second one's axes out of
# the registry and raises `KeyError: <Axes3D: >`. It is intermittent and
# invisible to a single-threaded probe: a blended 16-node run hit it twice.
_MENTAL_ROTATION_3D_FIGURE_LOCK = threading.RLock()


def _install_mental_rotation_3d_renderer_compatibility() -> None:
    """Keep rendered frames consistent with the observation space, and thread-safe."""
    module = importlib.import_module("gymnasium.envs.mental_rotation_3d_cube.mental_rotation_3d_cube")
    if getattr(module, "_nemo_gym_renderer_size_installed", False):
        return

    import numpy as np
    from PIL import Image

    original_render = module.MentalRotation3DCubeEnv._render
    original_close = module.MentalRotation3DCubeEnv.close

    def render(env: Any, rotation: Any) -> Any:
        with _MENTAL_ROTATION_3D_FIGURE_LOCK:
            image = original_render(env, rotation)
        height, width = (int(value) for value in env.image_size)
        if image.shape[:2] != (height, width):
            image = np.asarray(
                Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS),
                dtype=np.uint8,
            )
        return image

    def close(env: Any) -> Any:
        with _MENTAL_ROTATION_3D_FIGURE_LOCK:
            return original_close(env)

    module.MentalRotation3DCubeEnv._render = render
    module.MentalRotation3DCubeEnv.close = close
    module._nemo_gym_renderer_size_installed = True


# Configure rendering before importing the VisGym Gymnasium fork; importing
# first caches the wrong backend and leaves bare Figure canvases unusable.
_ensure_headless_defaults()
_ensure_matplotlib_canvas_compatibility()
_visgym_repo_root = os.getenv("VISGYM_REPO_ROOT")
if _visgym_repo_root and _visgym_repo_root not in sys.path:
    sys.path.insert(0, _visgym_repo_root)

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - exercised in containers lacking VisGym deps.
    gym = None  # type: ignore[assignment]


def _ensure_visgym_importable(env_id: str | None = None) -> Any:
    repo_root = os.getenv("VISGYM_REPO_ROOT")
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    global gym
    if gym is None:
        gym = importlib.import_module("gymnasium")
    importlib.import_module("gymnasium.envs")
    if env_id and env_id.startswith("mental_rotation_3d_cube/"):
        _install_mental_rotation_3d_renderer_compatibility()
    return gym


_PATH_LIKE_ENV_KWARGS = (
    "sample_dir",
    "asset_dir",
    "data_dir",
    "dataset_dir",
    "sample_path",
    "asset_root",
    # counting/easy carries this alongside sample_dir; resolving one but not
    # the other is worse than resolving neither, because the images load and
    # only the annotations go missing.
    "annotation_file",
)


def _resolve_asset_kwargs(env_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve relative asset paths in env_kwargs against Gym's search roots.

    Asset-backed VisGym tasks (jigsaw, colorization, mental_rotation_2d,
    zoom_in_puzzle, ...) point at rendered image directories. Manifests carry
    those as repo-relative paths so the same row works from a checkout, a code
    snapshot or a container mount; only an absolute path is taken literally.
    Without this the rows have to hard-code one deployment's layout, and the
    environment fails at reset everywhere else.
    """
    from nemo_gym import _resolve_under_cwd_or_install

    resolved = dict(env_kwargs)
    for key in _PATH_LIKE_ENV_KWARGS:
        value = resolved.get(key)
        if not isinstance(value, str) or not value:
            continue
        candidate = Path(value)
        if candidate.is_absolute():
            continue
        resolved[key] = str(_resolve_under_cwd_or_install(candidate))
    return resolved


class VisGymResourcesServer(SimpleResourcesServer):
    """Env-id-parametric server wrapping VisGym/Gymnasium envs.

    ``resources_servers.gymnasium.GymnasiumServer`` intentionally exposes a
    string-observation/model-response contract. VisGym's observations are
    multimodal messages and its actions are already-parsed strings, followed
    by an explicit reward-drain request. Keeping this server specialized avoids
    widening the generic server's public schemas for unrelated environments.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: VisGymResourcesServerConfig
    seed_request_model: ClassVar[type[VisGymSeedSessionRequest]] = VisGymSeedSessionRequest
    env_id_to_failure: dict[str, str] = Field(default_factory=dict)

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self.env_id_to_operation or session_id in self.env_id_to_total_reward

    async def initialize_session_state(self, session_id: str, body: VisGymSeedSessionRequest) -> None:
        await self._seed_native_session(None, body.model_copy(update={"env_id": session_id}))

    async def release_session_state(self, session_id: str) -> None:
        result = await self._close_native_session(None, VisGymCloseRequest(env_id=session_id, discard_reward=True))
        if not result.success:
            raise RuntimeError(result.message)
        self.env_id_to_failure.pop(session_id, None)

    task_rows: list[dict[str, Any]] = Field(default_factory=list)
    env_id_to_env: dict[str, Any] = Field(default_factory=dict)
    env_id_to_total_reward: dict[str, float] = Field(default_factory=lambda: defaultdict(float))
    env_id_to_task_row: dict[str, dict[str, Any]] = Field(default_factory=dict)
    env_id_to_turn_count: dict[str, int] = Field(default_factory=lambda: defaultdict(int))
    env_id_to_task_success: dict[str, bool | None] = Field(default_factory=dict)
    env_id_to_operation: dict[str, _SessionOperation] = Field(default_factory=dict, exclude=True)
    cleanup_tombstone_expiry: dict[str, float] = Field(default_factory=dict, exclude=True)
    background_close_tasks: set[asyncio.Task[VisGymCloseResponse]] = Field(default_factory=set, exclude=True)
    env_construction_lock: Any = Field(default_factory=asyncio.Lock)
    serial_env_op_lock: Any = Field(default_factory=asyncio.Lock)

    def model_post_init(self, _ctx: Any) -> None:
        _ensure_headless_defaults()
        for jsonl_path in self.config.task_jsonl_fpaths:
            with Path(jsonl_path).open() as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    try:
                        validated = VisGymTaskRow.model_validate(row)
                    except Exception:
                        logger.exception("Invalid VisGym task row in %s:%s", jsonl_path, line_no)
                        raise
                    self.task_rows.append(validated.model_dump(mode="json"))

    def multiple_workers_unsupported_reason(self) -> str:
        return "Native environments are keyed by the adapter's env_id and require a single owning worker."

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        @app.on_event("startup")
        async def _raise_env_op_threadpool_capacity() -> None:
            # run_in_threadpool (used by every env.reset/step/render/close call
            # below) delegates to anyio's default worker-thread limiter, which
            # defaults to 40 tokens process-wide. This single-process,
            # stateful server (env_id_to_env can't be sharded across uvicorn
            # workers) hits that ceiling long before generation capacity does,
            # so raising it here -- from inside the running event loop, where
            # the limiter actually lives -- is the direct fix. Must happen at
            # startup, before any request claims a token.
            limiter = anyio.to_thread.current_default_thread_limiter()
            limiter.total_tokens = self.config.env_op_threadpool_size
            if self.config.cap_render_lib_threads:
                try:
                    import cv2

                    cv2.setNumThreads(1)
                except ImportError:
                    pass

        app.post("/step")(self.step)
        app.post("/close")(self.close)
        return app

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest
        | Annotated[VisGymSeedSessionRequest, BeforeValidator(reject_partial_typed_seed)],
    ) -> ResourcesSeedSessionResponse | VisGymSeedSessionResponse:
        if isinstance(body, ResourcesSeedSessionRequest):
            return await super().seed_session(body, request=request)
        if body.env_id is not None and (self.session_is_closed(body.env_id) or self.session_is_verified(body.env_id)):
            raise HTTPException(status_code=409, detail="Cannot recreate a closed or verified session.")
        return await self._seed_native_session(request, body)

    async def _seed_native_session(
        self, request: Request | None, body: VisGymSeedSessionRequest
    ) -> VisGymSeedSessionResponse:
        row = self._resolve_task_row(body)
        # These experimental protocols must never silently fall back to a
        # different observation or reward contract in the reusable adapter.
        if "reward_shaping" in row.task_metadata:
            raise HTTPException(
                status_code=400, detail="Experimental reward_shaping is not supported; use native rewards."
            )
        if row.env_id.startswith(("maze_3d/", "fetch_")):
            raise HTTPException(
                status_code=400,
                detail="Maze3D and Fetch require separately qualified rendering support and are not supported by this adapter.",
            )
        has_binary_success_contract = row.env_id in VISGYM_BINARY_SUCCESS_ENV_IDS
        if self.config.require_audited_binary_success and not has_binary_success_contract:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"env_id={row.env_id!r} is outside VisGym's audited binary-success families; "
                    "use an audited VisGym easy/hard ID or disable strict success validation for "
                    "reward-only diagnostics."
                ),
            )
        env_kwargs = _resolve_asset_kwargs(row.env_kwargs)
        if row.seed_key:
            env_kwargs[row.seed_key] = row.seed

        env_id = body.env_id or str(uuid.uuid4())
        operation = self._reserve_session(env_id)
        env: Any | None = None
        constructed_env: list[Any | None] = [None]
        try:
            async with operation.lock:
                # Compatibility patches mutate process-global provider classes.
                async with self.env_construction_lock:
                    if row.env_id.startswith("mental_rotation_3d_cube/"):
                        _install_mental_rotation_3d_renderer_compatibility()
                    gym_module = _ensure_visgym_importable()

                    def construct_env() -> Any:
                        made_env = gym_module.make(row.env_id, **env_kwargs)
                        constructed_env[0] = made_env
                        return made_env

                    env = await _run_in_threadpool_to_completion(construct_env)
                obs, info = await self._reset_env(env, row)
                obs_msg = observation_to_user_message(
                    image_value=await self._image_value(env, obs, row.env_id),
                    env_id=row.env_id,
                    prefix_text=self._prompt_for_env(env, row),
                    feedback_text=self._feedback_text(info),
                    image_format=self.config.image_format,
                    image_jpeg_quality=self.config.image_jpeg_quality,
                    skip_images=self.config.skip_images,
                )
                obs_msg = attach_env_info(obs_msg, self._env_info(info, row.env_id))

                # /close can arrive while an ambiguously completed seed request
                # is still constructing/resetting. It marks the reservation
                # before waiting for this lock, so never publish a late orphan.
                if operation.close_requested:
                    raise HTTPException(status_code=409, detail=f"env_id={env_id} was closed while being seeded.")

                self.env_id_to_env[env_id] = env
                self.env_id_to_task_row[env_id] = row.model_dump(mode="json")
                self.env_id_to_total_reward[env_id] = 0.0
                self.env_id_to_turn_count[env_id] = 0
                self.env_id_to_task_success[env_id] = False if has_binary_success_contract else None
                operation.seed_finished.set()

                return VisGymSeedSessionResponse(env_id=env_id, obs=[obs_msg])
        except BaseException as exc:
            # The async-with has released the operation lock. Mark close intent
            # before the first await so a client that knows its requested ID
            # cannot race /step against partial-seed teardown.
            operation.close_requested = True
            if env is None:
                env = constructed_env[0]
            self._discard_session_state(env_id)
            close_failed = False
            try:
                if env is not None:
                    try:
                        await _run_in_threadpool_to_completion(env.close)
                    except BaseException:
                        # Retain the failed-close object and reservation so an
                        # abort /close can retry instead of losing an orphaned
                        # native environment when seed itself already failed.
                        close_failed = True
                        self.env_id_to_env[env_id] = env
                        logger.warning("Failed to close partially seeded VisGym env %s", row.env_id, exc_info=True)
            finally:
                operation.seed_finished.set()
                if not close_failed:
                    self._forget_operation(env_id, operation)
            if isinstance(exc, HTTPException):
                raise
            if not isinstance(exc, Exception):
                raise
            logger.exception("Failed to seed VisGym env %s with kwargs=%s", row.env_id, env_kwargs)
            raise HTTPException(
                status_code=500,
                detail=f"env construction failed: {type(exc).__name__}: {exc}",
            ) from exc

    def _horizon_reached(self, env_id: str, row_dict: dict[str, Any]) -> bool:
        """True when this session has used up its per-task horizon_cap."""
        return bool(
            self.config.enforce_horizon_cap
            and row_dict.get("horizon_cap") is not None
            and self.env_id_to_turn_count[env_id] >= row_dict["horizon_cap"]
        )

    def _check_typed_session_cookie(self, request: Request, env_id: str) -> None:
        # The legacy endpoints have no episode field. Do not let a caller bypass
        # typed ownership by supplying someone else's ID to /step or /close.
        if self.session_identity(env_id) is not None and request.session.get(SESSION_ID_KEY) != env_id:
            raise HTTPException(status_code=409, detail="Typed session requires its owning session cookie.")

    async def step(self, request: Request, body: VisGymStepRequest) -> VisGymStepResponse:
        self._check_typed_session_cookie(request, body.env_id)
        operation = self.env_id_to_operation.get(body.env_id)
        if operation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown env_id={body.env_id}; was the session closed?",
            )
        async with operation.lock:
            if self.env_id_to_operation.get(body.env_id) is not operation or operation.close_requested:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown env_id={body.env_id}; was the session closed?",
                )
            if body.env_id not in self.env_id_to_failure:
                try:
                    return await self._step_session(body)
                except Exception as exc:
                    logger.exception("Simulator step failed for %s", body.env_id)
                    self.env_id_to_failure[body.env_id] = f"{type(exc).__name__}: {exc}"
            return VisGymStepResponse(obs=[], reward=0.0, done=True, horizon_terminated=False)

    async def _step_session(self, body: VisGymStepRequest) -> VisGymStepResponse:
        if body.env_id not in self.env_id_to_env:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown env_id={body.env_id}; was the session closed?",
            )

        env = self.env_id_to_env[body.env_id]
        row_dict = self.env_id_to_task_row[body.env_id]
        env_id_str = row_dict["env_id"]

        obs, reward, terminated, truncated, info = await self._step_env(env, env_id_str, body.action_string)

        self.env_id_to_turn_count[body.env_id] += 1
        raw_env_reward = float(reward)
        if not math.isfinite(raw_env_reward):
            raise ValueError("Simulator returned a nonfinite reward.")
        has_binary_success_contract = self.env_id_to_task_success.get(body.env_id) is not None
        if has_binary_success_contract and bool(terminated):
            is_failure = math.isclose(raw_env_reward, 0.0, rel_tol=0.0, abs_tol=1e-9)
            is_success = math.isclose(raw_env_reward, 1.0, rel_tol=0.0, abs_tol=1e-9)
            if not (is_failure or is_success):
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Audited VisGym env_id={env_id_str!r} terminated with non-binary raw "
                        f"reward {raw_env_reward}; its success contract has drifted."
                    ),
                )
            if is_success:
                self.env_id_to_task_success[body.env_id] = True
        training_step_reward = raw_env_reward
        total_reward = self.env_id_to_total_reward[body.env_id] + training_step_reward
        if not math.isfinite(total_reward):
            raise ValueError("Simulator returned a nonfinite accumulated reward.")
        self.env_id_to_total_reward[body.env_id] = total_reward
        done = bool(terminated or truncated)

        horizon_terminated = self._horizon_reached(body.env_id, row_dict)
        if horizon_terminated:
            done = True

        env_info = self._env_info(info, env_id_str)
        env_info.update(
            {
                "raw_env_reward": raw_env_reward,
                "training_step_reward": training_step_reward,
                "training_reward": self.env_id_to_total_reward[body.env_id],
                "turn": self.env_id_to_turn_count[body.env_id],
            }
        )

        obs_msg = observation_to_user_message(
            image_value=await self._image_value(env, obs, env_id_str),
            env_id=env_id_str,
            feedback_text=self._feedback_text(info),
            image_format=self.config.image_format,
            image_jpeg_quality=self.config.image_jpeg_quality,
            skip_images=self.config.skip_images,
        )
        obs_msg = attach_env_info(obs_msg, env_info)

        return VisGymStepResponse(
            obs=[obs_msg],
            reward=training_step_reward,
            done=done,
            horizon_terminated=horizon_terminated,
        )

    async def close(self, request: Request, body: VisGymCloseRequest) -> VisGymCloseResponse:
        self._check_typed_session_cookie(request, body.env_id)
        if body.discard_reward:
            identity = self.session_identity(body.env_id)
            episode_id = (
                identity[0]
                if identity
                else EpisodeId(rollout_id="native-" + hashlib.sha256(body.env_id.encode()).hexdigest())
            )
            await self.close_resources_session(
                ResourcesCloseSessionRequest(resources_session_id=body.env_id, episode_id=episode_id)
            )
            return VisGymCloseResponse(success=True, message="closed")
        return await self._close_native_session(request, body)

    async def _close_native_session(self, request: Request | None, body: VisGymCloseRequest) -> VisGymCloseResponse:
        # An abort can reach this process before the seed request it is cleaning
        # up. Tombstone the client-generated ID before consulting live state so
        # the delayed mutation cannot publish an orphan.
        if body.discard_reward:
            self._record_cleanup_tombstone(body.env_id)
        operation = self.env_id_to_operation.get(body.env_id)
        if operation is None:
            return VisGymCloseResponse(success=True, message="already closed")

        operation.close_requested = True
        operation.discard_reward_requested |= body.discard_reward
        close_task = asyncio.create_task(self._complete_close(body.env_id, operation))
        try:
            return await asyncio.shield(close_task)
        except asyncio.CancelledError:
            self._track_background_close(close_task)
            raise

    async def _complete_close(self, env_id: str, operation: _SessionOperation) -> VisGymCloseResponse:
        async with operation.lock:
            await operation.seed_finished.wait()
            if self.env_id_to_operation.get(env_id) is not operation:
                return VisGymCloseResponse(success=True, message="already closed")
            return await self._close_session(env_id, operation)

    async def _close_session(self, env_id: str, operation: _SessionOperation) -> VisGymCloseResponse:
        env = self.env_id_to_env.get(env_id)
        if env is None:
            if operation.discard_reward_requested:
                self.env_id_to_total_reward.pop(env_id, None)
                self.env_id_to_task_success.pop(env_id, None)
                self._forget_operation(env_id, operation)
            return VisGymCloseResponse(success=True, message="already closed")

        try:
            await _run_in_threadpool_to_completion(env.close)
        except Exception as exc:
            logger.warning("VisGym env.close raised on env_id=%s: %s", env_id, exc)
            return VisGymCloseResponse(success=False, message=repr(exc))

        self.env_id_to_env.pop(env_id, None)
        self.env_id_to_task_row.pop(env_id, None)
        self.env_id_to_turn_count.pop(env_id, None)
        if operation.discard_reward_requested:
            self.env_id_to_total_reward.pop(env_id, None)
            self.env_id_to_task_success.pop(env_id, None)
            self._forget_operation(env_id, operation)

        return VisGymCloseResponse(success=True, message="ok")

    async def verify(self, request: Request, body: VisGymAgentVerifyRequest) -> VisGymAgentVerifyResponse:
        env_id = body.response.env_id
        self._check_typed_session_cookie(request, env_id)
        if env_id in self.env_id_to_env:
            raise HTTPException(status_code=409, detail="Close the native environment before verification.")
        fingerprint = hashlib.sha256(json.dumps(body.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()
        async with self.verifying(env_id) as verification:
            if verification.is_lost:
                return VisGymAgentVerifyResponse(
                    responses_create_params=body.responses_create_params,
                    response=body.response,
                    task_success=None,
                    **lost_session_verification("Episode state and retry evidence are unavailable."),
                )
            if verification.retained is not None:
                receipt = verification.retained
                verification.retain(receipt)
                if receipt["request_sha256"] != fingerprint:
                    return VisGymAgentVerifyResponse(
                        responses_create_params=body.responses_create_params,
                        response=body.response,
                        reward=0.0,
                        task_success=None,
                        mask_sample=True,
                        failure_kind="visgym:verification_mismatch",
                        failure_reason="Verification retry does not match the originally graded request.",
                    )
                response = body.response.model_copy(deep=True)
                if receipt["training_reward"] is not None:
                    response.metadata = dict(response.metadata or {}) | {"training_reward": receipt["training_reward"]}
                return VisGymAgentVerifyResponse(
                    responses_create_params=body.responses_create_params, response=response, **receipt["grade"]
                )
            operation = self.env_id_to_operation.get(env_id)
            if operation is not None:
                async with operation.lock:
                    result = self._verify_session(body)
            else:
                result = self._verify_session(body)
            verification.retain(
                {
                    "request_sha256": fingerprint,
                    "training_reward": (result.response.metadata or {}).get("training_reward"),
                    "grade": result.model_dump(
                        include={"reward", "task_success", "mask_sample", "failure_kind", "failure_reason"}
                    ),
                }
            )
            return result

    def _verify_session(self, body: VisGymAgentVerifyRequest) -> VisGymAgentVerifyResponse:
        env_id = body.response.env_id
        if env_id not in self.env_id_to_total_reward:
            return VisGymAgentVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=body.response,
                task_success=None,
                **lost_session_verification(
                    "Episode reward is unavailable; the session was lost or already verified."
                ),
            )
        if env_id in self.env_id_to_failure:
            return VisGymAgentVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=body.response,
                reward=0.0,
                task_success=None,
                mask_sample=True,
                failure_kind="visgym:simulator_failed",
                failure_reason=self.env_id_to_failure[env_id],
            )
        reward = self.env_id_to_total_reward.pop(env_id, 0.0)
        training_reward = reward
        task_success = self.env_id_to_task_success.pop(env_id, None)
        if self.config.reward_mode == "task_success":
            if task_success is None:
                return VisGymAgentVerifyResponse(
                    responses_create_params=body.responses_create_params,
                    response=body.response,
                    reward=0.0,
                    task_success=None,
                    mask_sample=True,
                    failure_kind="visgym:success_unavailable",
                    failure_reason="Binary evaluation requires an audited task-success predicate.",
                )
            reward = float(task_success)
        metadata = dict(body.response.metadata or {})
        metadata["training_reward"] = str(training_reward)
        response = body.response.model_copy(update={"metadata": metadata})
        return VisGymAgentVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=response,
            reward=reward,
            task_success=task_success,
        )

    def _session_id_in_use(self, env_id: str) -> bool:
        return any(
            env_id in state
            for state in (
                self.env_id_to_env,
                self.env_id_to_total_reward,
                self.env_id_to_task_row,
                self.env_id_to_turn_count,
                self.env_id_to_task_success,
                self.env_id_to_operation,
            )
        )

    def _reserve_session(self, env_id: str) -> _SessionOperation:
        if self._has_active_cleanup_tombstone(env_id):
            raise HTTPException(status_code=409, detail=f"env_id={env_id} was already closed.")
        if self._session_id_in_use(env_id):
            raise HTTPException(status_code=409, detail=f"env_id={env_id} is already in use.")
        operation = _SessionOperation()
        self.env_id_to_operation[env_id] = operation
        return operation

    def _record_cleanup_tombstone(self, env_id: str) -> None:
        now = monotonic()
        self._prune_cleanup_tombstones(now)
        self.cleanup_tombstone_expiry[env_id] = now + self.config.cleanup_tombstone_ttl_seconds

    def _has_active_cleanup_tombstone(self, env_id: str) -> bool:
        now = monotonic()
        expiry = self.cleanup_tombstone_expiry.get(env_id)
        if expiry is None:
            return False
        if expiry <= now:
            self.cleanup_tombstone_expiry.pop(env_id, None)
            return False
        return True

    def _prune_cleanup_tombstones(self, now: float) -> None:
        for env_id, expiry in list(self.cleanup_tombstone_expiry.items()):
            if expiry <= now:
                self.cleanup_tombstone_expiry.pop(env_id, None)

    def _discard_session_state(self, env_id: str) -> None:
        self.env_id_to_env.pop(env_id, None)
        self.env_id_to_task_row.pop(env_id, None)
        self.env_id_to_total_reward.pop(env_id, None)
        self.env_id_to_turn_count.pop(env_id, None)
        self.env_id_to_task_success.pop(env_id, None)

    def _forget_operation(self, env_id: str, operation: _SessionOperation) -> None:
        if self.env_id_to_operation.get(env_id) is operation:
            self.env_id_to_operation.pop(env_id, None)

    def _track_background_close(self, task: asyncio.Task[VisGymCloseResponse]) -> None:
        self.background_close_tasks.add(task)
        task.add_done_callback(self._background_close_finished)

    def _background_close_finished(self, task: asyncio.Task[VisGymCloseResponse]) -> None:
        self.background_close_tasks.discard(task)
        if task.cancelled():
            logger.warning("Background VisGym session close was cancelled before cleanup completed.")
            return
        try:
            task.result()
        except BaseException:
            logger.exception("Background VisGym session close failed unexpectedly.")

    def _resolve_task_row(self, body: VisGymSeedSessionRequest) -> VisGymTaskRow:
        if body.task_row is not None:
            return body.task_row
        if body.task_idx is None:
            raise HTTPException(status_code=400, detail="Either task_row or task_idx is required.")
        if body.task_idx >= len(self.task_rows):
            raise HTTPException(
                status_code=400,
                detail=f"task_idx={body.task_idx} out of range; server has {len(self.task_rows)} rows.",
            )
        return VisGymTaskRow.model_validate(self.task_rows[body.task_idx])

    async def _reset_env(self, env: Any, row: VisGymTaskRow) -> tuple[Any, dict[str, Any]]:
        if self._serializes_env_ops(row.env_id):
            async with self.serial_env_op_lock:
                return await self._reset_env_unlocked(env, row)
        return await self._reset_env_unlocked(env, row)

    async def _reset_env_unlocked(self, env: Any, row: VisGymTaskRow) -> tuple[Any, dict[str, Any]]:
        if row.init_state is not None:
            try:
                return await _run_in_threadpool_to_completion(env.reset, seed=row.seed, init_state=row.init_state)
            except TypeError:
                return await _run_in_threadpool_to_completion(
                    env.reset,
                    seed=row.seed,
                    options={"init_state": row.init_state},
                )
        return await _run_in_threadpool_to_completion(env.reset, seed=row.seed)

    async def _step_env(self, env: Any, env_id: str, action_string: str) -> tuple[Any, Any, Any, Any, Any]:
        if self._serializes_env_ops(env_id):
            async with self.serial_env_op_lock:
                return await _run_in_threadpool_to_completion(env.step, action_string)
        return await _run_in_threadpool_to_completion(env.step, action_string)

    async def _image_value(self, env: Any, obs: Any, env_id: str) -> Any:
        """Return something image-like for this observation, rendering if needed.

        The config field and the README both promise a render fallback when the
        observation "is not image-like", but only checking for None means an
        environment returning a state vector or a text-only dict silently sends
        the model a message with no image at all.
        """
        if obs is not None and coerce_images(obs):
            return obs
        if self.config.render_on_missing_image:
            rendered = await self._safe_render(env, env_id)
            if rendered is not None:
                return rendered
            if obs is not None:
                logger.warning(
                    "VisGym env_id=%s produced a non-image observation (%s) and "
                    "env.render() returned nothing; the model will see text only.",
                    env_id,
                    type(obs).__name__,
                )
        return obs

    async def _safe_render(self, env: Any, env_id: str) -> Any:
        try:
            if self._serializes_env_ops(env_id):
                async with self.serial_env_op_lock:
                    return await _run_in_threadpool_to_completion(env.render)
            return await _run_in_threadpool_to_completion(env.render)
        except Exception:
            logger.debug("VisGym env.render failed", exc_info=True)
            return None

    @staticmethod
    def _serializes_env_ops(env_id: str) -> bool:
        return env_id.startswith("matchstick_")

    def _feedback_text(self, info: Any) -> str | None:
        if not self.config.include_env_feedback:
            return None
        if isinstance(info, dict):
            feedback = info.get("env_feedback")
            if feedback:
                return str(feedback)
        return None

    @staticmethod
    def _env_info(info: Any, env_id: str) -> dict[str, Any]:
        if isinstance(info, dict):
            sanitized = sanitize_metadata(info)
            if isinstance(sanitized, dict):
                sanitized.setdefault("env_id", env_id)
                return sanitized
        return {"env_id": env_id, "info": sanitize_metadata(info)}

    @staticmethod
    def _prompt_for_env(env: Any, row: VisGymTaskRow) -> str:
        get_prompt = getattr(env, "get_prompt", None)
        if callable(get_prompt):
            try:
                prompt = get_prompt(**row.prompt_kwargs)
            except TypeError:
                prompt = get_prompt()
            if isinstance(prompt, str):
                return prompt
        return (
            f"You are interacting with the VisGym environment {row.env_id}. "
            "Each turn, inspect the image and return exactly one valid action."
        )


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=partial(create_server, VisGymResourcesServer, VisGymResourcesServerConfig),
    request_model=VisGymAgentVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
    invoke=invoke,
)


if __name__ == "__main__":
    VisGymResourcesServer.run_webserver()
