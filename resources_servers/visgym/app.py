# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import logging
import math
import os
import sys
import threading
import types
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import SimpleResourcesServer
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


logger = logging.getLogger(__name__)

# distance_delta reward shaping expresses progress as a fraction of the
# episode's own starting distance (see _training_step_reward), so one shared
# weight works across environments whose raw distance is in unrelated units
# (maze cells, degrees, pixel-space color distance, ...). 0.5 is the ceiling:
# above it an unsolved episode that fully closes the distance without ever
# emitting the stop action could outscore a solved one, which would let the
# policy learn to farm progress instead of finishing tasks.
DEFAULT_SHAPING_WEIGHT = 0.3
MAX_SHAPING_WEIGHT = 0.5


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


def _install_skimage_io_compatibility() -> None:
    """Provide the one optional skimage API imported by RefCOCO's loader."""
    try:
        importlib.import_module("skimage.io")
        return
    except ImportError:
        pass

    import numpy as np
    from PIL import Image

    skimage_module = types.ModuleType("skimage")
    io_module = types.ModuleType("skimage.io")

    def imread(path: str) -> Any:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    io_module.imread = imread  # type: ignore[attr-defined]
    skimage_module.io = io_module  # type: ignore[attr-defined]
    sys.modules.setdefault("skimage", skimage_module)
    sys.modules.setdefault("skimage.io", io_module)


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


def _install_maze3d_headless_renderer() -> None:
    """Replace Maze3D's process-global Ursina scene with a session-safe renderer."""
    maze_module = importlib.import_module("gymnasium.envs.maze_3d.maze_3d")
    if getattr(maze_module, "_nemo_gym_headless_renderer_installed", False):
        return

    import numpy as np
    from PIL import Image, ImageDraw

    class HeadlessScene:
        entities: tuple[Any, ...] = ()

        @staticmethod
        def clear() -> None:
            return None

    def no_op(*_args: Any, **_kwargs: Any) -> None:
        return None

    def is_wall(env: Any, location: Any) -> bool:
        row, col = int(location[0]), int(location[1])
        return not (0 <= row < env.maze_height and 0 <= col < env.maze_width and env.maze_map[row, col] != 1)

    def render_frame(env: Any) -> Any:
        render_size = env._render_size or (336, 336)
        width, height = int(render_size[0]), int(render_size[1])
        image = Image.new("RGB", (width, height), (72, 78, 88))
        draw = ImageDraw.Draw(image)
        horizon = height // 2
        draw.rectangle((0, horizon, width, height), fill=(104, 99, 88))

        direction = np.asarray(env._action_to_direction[env._cam_dir], dtype=int)
        right = np.asarray([direction[1], -direction[0]], dtype=int)
        origin = np.asarray(env._agent_location, dtype=int)
        target = np.asarray(env._target_location, dtype=int)
        max_depth = max(2, min(6, max(env.maze_width, env.maze_height) - 2))

        visible_depth = max_depth
        for depth in range(1, max_depth + 1):
            if is_wall(env, origin + direction * depth):
                visible_depth = depth
                break

        def bounds(depth: int) -> tuple[int, int, int, int]:
            scale = 1.0 / (1.0 + 0.58 * depth)
            half_w = max(10, int(width * 0.48 * scale))
            half_h = max(10, int(height * 0.48 * scale))
            return width // 2 - half_w, horizon - half_h, width // 2 + half_w, horizon + half_h

        far = bounds(visible_depth)
        draw.rectangle(far, fill=(112, 108, 100), outline=(44, 47, 52), width=3)

        for depth in range(visible_depth, 0, -1):
            near_box = bounds(depth - 1)
            far_box = bounds(depth)
            cell = origin + direction * max(0, depth - 1)
            left_wall = is_wall(env, cell - right)
            right_wall = is_wall(env, cell + right)

            if left_wall:
                polygon = [
                    (near_box[0], near_box[1]),
                    (far_box[0], far_box[1]),
                    (far_box[0], far_box[3]),
                    (near_box[0], near_box[3]),
                ]
                draw.polygon(polygon, fill=(88, 94, 102), outline=(42, 45, 50))
            if right_wall:
                polygon = [
                    (far_box[2], far_box[1]),
                    (near_box[2], near_box[1]),
                    (near_box[2], near_box[3]),
                    (far_box[2], far_box[3]),
                ]
                draw.polygon(polygon, fill=(82, 89, 98), outline=(42, 45, 50))

            draw.line((near_box[0], near_box[3], far_box[0], far_box[3]), fill=(55, 58, 61), width=2)
            draw.line((near_box[2], near_box[3], far_box[2], far_box[3]), fill=(55, 58, 61), width=2)

        for depth in range(1, visible_depth + 1):
            location = origin + direction * depth
            if np.array_equal(location, target) and not is_wall(env, location):
                target_box = bounds(depth)
                radius = max(6, int(width * 0.08 / (1.0 + 0.45 * depth)))
                cx = (target_box[0] + target_box[2]) // 2
                cy = (target_box[1] + target_box[3]) // 2
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(220, 42, 48))
                break

        return np.array(image, dtype=np.uint8, copy=True)

    maze_module.scene = HeadlessScene()
    maze_module.RandomMaze3DEnv._initialize_ursina = no_op
    maze_module.RandomMaze3DEnv._build_scene_entities = no_op
    maze_module.RandomMaze3DEnv._render_frame = render_frame
    maze_module._nemo_gym_headless_renderer_installed = True


def _render_fetch_state(env: Any, obs: dict[str, Any]) -> Any:
    """Render Fetch state without creating an OpenGL context."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    size = 224
    output_size = 128
    image = Image.new("RGB", (size, size), (238, 240, 241))
    draw = ImageDraw.Draw(image)
    table = (16, 16, 178, 208)
    draw.rectangle(table, fill=(214, 218, 216), outline=(45, 49, 52), width=2)

    state = np.asarray(obs.get("observation", []), dtype=float)
    default_pose = np.array([1.3, 0.75, 0.5], dtype=float)

    def pose(value: Any, fallback: Any) -> Any:
        candidate = np.asarray(value, dtype=float).reshape(-1)
        return candidate[:3] if candidate.size >= 3 else np.asarray(fallback, dtype=float)

    gripper = pose(state[:3], default_pose)
    achieved = pose(obs.get("achieved_goal", gripper), gripper)
    goal = pose(obs.get("desired_goal", getattr(env, "goal", gripper)), gripper)
    has_object = bool(getattr(env, "has_object", False))
    object_pos = pose(state[3:6], achieved) if has_object else achieved

    x_bounds = (0.95, 1.65)
    y_bounds = (0.35, 1.15)
    z_bounds = (0.35, 1.05)

    def project_xy(position: Any) -> tuple[int, int]:
        x = float(np.clip(position[0], *x_bounds))
        y = float(np.clip(position[1], *y_bounds))
        px = table[0] + int((x - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) * (table[2] - table[0]))
        py = table[3] - int((y - y_bounds[0]) / (y_bounds[1] - y_bounds[0]) * (table[3] - table[1]))
        return px, py

    try:
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except OSError:
        label_font = ImageFont.load_default()

    # Goal, object, and gripper are deliberately high-contrast and shape-distinct.
    gx, gy = project_xy(goal)
    draw.ellipse((gx - 9, gy - 9, gx + 9, gy + 9), outline=(205, 36, 45), width=4)
    draw.text((gx + 10, gy - 8), "T", fill=(160, 20, 28), font=label_font)
    if has_object:
        ox, oy = project_xy(object_pos)
        draw.rectangle((ox - 7, oy - 7, ox + 7, oy + 7), fill=(238, 151, 35), outline=(76, 50, 16), width=2)
        draw.text((ox + 9, oy - 8), "O", fill=(92, 55, 8), font=label_font)
    rx, ry = project_xy(gripper)
    draw.line((rx - 9, ry, rx + 9, ry), fill=(20, 93, 178), width=4)
    draw.line((rx, ry - 9, rx, ry + 9), fill=(20, 93, 178), width=4)
    draw.text((rx + 10, ry - 8), "G", fill=(12, 62, 126), font=label_font)

    # Direction labels match the continuous Fetch action components [dx, dy, dz].
    draw.text((139, 190), "X+", fill=(20, 24, 26), font=label_font)
    draw.text((20, 18), "Y+", fill=(20, 24, 26), font=label_font)
    draw.line((132, 202, 169, 202), fill=(20, 24, 26), width=2)
    draw.polygon(((169, 202), (162, 198), (162, 206)), fill=(20, 24, 26))
    draw.line((23, 57, 23, 27), fill=(20, 24, 26), width=2)
    draw.polygon(((23, 27), (19, 34), (27, 34)), fill=(20, 24, 26))

    gauge_left, gauge_right = 190, 214
    gauge_top, gauge_bottom = 25, 202
    draw.rectangle(
        (gauge_left, gauge_top, gauge_right, gauge_bottom), fill=(248, 248, 248), outline=(45, 49, 52), width=2
    )
    draw.text((190, 7), "Z+", fill=(20, 24, 26), font=label_font)

    def draw_height(position: Any, color: tuple[int, int, int], offset: int) -> None:
        z = float(np.clip(position[2], *z_bounds))
        py = gauge_bottom - int((z - z_bounds[0]) / (z_bounds[1] - z_bounds[0]) * (gauge_bottom - gauge_top))
        draw.line((gauge_left + 2 + offset, py, gauge_right - 2 + offset, py), fill=color, width=3)

    draw_height(goal, (205, 36, 45), 0)
    draw_height(gripper, (20, 93, 178), -4)
    if has_object:
        draw_height(object_pos, (238, 151, 35), 4)

    image = image.resize((output_size, output_size), Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.uint8, copy=True)


@contextlib.contextmanager
def _fetch_gl_environment() -> Any:
    """Scope Fetch's GLFW rendering settings to the calling block.

    Setting these process-wide is order-dependent poison in a blended
    manifest: `_ensure_headless_defaults` uses setdefault, so once a fetch_*
    session flips MUJOCO_GL to glfw and drops PYOPENGL_PLATFORM, nothing
    restores egl and every later EGL-dependent environment in that worker
    (maze_3d, mental_rotation_3d, ...) fails at reset -- unless the fetch rows
    happened to be scheduled last.
    """
    previous = {name: os.environ.get(name) for name in ("MUJOCO_GL", "PYOPENGL_PLATFORM")}
    os.environ["MUJOCO_GL"] = "glfw"
    os.environ.pop("PYOPENGL_PLATFORM", None)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _install_fetch_headless_renderer() -> None:
    """Keep Fetch physics while replacing its EGL-dependent image renderer."""

    module_names = (
        "gymnasium_robotics.envs.fetch.pick_and_place_discrete",
        "gymnasium_robotics.envs.fetch.reach_discrete",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        if getattr(module, "_nemo_gym_headless_renderer_installed", False):
            continue

        mixin = module._DiscreteMixin
        original_init = mixin.__init__

        def headless_init(self: Any, *args: Any, _original_init: Any = original_init, **kwargs: Any) -> None:
            _original_init(self, *args, **kwargs)
            import numpy as np

            self._image_height = 128
            self._image_width = 128
            self._use_follow_camera = False
            spaces = self.observation_space.spaces.copy()
            spaces["image"] = gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
            self.observation_space = gym.spaces.Dict(spaces)

        def headless_get_obs(self: Any, _mixin: Any = mixin) -> dict[str, Any]:
            obs = super(_mixin, self)._get_obs()
            obs["image"] = _render_fetch_state(self, obs)
            return obs

        mixin.__init__ = headless_init
        mixin._get_obs = headless_get_obs
        module._nemo_gym_headless_renderer_installed = True


def _ensure_visgym_importable(env_id: str | None = None) -> Any:
    repo_root = os.getenv("VISGYM_REPO_ROOT")
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    global gym
    if gym is None:
        gym = importlib.import_module("gymnasium")
    importlib.import_module("gymnasium.envs")
    if env_id and env_id.startswith("referring_dot_pointing/"):
        _install_skimage_io_compatibility()
    if env_id and env_id.startswith("mental_rotation_3d_cube/"):
        _install_mental_rotation_3d_renderer_compatibility()
    if env_id and env_id.startswith("maze_3d/"):
        _install_maze3d_headless_renderer()
    if env_id and env_id.startswith("fetch_"):
        _install_fetch_headless_renderer()
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
    """Env-id-parametric server wrapping VisGym/Gymnasium envs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: VisGymResourcesServerConfig

    task_rows: list[dict[str, Any]] = Field(default_factory=list)
    env_id_to_env: dict[str, Any] = Field(default_factory=dict)
    env_id_to_total_reward: dict[str, float] = Field(default_factory=lambda: defaultdict(float))
    env_id_to_task_row: dict[str, dict[str, Any]] = Field(default_factory=dict)
    env_id_to_turn_count: dict[str, int] = Field(default_factory=lambda: defaultdict(int))
    env_id_to_reward_state: dict[str, dict[str, float]] = Field(default_factory=dict)
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

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/step")(self.step)
        app.post("/close")(self.close)
        return app

    async def seed_session(self, request: Request, body: VisGymSeedSessionRequest) -> VisGymSeedSessionResponse:
        row = self._resolve_task_row(body)
        env_kwargs = _resolve_asset_kwargs(row.env_kwargs)
        if row.seed_key:
            env_kwargs[row.seed_key] = row.seed

        try:
            if row.env_id.startswith("referring_dot_pointing/"):
                _install_skimage_io_compatibility()
            if row.env_id.startswith("mental_rotation_3d_cube/"):
                _install_mental_rotation_3d_renderer_compatibility()
            gym_module = _ensure_visgym_importable()
            if row.env_id.startswith("maze_3d/"):
                _install_maze3d_headless_renderer()
            if row.env_id.startswith("fetch_"):
                _install_fetch_headless_renderer()
                with _fetch_gl_environment():
                    env = await run_in_threadpool(gym_module.make, row.env_id, **env_kwargs)
            else:
                env = await run_in_threadpool(gym_module.make, row.env_id, **env_kwargs)
            obs, info = await self._reset_env(env, row)
            info = self._augment_info(env, info, row.env_id)
        except Exception as exc:
            logger.exception("Failed to seed VisGym env %s with kwargs=%s", row.env_id, env_kwargs)
            raise HTTPException(
                status_code=500,
                detail=f"env construction failed: {type(exc).__name__}: {exc}",
            ) from exc

        env_id = str(uuid.uuid4())
        self.env_id_to_env[env_id] = env
        self.env_id_to_task_row[env_id] = row.model_dump(mode="json")
        self.env_id_to_total_reward[env_id] = 0.0
        self.env_id_to_turn_count[env_id] = 0
        self.env_id_to_reward_state[env_id] = self._initial_reward_state(info, row.task_metadata)

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

        return VisGymSeedSessionResponse(env_id=env_id, obs=[obs_msg])

    def _horizon_reached(self, env_id: str, row_dict: dict[str, Any]) -> bool:
        """True when this session has used up its per-task horizon_cap."""
        return bool(
            self.config.enforce_horizon_cap
            and row_dict.get("horizon_cap") is not None
            and self.env_id_to_turn_count[env_id] >= row_dict["horizon_cap"]
        )

    async def step(self, request: Request, body: VisGymStepRequest) -> VisGymStepResponse:
        if body.env_id not in self.env_id_to_env:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown env_id={body.env_id}; was the session closed?",
            )

        env = self.env_id_to_env[body.env_id]
        row_dict = self.env_id_to_task_row[body.env_id]
        env_id_str = row_dict["env_id"]

        try:
            obs, reward, terminated, truncated, info = await self._step_env(env, env_id_str, body.action_string)
            info = self._augment_info(env, info, env_id_str)
        except Exception as exc:
            logger.warning(
                "VisGym env.step raised on env_id=%s (%s) action=%r: %s: %s",
                body.env_id,
                env_id_str,
                body.action_string,
                type(exc).__name__,
                exc,
            )
            recovery = observation_to_user_message(
                image_value=await self._safe_render(env, env_id_str),
                env_id=env_id_str,
                feedback_text=f"Invalid action {body.action_string!r}: {type(exc).__name__}: {exc}",
                image_format=self.config.image_format,
                image_jpeg_quality=self.config.image_jpeg_quality,
                skip_images=self.config.skip_images,
            )
            recovery = attach_env_info(recovery, {"env_step_exception": str(exc)})
            # A rejected action still costs a turn. Returning early without
            # counting it means horizon_cap can never terminate an episode of
            # nothing but invalid actions, leaving the agent's max_steps as the
            # only bound -- and that defaults to None.
            self.env_id_to_turn_count[body.env_id] += 1
            horizon_terminated = self._horizon_reached(body.env_id, row_dict)
            return VisGymStepResponse(
                obs=[recovery],
                reward=0.0,
                done=horizon_terminated,
                horizon_terminated=horizon_terminated,
            )

        self.env_id_to_turn_count[body.env_id] += 1
        raw_env_reward = float(reward)
        training_step_reward = self._training_step_reward(
            body.env_id,
            raw_env_reward,
            info,
            row_dict.get("task_metadata", {}),
        )
        self.env_id_to_total_reward[body.env_id] += training_step_reward
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
        env = self.env_id_to_env.pop(body.env_id, None)
        self.env_id_to_task_row.pop(body.env_id, None)
        self.env_id_to_turn_count.pop(body.env_id, None)
        self.env_id_to_reward_state.pop(body.env_id, None)
        if env is None:
            return VisGymCloseResponse(success=True, message="already closed")

        try:
            await run_in_threadpool(env.close)
        except Exception as exc:
            logger.warning("VisGym env.close raised on env_id=%s: %s", body.env_id, exc)
            return VisGymCloseResponse(success=False, message=repr(exc))

        return VisGymCloseResponse(success=True, message="ok")

    async def verify(self, request: Request, body: VisGymAgentVerifyRequest) -> VisGymAgentVerifyResponse:
        env_id = body.response.env_id
        known_env_id = env_id in self.env_id_to_total_reward
        reward = self.env_id_to_total_reward.pop(env_id, 0.0)
        metadata = dict(body.response.metadata or {})
        metadata["training_reward"] = str(reward)
        body.response.metadata = metadata
        if not known_env_id:
            logger.info("/verify drained unknown VisGym env_id=%s; returning 0.0", env_id)
        return VisGymAgentVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reward=reward,
        )

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
                return await run_in_threadpool(env.reset, seed=row.seed, init_state=row.init_state)
            except TypeError:
                return await run_in_threadpool(
                    env.reset,
                    seed=row.seed,
                    options={"init_state": row.init_state},
                )
        return await run_in_threadpool(env.reset, seed=row.seed)

    async def _step_env(self, env: Any, env_id: str, action_string: str) -> tuple[Any, Any, Any, Any, Any]:
        if self._serializes_env_ops(env_id):
            async with self.serial_env_op_lock:
                return await run_in_threadpool(env.step, action_string)
        return await run_in_threadpool(env.step, action_string)

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
                    return await run_in_threadpool(env.render)
            return await run_in_threadpool(env.render)
        except Exception:
            logger.debug("VisGym env.render failed", exc_info=True)
            return None

    @staticmethod
    def _serializes_env_ops(env_id: str) -> bool:
        return env_id.startswith("matchstick_")

    @staticmethod
    def _distance_value(info: Any, info_key: str) -> float | None:
        if not isinstance(info, dict):
            return None
        try:
            value = float(info[info_key])
        except (KeyError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @staticmethod
    def _augment_info(env: Any, info: Any, env_id: str) -> Any:
        if not isinstance(info, dict):
            return info
        enriched = dict(info)
        if env_id.startswith("matchstick_equation/"):
            is_correct = info.get("is_correct")
            if isinstance(is_correct, bool):
                enriched["matchstick_distance"] = 0.0 if is_correct else 1.0
            return enriched
        if env_id.startswith("fetch_pick_and_place/"):
            try:
                gripper = tuple(float(value) for value in env.unwrapped.data.site("robot0:grip").xpos)
                object_position = tuple(float(value) for value in info["achieved_goal"])
                goal_position = tuple(float(value) for value in info["desired_goal"])
                if not (len(gripper) == len(object_position) == len(goal_position) == 3):
                    raise ValueError("Fetch positions must be three-dimensional")
                distance = math.dist(gripper, object_position) + math.dist(object_position, goal_position)
                if math.isfinite(distance):
                    enriched["fetch_pick_distance"] = distance
            except (AttributeError, KeyError, TypeError, ValueError):
                logger.debug("Could not compute Fetch pick-and-place distance", exc_info=True)
            return enriched
        if not env_id.startswith("sliding_block/"):
            return info
        try:
            core_env = env.unwrapped
            distance = 0.0
            for block_id, block in core_env.blocks.items():
                current = block["position"]
                target = core_env.target_blocks[block_id]["position"]
                distance += abs(float(current[0]) - float(target[0]))
                distance += abs(float(current[1]) - float(target[1]))
            enriched["sliding_distance"] = distance
        except (AttributeError, KeyError, TypeError, ValueError):
            logger.debug("Could not compute sliding-block distance", exc_info=True)
        return enriched

    def _initial_reward_state(
        self,
        info: Any,
        task_metadata: dict[str, Any],
    ) -> dict[str, float]:
        shaping = task_metadata.get("reward_shaping", {})
        if not isinstance(shaping, dict) or shaping.get("type") != "distance_delta":
            return {}
        info_key = str(shaping.get("info_key", "distance"))
        initial_distance = self._distance_value(info, info_key)
        if initial_distance is None or initial_distance <= 0:
            # Nothing to normalize progress against: either the environment
            # hasn't reported the key yet, or the task spawned already at the
            # goal (dividing by zero). Either way shaping stays off for the
            # whole episode and only the terminal reward is paid.
            return {}
        return {"initial_distance": initial_distance, "previous_progress": 0.0}

    @staticmethod
    def _clamped_shaping_weight(raw_weight: Any, reward_state: dict[str, float]) -> float:
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            weight = DEFAULT_SHAPING_WEIGHT
        clamped = min(MAX_SHAPING_WEIGHT, max(0.0, weight))
        if clamped != weight and not reward_state.get("warned_weight_out_of_range"):
            reward_state["warned_weight_out_of_range"] = True
            logger.warning(
                "VisGym reward_shaping weight=%r is outside (0, %.1f]; clamped to %.3f so a "
                "solved episode always scores higher than an unsolved one.",
                weight,
                MAX_SHAPING_WEIGHT,
                clamped,
            )
        return clamped

    def _training_step_reward(
        self,
        env_id: str,
        raw_reward: float,
        info: Any,
        task_metadata: dict[str, Any],
    ) -> float:
        shaping = task_metadata.get("reward_shaping", {})
        if not isinstance(shaping, dict) or shaping.get("type") != "distance_delta":
            return raw_reward

        info_key = str(shaping.get("info_key", "distance"))
        current_distance = self._distance_value(info, info_key)
        reward_state = self.env_id_to_reward_state.setdefault(env_id, {})
        initial_distance = reward_state.get("initial_distance")
        if current_distance is None or initial_distance is None:
            # A shaping block naming a key the environment never reports, or
            # one whose initial distance was non-positive (see
            # _initial_reward_state), is indistinguishable from no shaping at
            # all: every step silently falls through to the terminal-only
            # reward, and the run just learns slowly for no visible reason.
            # Say it once per session.
            if current_distance is None and not reward_state.get("warned_missing_info_key"):
                reward_state["warned_missing_info_key"] = True
                logger.warning(
                    "VisGym reward_shaping is configured with info_key=%r, but the "
                    "environment did not report it on this step; the shaped term is "
                    "inactive and only the terminal reward will be paid. Available "
                    "numeric info keys: %s",
                    info_key,
                    sorted(
                        k for k, v in (info or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                    )
                    if isinstance(info, dict)
                    else "<non-mapping info>",
                )
            return raw_reward

        # Progress is a potential function normalized to the episode's own
        # starting distance: Phi(s) = clip((initial - current) / initial, 0,
        # 1). The per-step shaped term Phi(s') - Phi(s) telescopes across the
        # episode to exactly Phi(final) in [0, 1], regardless of the raw
        # distance's units.
        progress = min(1.0, max(0.0, (initial_distance - current_distance) / initial_distance))
        previous_progress = reward_state.get("previous_progress", 0.0)
        reward_state["previous_progress"] = progress

        weight = self._clamped_shaping_weight(shaping.get("weight", DEFAULT_SHAPING_WEIGHT), reward_state)
        shaped_delta = weight * (progress - previous_progress)
        # raw_reward is 0.0 on every non-terminal step and one of {0.0, 1.0}
        # -- every VisGym environment's _compute_reward returns exactly one
        # of those two values -- on the step that ends the episode. Mixing it
        # with the bounded shaped_delta as a convex combination means the
        # reward returned here, and therefore its sum across the whole
        # episode (which telescopes to (1 - weight) * terminal + weight *
        # Phi(final)), is always in [0, 1].
        return (1.0 - weight) * raw_reward + shaped_delta

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


if __name__ == "__main__":
    VisGymResourcesServer.run_webserver()
