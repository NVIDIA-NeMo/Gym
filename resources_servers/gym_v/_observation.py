# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import base64
import io
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from ._metadata import sanitize_metadata
from .schemas import GymVEnvStateEasyInputMessage


@runtime_checkable
class ObservationLike(Protocol):
    """Structural type matching `gym_v.Observation` for helper-level use.

    We use a Protocol instead of importing `gym_v.Observation` so this module
    is unit-testable without `gym_v` installed. The server itself imports the
    real `gym_v.Env` / `gym_v.Observation` elsewhere when it actually runs envs.
    """

    text: str | None
    image: Image.Image | list[Image.Image] | None
    metadata: dict[str, Any]


_VISIBLE_METADATA_KEYS: dict[str, list[str]] = {
    # reasoning-gym envs expose state_text + text_prompt in obs.metadata.
    # text_prompt duplicates env.description (already in prefix_text) plus
    # the ASCII grid from state_text — surfacing both means the model sees
    # rules+question+options TWICE and the ASCII grid TWICE. That wastes
    # ~900 tokens and lets the model solve the task from text alone,
    # bypassing the image entirely.
    #
    # For envs where we want to test visual reasoning, set the allowlist
    # to [] so only prefix_text (env.description) + image are sent.
    # For envs where the ASCII fallback is intentional (e.g. GameOfLife
    # whose grid output format matches state_text), keep ["state_text"].
    # "Algorithmic/GameOfLife-v0": ["state_text"],
    "Algorithmic/GameOfLife-v0": [],
    "Logic/MiniSudoku-v0": [],
    "Puzzles/TowerOfHanoi-v0": [],
    "Puzzles/Maze-QA-v0": ["state_text"],
    "Arc/ArcAgi-v0": [],
    "Games/FrozenLake-v0": [],
    "Games/Game2048-v0": [],
    "Games/LightsOut-v0": [],
    "Games/Minesweeper-v0": [],
    "Spatial/DoorKey-v0": [],
    "Spatial/FourRooms2D-v0": [],
}


def image_to_data_url(
    image: Image.Image,
    fmt: str = "PNG",
    jpeg_quality: int = 90,
    max_image_wh: int | None = None,
) -> str:
    """Encode a PIL image as an OpenAI-compatible base64 data URL.

    If ``max_image_wh`` is set and ``max(W, H) > max_image_wh``, the image is
    resized in aspect-ratio-preserving fashion so that ``max(W, H) ==
    max_image_wh`` — both dimensions are scaled by ``max_image_wh / max(W,
    H)`` via ``PIL.Image.thumbnail``. Examples with ``max_image_wh=512``:
    ``1220x1040 → 512x436``, ``512x1024 → 256x512``, ``400x400 → 400x400``
    (unchanged, already ≤ cap).
    """

    if max_image_wh is not None and max(image.size) > max_image_wh:
        # thumbnail mutates in place; copy so we don't clobber the caller's
        # cached observation image across a subsequent env reset/step.
        image = image.copy()
        image.thumbnail(
            (max_image_wh, max_image_wh),
            Image.Resampling.LANCZOS,
        )

    normalized_fmt = fmt.upper()
    buf = io.BytesIO()
    if normalized_fmt == "JPEG":
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        mime = "jpeg"
    else:
        image.save(buf, format="PNG")
        mime = "png"

    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


def _format_visible_metadata(env_id: str, metadata: dict[str, Any]) -> str | None:
    """Render the per-env allow-listed metadata subset into prompt text.

    Unlisted envs default to an empty allow-list (i.e., no `obs.metadata`
    fields are surfaced as prompt text). This is behaviour-identical to the
    explicit `[]` entries in `_VISIBLE_METADATA_KEYS` (e.g. `Games/FrozenLake-v0`,
    `Spatial/DoorKey-v0`) — the allow-list still gates which envs can promote
    metadata into the model's user-message text; the default just doesn't
    crash on envs that haven't been audited yet. To expose metadata for a new
    env, add a key + list of fields explicitly.
    """

    keys = _VISIBLE_METADATA_KEYS.get(env_id, [])
    if not keys:
        return None

    sanitized = sanitize_metadata(metadata)
    parts = [f"{key}: {sanitized[key]}" for key in keys if key in sanitized]
    return "\n".join(parts) if parts else None


def observation_to_user_message(
    obs: ObservationLike,
    env_id: str,
    prefix_text: str | None = None,
    image_format: str = "PNG",
    image_jpeg_quality: int = 90,
    skip_images: bool = False,
    max_image_wh: int | None = None,
) -> GymVEnvStateEasyInputMessage:
    """Build the multimodal user message emitted by the Gym-V resources server.

    ``max_image_wh`` (optional) caps ``max(W, H)`` of each observation image
    with aspect ratio preserved; see :func:`image_to_data_url` for details.
    """

    parts: list[dict[str, Any]] = []

    text_parts: list[str] = []
    if prefix_text:
        text_parts.append(prefix_text)
    if obs.text:
        text_parts.append(obs.text)

    metadata_text = _format_visible_metadata(env_id, obs.metadata)
    if metadata_text:
        text_parts.append(metadata_text)

    if text_parts:
        parts.append({"type": "input_text", "text": "\n\n".join(text_parts)})

    if obs.image is not None and not skip_images:
        images = obs.image if isinstance(obs.image, list) else [obs.image]
        for image in images:
            parts.append(
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(
                        image,
                        fmt=image_format,
                        jpeg_quality=image_jpeg_quality,
                        max_image_wh=max_image_wh,
                    ),
                    # ResponseInputImageParam.detail is required by the OpenAI
                    # Responses API; vLLM accepts "auto" verbatim.
                    "detail": "auto",
                }
            )

    return GymVEnvStateEasyInputMessage(role="user", content=parts, env_info=None)


def _attach_env_info(obs_msg: GymVEnvStateEasyInputMessage, info_dict: dict[str, Any]) -> GymVEnvStateEasyInputMessage:
    """Attach sanitized env.step info to the inspector-visible side channel."""

    return obs_msg.model_copy(update={"env_info": sanitize_metadata(info_dict)})
