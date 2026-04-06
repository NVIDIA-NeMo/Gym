# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert CUBE `Observation` / `ActionSchema` to NeMo-Gym message and OpenAI tool shapes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from nemo_gym.openai_utils import NeMoGymEasyInputMessage
from resources_servers.cube.schemas import CubeEnvStateEasyInputMessage


if TYPE_CHECKING:
    from cube.core import ActionSchema, Observation


def _parameters_for_openai_strict_tool(params: dict[str, Any]) -> dict[str, Any]:
    """OpenAI Responses strict tools require ``required`` to include every key in ``properties``."""
    out = dict(params)
    props = out.get("properties")
    if isinstance(props, dict) and props:
        out["required"] = sorted(props.keys())
    out.setdefault("additionalProperties", False)
    return out


def action_schemas_to_openai_tools(schemas: List["ActionSchema"]) -> list[dict[str, Any]]:
    """Build Responses-API function tools from CUBE action schemas."""
    tools: list[dict[str, Any]] = []
    for schema in schemas:
        params = _parameters_for_openai_strict_tool(dict(schema.parameters))
        tools.append(
            {
                "type": "function",
                "name": schema.name,
                "description": schema.description,
                "parameters": params,
                "strict": True,
            }
        )
    return tools


def _cube_llm_dict_to_easy_message(msg: dict) -> NeMoGymEasyInputMessage | CubeEnvStateEasyInputMessage:
    """Map a single CUBE `to_llm_message()` dict to NeMoGymEasyInputMessage."""
    role = msg.get("role", "user")
    content = msg.get("content")
    if role == "tool":
        role = "user"
        if isinstance(content, str):
            content = f"[tool result]\n{content}"
        elif isinstance(content, list):
            content = [
                {"type": "input_text", "text": f"[tool result]\n{_textify_content_list(content)}"},
            ]
    if role not in ("user", "assistant", "system", "developer"):
        role = "user"

    if isinstance(content, list):
        content = _normalize_content_list(content)

    cls = CubeEnvStateEasyInputMessage if _is_desktop_env_content(content) else NeMoGymEasyInputMessage
    return cls(role=role, content=content, type="message")  # type: ignore[arg-type]


def _normalize_content_list(parts: list) -> list:
    """Map CUBE / chat-completions-style parts to OpenAI Responses `input_*` shapes."""
    out: list = []
    for p in parts:
        if not isinstance(p, dict):
            out.append(p)
            continue
        t = p.get("type")
        if t == "image_url":
            iu = p.get("image_url")
            url = iu.get("url", "") if isinstance(iu, dict) else str(iu or "")
            out.append({"type": "input_image", "image_url": url, "detail": "auto"})
        elif t == "text":
            out.append({"type": "input_text", "text": str(p.get("text", ""))})
        elif t == "input_image":
            out.append({**p, "detail": p.get("detail", "auto")})
        else:
            out.append(p)
    return out


def _textify_content_list(parts: list) -> str:
    chunks: list[str] = []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            chunks.append(str(p.get("text", "")))
        else:
            chunks.append(str(p))
    return "\n".join(chunks)


def _is_desktop_env_content(content: Any) -> bool:
    """Heuristic: screenshot-bearing turns are tagged as env-state for collapse."""
    if isinstance(content, list):
        for p in content:
            if not isinstance(p, dict):
                continue
            if p.get("type") in ("image_url", "input_image"):
                return True
    return False


def observation_to_input_messages(obs: "Observation") -> list[NeMoGymEasyInputMessage | CubeEnvStateEasyInputMessage]:
    """Turn a CUBE observation into NeMo-Gym input messages."""
    return [_cube_llm_dict_to_easy_message(m) for m in obs.to_llm_messages()]
