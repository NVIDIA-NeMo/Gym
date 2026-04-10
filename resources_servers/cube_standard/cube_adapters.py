# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Format conversion layer between CUBE and NeMo Gym representations.

All functions in this module are pure (no I/O side effects) except for
_serialize_env_output(), which writes screenshot files to disk.

Conversion functions:
    action_schema_to_function_tool_param  — CUBE ActionSchema → FunctionToolParam
    _action_set_to_function_tool_params   — list[ActionSchema] → list[FunctionToolParam]
    _observation_to_nemo_gym_messages     — CUBE Observation → list[NeMoGymEasyInputMessage]
    _serialize_env_output                 — EnvironmentOutput → (output: str, content_type: str)
    build_action                          — (call_id, name, arguments) → CUBE Action
    extract_reward                        — EnvironmentOutput → (reward: float, info: dict)
"""

import base64
import logging
from pathlib import Path
from typing import Any

from openai.types.responses import FunctionToolParam

from nemo_gym.openai_utils import NeMoGymEasyInputMessage

logger = logging.getLogger(__name__)


def action_schema_to_function_tool_param(schema: Any) -> FunctionToolParam:
    """
    Convert a CUBE ActionSchema to a NeMo Gym FunctionToolParam.

    CUBE format (nested):
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

    NeMo Gym / OpenAI Responses API format (flat):
        {"type": "function", "name": ..., "description": ..., "parameters": ...}

    The FunctionToolParam constructor accepts keyword args directly, so we unwrap
    the nested "function" key and pass name/description/parameters at the top level.
    """
    cube_dict = schema.as_dict()
    fn = cube_dict["function"]
    return FunctionToolParam(
        type="function",
        name=fn["name"],
        description=fn.get("description", ""),
        parameters=fn.get("parameters", {}),
        strict=None,
    )


def _action_set_to_function_tool_params(action_set: Any) -> list[FunctionToolParam]:
    """
    Convert a CUBE action set (list of ActionSchema) to a list of FunctionToolParam.

    The STOP_ACTION ("final_step") is included when accept_agent_stop=True (CUBE default).
    It converts naturally — task.step() detects it internally and returns done=True.
    """
    return [action_schema_to_function_tool_param(s) for s in action_set]


def _observation_to_nemo_gym_messages(obs: Any) -> list[NeMoGymEasyInputMessage]:
    """
    Convert a CUBE Observation to a list of NeMoGymEasyInputMessage.

    Used for the initial observation from task.reset() only.
    Subsequent step observations are handled by _serialize_env_output().

    CUBE to_llm_messages() returns ChatCompletion format (role may be "tool").
    NeMoGymEasyInputMessage only accepts: user / assistant / system / developer.
    Remapping: "tool" → "user", any other unknown role → "user".
    """
    messages = obs.to_llm_messages()
    result = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role not in ("user", "assistant", "system", "developer"):
            logger.debug("Remapping unknown role '%s' to 'user' in observation message.", role)
            role = "user"
        result.append(NeMoGymEasyInputMessage(role=role, content=content))
    return result


def _serialize_env_output(
    env_output: Any,
    session_id: str,
    step_index: int,
    screenshot_dir: Path,
    base_url: str,
) -> tuple[str, str]:
    """
    Serialize a CUBE EnvironmentOutput to (output, content_type) for CubeStepResponse.

    For text/structured observations:
        content_type = "text/plain"
        output = concatenated text strings

    For image observations (ImageContent with base64 data URL):
        content_type = "image/png"
        output = URL to the PNG file served at /screenshots/<session_id>/<filename>

    The model server fetches the URL directly from the cube server's static file
    endpoint — no base64 passes through the agent's JSON payload.

    For mixed observations (text + image in same step):
        Returns the image URL if any ImageContent is present; text is logged only.
        Rationale: vision models receive both the tool_call_output text ("Screenshot captured")
        and the image_url content block, so no information is lost.

    StepError: always returns ("Error: <message>", "text/plain").

    Why write to disk:
        FastAPI's StaticFiles serves from disk. The model server (potentially on
        a different host) makes a GET request to fetch the file. In-memory
        alternatives would defeat the bandwidth goal.
    """
    if env_output.error is not None:
        return f"Error: {env_output.error.exception_str}", "text/plain"

    messages = env_output.obs.to_llm_messages()
    text_parts: list[str] = []
    image_url: str | None = None

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            if content:
                text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text", "")
                    if text:
                        text_parts.append(text)
                elif block_type == "image_url":
                    data_url = block.get("image_url", {}).get("url", "")
                    if data_url.startswith("data:image/png;base64,"):
                        raw_b64 = data_url[len("data:image/png;base64,"):]
                        try:
                            png_bytes = base64.b64decode(raw_b64)
                        except Exception as e:
                            logger.warning("Failed to decode base64 image: %s", e)
                            continue
                    else:
                        # Non-standard format — attempt to use raw bytes
                        logger.warning(
                            "Unexpected image data URL format (expected data:image/png;base64,). "
                            "Treating as raw bytes."
                        )
                        png_bytes = data_url.encode()

                    filename = f"step_{step_index:04d}.png"
                    filepath = screenshot_dir / filename
                    try:
                        filepath.write_bytes(png_bytes)
                    except Exception as e:
                        logger.error("Failed to write screenshot %s: %s", filepath, e)
                        continue

                    # URL path: /screenshots/<session_id>/<filename>
                    image_url = f"{base_url}/screenshots/{session_id}/{filename}"
                    logger.debug("Screenshot saved: %s → %s", filepath, image_url)

    if image_url is not None:
        if text_parts:
            logger.debug(
                "Mixed observation (text + image) at step %d — returning image URL only. "
                "Text: %s",
                step_index,
                " ".join(text_parts)[:200],
            )
        return image_url, "image/png"

    return "\n".join(text_parts) if text_parts else "Success", "text/plain"


def build_action(call_id: str, name: str, arguments: dict) -> Any:
    """
    Construct a CUBE Action from tool call components.

    id=call_id propagates so CUBE sets tool_call_id on the resulting Observation content,
    allowing the agent to correctly link function_call_output to the right call_id.

    arguments must already be a dict (the agent calls json.loads before sending).
    """
    from cube.core import Action  # imported here to avoid hard dep at module load time

    return Action(id=call_id, name=name, arguments=arguments)


def extract_reward(env_output: Any) -> tuple[float, dict]:
    """
    Extract terminal reward and info dict from a CUBE EnvironmentOutput.

    task.step() calls task.evaluate() internally when done=True, so
    env_output.reward is always the correct terminal reward.
    No additional evaluate() call is needed.
    """
    return env_output.reward, env_output.info
