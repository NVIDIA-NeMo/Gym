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
import logging
import re
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI

from nemo_gym.base_responses_api_model import Body
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import request
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


logger = logging.getLogger(__name__)

_SENSITIVE_HEADER_RE = re.compile(r"('Authorization': ')[^']*(')", re.IGNORECASE)
_SENSITIVE_COOKIE_RE = re.compile(r"('(?:Set-)?Cookie': ')[^']*(')", re.IGNORECASE)


def _sanitize_error(e: Exception) -> str:
    """Strip sensitive headers (API keys, cookies) from error repr for safe logging."""
    msg = repr(e)
    msg = _SENSITIVE_HEADER_RE.sub(r"\1[REDACTED]\2", msg)
    msg = _SENSITIVE_COOKIE_RE.sub(r"\1[REDACTED]\2", msg)
    return msg


def _normalize_to_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a LiteLLM proxy response to the Responses API format.

    LiteLLM proxies have two known quirks when proxying /v1/responses:
    1. They may return ``reasoning.effort = "none"`` (string) instead of ``null``.
    2. They may downgrade the call to chat completions internally and return
       ``object="chat.completion"`` instead of ``"response"``.
    """
    # Fix fields that cause validation errors even in native response format.
    reasoning = data.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") == "none":
        reasoning["effort"] = None

    if data.get("object") != "chat.completion":
        return data

    logger.info("Normalizing chat.completion response to Responses API format")

    # LiteLLM proxies may return a hybrid format: object="chat.completion" but
    # content in either choices[] (standard) or output[] (Responses API style).
    text = ""

    # Try output[] first (LiteLLM hybrid format)
    for item in data.get("output", []):
        if isinstance(item, dict):
            for block in item.get("content", []):
                if isinstance(block, dict) and block.get("type") == "output_text":
                    text = block.get("text", "") or ""
                    if text:
                        break
        if text:
            break

    # Fall back to choices[] (standard chat completion format)
    if not text:
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            text = msg.get("content", "") or ""
            if text:
                break

    # Build a Responses API shaped dict
    usage = data.get("usage", {}) or {}
    return {
        "id": data.get("id", ""),
        "created_at": data.get("created", 0),
        "model": data.get("model", ""),
        "object": "response",
        "output": [
            {
                "id": f"msg_{data.get('id', '')[-16:]}",
                "content": [{"annotations": [], "text": text, "type": "output_text", "logprobs": None}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0) or usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


# Versions with credential-harvesting malware (supply chain incident, March 2026).
# https://docs.litellm.ai/blog/security-update-march-2026
_COMPROMISED_VERSIONS = frozenset({"1.82.7", "1.82.8"})
_MIN_SAFE_VERSION = "1.83.0"
_MIN_SAFE_VERSION_TUPLE = (1, 83, 0)
_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(.*)$")
_ALLOWED_VERSION_SUFFIX_RE = re.compile(r"^(?:|-stable(?:\.patch\.\d+)?|\.post\d+)$")


def _parse_litellm_version(version: str) -> tuple[int, int, int]:
    version = version.strip()
    match = _VERSION_RE.match(version)
    if not match:
        raise ValueError(f"Unsupported LiteLLM version format: {version!r}")

    suffix = match.group(4)
    if not _ALLOWED_VERSION_SUFFIX_RE.match(suffix):
        raise ValueError(f"Unsupported LiteLLM version suffix: {suffix!r}")

    return tuple(int(part) for part in match.groups()[:3])


class LiteLLMModelServerConfig(SimpleModelServerConfig):
    pass


class LiteLLMModelServer(SimpleModelServer):
    config: LiteLLMModelServerConfig

    async def _check_proxy_version(self) -> None:
        """Verify the LiteLLM proxy is running a known-safe version."""
        base_url = re.sub(r"/v1/?$", "", self.config.openai_base_url.rstrip("/"))
        health_url = f"{base_url}/health/readiness/details"
        auth_header = self.config.openai_api_key
        if not auth_header.lower().startswith("bearer "):
            auth_header = f"Bearer {auth_header}"

        try:
            resp = await request(
                "GET",
                health_url,
                _internal=True,
                headers={"Authorization": auth_header, "x-litellm-key": auth_header},
            )
            if not resp.ok:
                raise RuntimeError(f"status={resp.status}")
            data = await resp.json(content_type=None)
            version = data.get("litellm_version", "")
        except Exception as e:
            raise RuntimeError(f"Could not verify LiteLLM proxy version at {health_url}: {e}") from e

        if not version:
            raise RuntimeError(
                f"LiteLLM proxy at {health_url} did not report litellm_version. "
                f"Cannot enforce minimum safe version >= {_MIN_SAFE_VERSION}."
            )

        try:
            version_tuple = _parse_litellm_version(version)
        except ValueError as e:
            raise RuntimeError(
                f"LiteLLM proxy reported an unsupported version ({version!r}). "
                f"Cannot enforce minimum safe version >= {_MIN_SAFE_VERSION}."
            ) from e

        normalized_version = ".".join(str(part) for part in version_tuple)
        if normalized_version in _COMPROMISED_VERSIONS:
            raise RuntimeError(
                f"LiteLLM proxy is running a compromised version ({version}). "
                f"Versions {sorted(_COMPROMISED_VERSIONS)} contain credential-harvesting malware "
                f"(supply chain incident, March 2026 — see https://docs.litellm.ai/blog/security-update-march-2026). "
                f"Upgrade the proxy to >= {_MIN_SAFE_VERSION}."
            )

        if version_tuple < _MIN_SAFE_VERSION_TUPLE:
            raise RuntimeError(
                f"LiteLLM proxy version {version} is below the minimum safe version {_MIN_SAFE_VERSION}. "
                f"Upgrade the proxy before starting this model server."
            )

        logger.info("LiteLLM proxy version check passed (version=%s)", version or "unknown")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        main_lifespan = app.router.lifespan_context
        server = self

        @asynccontextmanager
        async def lifespan_with_version_check(app):
            await server._check_proxy_version()
            async with main_lifespan(app) as state:
                yield state

        app.router.lifespan_context = lifespan_with_version_check
        return app

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        try:
            openai_response_dict = await self._client.create_response(**body_dict)
        except Exception as e:
            logger.error("LiteLLM API call failed: %s", _sanitize_error(e))
            raise
        openai_response_dict = _normalize_to_response(openai_response_dict)
        return NeMoGymResponse.model_validate(openai_response_dict)


if __name__ == "__main__":
    LiteLLMModelServer.run_webserver()
