# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Originally Apple MIT License
#
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. The original Apple-authored portions of this
# file remain subject to the Apple license referenced above.

"""Agent role for any OpenAI-compatible chat completion endpoint.

The agent is configured at construction time via an :class:`OpenAIRoleConfig`
holding base_url, model, api_key and optional sampling parameters
(temperature, top_p, max_tokens, enable_thinking). Sampling fields default to
``None`` and are omitted from the request when unset so the server's own
defaults apply. ``enable_thinking`` toggles vLLM's
``chat_template_kwargs.enable_thinking`` via ``extra_body`` for reasoning
models that honour the kwarg (Qwen3, Nemotron, Gemma-thinking, …).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Optional

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

LOGGER = getLogger(__name__)

# Transient OpenAI-SDK errors we retry. Permanent client errors
# (AuthenticationError, BadRequestError, PermissionDeniedError, NotFoundError,
# UnprocessableEntityError, ConflictError) are intentionally NOT retried — they
# won't fix themselves and we shouldn't waste budget hammering the endpoint.
_RETRIABLE_OPENAI_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

openai_retry = retry(
    reraise=True,
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=1, max=8) + wait_random(0, 0.5),
    retry=retry_if_exception_type(_RETRIABLE_OPENAI_ERRORS),
    before_sleep=before_sleep_log(LOGGER, logging.WARNING),
)


@dataclass(frozen=True)
class OpenAIRoleConfig:
    """Configuration for an OpenAI-compatible role (agent or user simulator).

    ``api_key`` may be an empty string for endpoints that accept anonymous
    access; the SDK still requires a non-empty value, so a placeholder is
    substituted when building the client.

    Sampling fields default to ``None``; ``None`` means "do not send the
    parameter so the server-side default is used".
    """

    base_url: str
    model: str
    api_key: str = ""
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    enable_thinking: Optional[bool] = None


def _is_openai_reasoning_model(model: str) -> bool:
    """Heuristic: does the model accept top-level ``reasoning_effort``?

    Matches OpenAI reasoning families (o-series, gpt-5+) with or without an
    ``openai/`` provider prefix. Other model ids fall back to the vLLM-style
    ``chat_template_kwargs.enable_thinking`` mechanism.
    """
    m = model.lower().rsplit("/", maxsplit=1)[-1]
    if m.startswith(("o1", "o3", "o4", "o5")):
        return True
    if m.startswith("gpt-5") or m.startswith("gpt-6"):
        return True
    return False


def _sampling_kwargs(config: OpenAIRoleConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    is_openai = _is_openai_reasoning_model(config.model)
    if config.max_tokens is not None:
        if is_openai:
            kwargs["max_completion_tokens"] = config.max_tokens
        else:
            kwargs["max_tokens"] = config.max_tokens
    if config.enable_thinking is not None:
        if is_openai:
            # OpenAI/GPT-style: top-level reasoning_effort. OpenAI rejects
            # unknown body params, so do NOT send chat_template_kwargs here.
            kwargs["reasoning_effort"] = "high" if config.enable_thinking else "low"
        else:
            # vLLM/Qwen-style toggle via extra_body.
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": config.enable_thinking}
            }
    return kwargs
