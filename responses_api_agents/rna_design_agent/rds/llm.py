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
"""The controller's link to a model. Nothing above this file knows a URL, wire format or vendor.

Stdlib only: the gateway speaks the OpenAI chat format, which is a POST and a JSON body.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


#: No default endpoint or model. Set RDS_LLM_BASE_URL and RDS_LLM_MODEL, or pass them to the provider.
DEFAULT_BASE_URL = ""
DEFAULT_MODEL = ""

#: Checked in order. The first that yields a key wins.
KEY_ENV_VARS = ("RDS_LLM_API_KEY", "NVIDIA_API_KEY", "NGC_API_KEY", "OPENAI_API_KEY")
KEY_FILES = ("inferencekey.txt",)


@dataclass(frozen=True)
class Usage:
    """What a call cost. Reported by the API, never estimated."""

    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.latency_s + other.latency_s,
        )


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    #: Why the model's arguments could not be read. The call then fails as the model's own mistake.
    error: str = ""


@dataclass(frozen=True)
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""
    #: Reasoning some models return apart from the answer, kept for the trace.
    reasoning: str = ""


class Provider(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 4096) -> Reply: ...


def find_api_key(search_dir: str | Path = ".") -> str | None:
    for var in KEY_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    for name in KEY_FILES:
        path = Path(search_dir) / name
        if path.is_file():
            value = path.read_text().strip()
            if value:
                return value
    return None


#: Words a gateway uses when a 401 is its own trouble, not the key.
TRANSIENT_AUTH = ("connector", "too many clients", "database", "timeout")


def _as_openai_tools(tools: list[dict]) -> list[dict]:
    """The registry's tool dicts in the chat API's shape."""
    return [
        {
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
        }
        for t in tools
    ]


class GatewayProvider:
    """Any OpenAI-compatible chat endpoint. The endpoint and model are configuration, never defaults."""

    name = "gateway"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 300.0,
        max_retries: int = 0,
    ) -> None:
        self.model = model or os.environ.get("RDS_LLM_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or os.environ.get("RDS_LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or find_api_key()
        self.timeout_s = timeout_s
        # A gateway can be away for minutes, and a run is a long sequence of calls, so one unlucky
        # minute would end hours of work. Eight attempts with the backoff below span about three minutes.
        self.max_retries = max_retries or int(os.environ.get("RDS_LLM_MAX_RETRIES", "8"))
        if not self.base_url:
            raise ValueError("no endpoint: set RDS_LLM_BASE_URL or pass base_url (any OpenAI-compatible /v1 URL)")
        if not self.model:
            raise ValueError("no model: set RDS_LLM_MODEL or pass model")
        if not self.api_key:
            raise RuntimeError(
                "no API key: set one of "
                + ", ".join(KEY_ENV_VARS)
                + f" or place {KEY_FILES[0]} in the working directory"
            )

    def chat(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 4096) -> Reply:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if tools:
            body["tools"] = _as_openai_tools(tools)
        payload = json.dumps(body).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        started = time.monotonic()
        data = self._send(request)
        elapsed = time.monotonic() - started
        return self._parse(data, elapsed)

    def _send(self, request: urllib.request.Request) -> dict:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:400]
                # Other 4xx will not improve by waiting, except a 401 for the gateway's own trouble
                # (TRANSIENT_AUTH). A wrong key reads differently.
                lowered = detail.lower()
                transient = exc.code in (401, 429) and (
                    exc.code == 429 or any(mark in lowered for mark in TRANSIENT_AUTH)
                )
                if not transient and exc.code < 500:
                    raise RuntimeError(f"gateway HTTP {exc.code}: {detail}") from exc
                last = RuntimeError(f"gateway HTTP {exc.code}: {detail}")
            except (urllib.error.URLError, TimeoutError) as exc:
                last = RuntimeError(f"gateway unreachable: {exc}")
            if attempt == self.max_retries - 1:
                break
            # Jittered because sessions run in parallel: without it they all come back at the same
            # instant and hand the gateway the same crowd it just turned away.
            time.sleep(min(2**attempt, 60.0) * (0.5 + random.random()))
        raise last or RuntimeError("gateway call failed")

    @staticmethod
    def _parse(data: dict, elapsed: float) -> Reply:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            raw = function.get("arguments") or "{}"
            arguments, error = {}, ""
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError as exc:
                error = f"the arguments were not valid JSON: {exc}"
            calls.append(ToolCall(call.get("id", ""), function.get("name", ""), arguments, error))
        text = message.get("content") or ""
        if not calls:
            calls = _tool_calls_in_text(text)
        usage_data = data.get("usage") or {}
        return Reply(
            text=text,
            tool_calls=calls,
            usage=Usage(usage_data.get("prompt_tokens", 0), usage_data.get("completion_tokens", 0), elapsed),
            finish_reason=choice.get("finish_reason", ""),
            reasoning=message.get("reasoning_content") or "",
        )


TEXT_TOOL_CALL = re.compile(r"<TOOLCALL>\s*(\[.*?\])\s*</TOOLCALL>", re.S)


def _tool_calls_in_text(text: str) -> list[ToolCall]:
    """Recover tool calls a model wrote into its message instead of the field.

    Some models emit a `<TOOLCALL>[{"name": ..., "arguments": {...}}]</TOOLCALL>` block in the content.
    """
    match = TEXT_TOOL_CALL.search(text or "")
    if not match:
        return []
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    calls = []
    for index, item in enumerate(payload if isinstance(payload, list) else [payload]):
        name = (item or {}).get("name")
        if not name:
            continue
        arguments = (item or {}).get("arguments") or (item or {}).get("parameters") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        calls.append(ToolCall(id=f"text-{index}", name=name, arguments=dict(arguments)))
    return calls
