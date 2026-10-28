# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Chat Completions client for the vision judge, on raw ``requests``.

Raw ``requests`` rather than a client library because the training image installs MimoAgent
without its declared dependencies, so nothing from that dependency tree can be assumed
present. ``requests`` is a hard dependency of both verl and MimoAgent and always is. Chat
Completions passes ``image_url`` content parts through natively, which is exactly what a
vision judge needs.

Used only by the evaluation grader (``eval_mode.py``). The training path never calls a judge
itself -- that happens inside the grading service.
"""

from __future__ import annotations

import os
import time
from typing import Any


class RequestsChatModel:
    """Minimal Chat Completions client implementing MimoAgent's ``Model`` protocol.

    ``query(messages, **kwargs) -> {"content", "tool_calls"?, "reasoning_content"?}`` plus
    ``get_template_vars() -> dict``.
    """

    _RETRIES = 3

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        temperature: float = 1.0,
        timeout: int = 120,
        **kwargs,  # absorb extra config keys without error
    ):
        raw_name = model_name or os.getenv("WEBDEV_EVAL_JUDGE_MODEL") or os.getenv("GRADER_MODEL")
        if not raw_name:
            raise ValueError("no judge model: pass model_name=, or set WEBDEV_EVAL_JUDGE_MODEL")
        self.model_name = raw_name.split("/", 1)[1] if raw_name.startswith("openai/") else raw_name
        base_url = base_url or os.getenv("WEBDEV_EVAL_JUDGE_BASE_URL") or os.getenv("LLM_JUDGE_BASE_URL")
        if not base_url:
            raise ValueError(
                "no judge base_url: pass base_url=, or set WEBDEV_EVAL_JUDGE_BASE_URL "
                "(any OpenAI-compatible /v1/chat/completions endpoint)"
            )
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.api_key = api_key or os.environ.get("LLM_JUDGE_API_KEY", "")
        self.temperature = temperature
        self.timeout = timeout

    def query(self, messages: list[dict], **kwargs) -> dict[str, Any]:
        import requests

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        for key in ("tools", "tool_choice"):
            if kwargs.get(key) is not None:
                payload[key] = kwargs[key]

        last_err: Exception | None = None
        for attempt in range(self._RETRIES):
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout,
                )
                if r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                r.raise_for_status()
                message = r.json()["choices"][0]["message"]
                break
            except Exception as e:  # noqa: BLE001 - every failure retries, then raises once
                last_err = e
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"judge call failed after {self._RETRIES} tries: {last_err}")

        out: dict[str, Any] = {"content": message.get("content")}
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {
                        "name": (tc.get("function") or {}).get("name"),
                        "arguments": (tc.get("function") or {}).get("arguments"),
                    },
                }
                for tc in tool_calls
                if isinstance(tc, dict)
            ]
        if message.get("reasoning_content"):
            out["reasoning_content"] = message["reasoning_content"]
        return out

    def get_template_vars(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def completion(self, messages: list[dict], **kwargs) -> str:
        """Simplified interface for the judge call: the raw content string."""
        result = self.query(messages, **kwargs)
        return result.get("content") or ""


__all__ = ["RequestsChatModel"]
