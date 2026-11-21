# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve accepted output at the pinned Hermes compression-disabled length exit.

Compatibility target: https://github.com/cmunley1/hermes-agent/blob/
26bb847a88493342ca1b194e0455b479073ae21d/run_agent.py (MIT, Nous Research).
The native method remains responsible for execution and outcome classification.
Its early length return omits the just-accepted message from its transcript;
this adapter retains that real provider reply and uses Hermes' ordinary builder.

Upstream notice for this adaptation:
Copyright (c) 2025 Nous Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of
the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable


if TYPE_CHECKING:
    from run_agent import AIAgent


HERMES_COMMIT = "26bb847a88493342ca1b194e0455b479073ae21d"
RUN_AGENT_SHA256 = "48a8013315223e9656b7c0df3d52bf688f36b9828fc388c8c63ac5367302b2b3"
_SOURCE_URL = "https://github.com/cmunley1/hermes-agent"
_LENGTH_ERROR = "Response truncated due to output length limit"
_installed: dict[type[AIAgent], Callable[..., dict[str, Any]]] = {}


def _require_pinned_source(agent_class: type[AIAgent]) -> None:
    source = inspect.getsourcefile(agent_class)
    if source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != RUN_AGENT_SHA256:
        raise RuntimeError("Hermes transcript compatibility requires the exact declared 26bb847a source")
    distribution = importlib.metadata.distribution("hermes-agent")
    provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
    url = str(provenance.get("url") or "").removesuffix(".git")
    vcs_commit = (provenance.get("vcs_info") or {}).get("commit_id")
    if not (
        (url == _SOURCE_URL and vcs_commit == HERMES_COMMIT) or url == f"{_SOURCE_URL}/archive/{HERMES_COMMIT}.tar.gz"
    ):
        raise RuntimeError("Hermes transcript compatibility requires the declared Git source pin")


def install_length_transcript_compat(agent_class: type[AIAgent]) -> None:
    """Wrap the genuine native entrypoint; preserve ordinary replies and error policy."""
    _require_pinned_source(agent_class)
    if agent_class in _installed:
        if agent_class.run_conversation is not _installed[agent_class]:
            raise RuntimeError("Hermes conversation entrypoint changed after compatibility installation")
        return
    original = agent_class.run_conversation
    if (
        original.__module__ != "run_agent"
        or original.__name__ != "run_conversation"
        or original.__code__.co_firstlineno != 6392
    ):
        raise RuntimeError("Hermes conversation entrypoint is not the pinned native implementation")

    @functools.wraps(original)
    def run_conversation(self: AIAgent, *args: Any, **kwargs: Any) -> dict[str, Any]:
        accepted_response = None
        # The pinned non-streaming loop accepts the response returned by this
        # method; retain its actual reply without changing retries or clients.
        original_call = self._interruptible_api_call
        had_local_call = "_interruptible_api_call" in self.__dict__
        local_call = self.__dict__.get("_interruptible_api_call")

        @functools.wraps(original_call)
        def capture_api_call(*call_args: Any, **call_kwargs: Any) -> Any:
            nonlocal accepted_response
            response = original_call(*call_args, **call_kwargs)
            accepted_response = response
            return response

        self._interruptible_api_call = capture_api_call
        try:
            result = original(self, *args, **kwargs)
        finally:
            if had_local_call:
                self._interruptible_api_call = local_call
            else:
                del self._interruptible_api_call
        if (
            self.api_mode == "chat_completions"
            and not self.compression_enabled
            and isinstance(result, dict)
            and result.get("partial") is True
            and result.get("error") == _LENGTH_ERROR
        ):
            choices = getattr(accepted_response, "choices", None)
            if not choices or choices[0].finish_reason != "length":
                raise RuntimeError("Hermes length exit has no matching accepted native response")
            messages = result.get("messages")
            if not isinstance(messages, list):
                raise RuntimeError("Hermes length exit returned no native conversation")
            messages.append(self._build_assistant_message(choices[0].message, "length"))
        return result

    agent_class.run_conversation = run_conversation
    _installed[agent_class] = run_conversation
