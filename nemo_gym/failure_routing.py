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
"""Shared construction helpers for rollout failure rows."""

from collections.abc import Mapping
from typing import Any


NG_FAILURE_CLASS_KEY = "_ng_failure_class"
NG_NO_PERSIST_KEY = "_ng_no_persist"
NG_TERMINAL_KEY = "_ng_failure_terminal"

_FAILURE_RESULT_RESERVED_KEYS = (
    "reward",
    "response",
    "error",
    NG_FAILURE_CLASS_KEY,
    NG_NO_PERSIST_KEY,
    NG_TERMINAL_KEY,
)


def minimal_failure_response(
    response_id: str = "failure",
    *,
    model: str = "unknown",
    message_id: str = "msg_0",
    text: str = "",
    created_at: float = 0.0,
    tool_choice: str = "auto",
) -> dict[str, Any]:
    """Return a minimal serialized Responses object for a rollout with no trajectory."""
    return {
        "id": response_id,
        "created_at": created_at,
        "model": model,
        "object": "response",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "id": message_id,
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tools": [],
        "tool_choice": tool_choice,
    }


def build_failure_result(
    record: Mapping[str, Any],
    *,
    failure_class: str,
    error: Any,
    response: Mapping[str, Any] | None = None,
    terminal: bool = False,
    no_persist: bool = False,
    error_key: str = "error",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a fresh reward-0 failure row without mutating ``record``.

    Input rows may be recycled from a rollouts or failures JSONL and therefore
    carry stale result and routing fields. Those fields are removed before the
    new failure contract is applied. Canonical fields win over ``extra`` so a
    caller cannot accidentally replace the fresh routing state.
    """
    if error_key in _FAILURE_RESULT_RESERVED_KEYS and error_key != "error":
        raise ValueError(f"error_key conflicts with a reserved failure-result key: {error_key!r}")

    result = dict(record)
    if extra is not None:
        result.update(extra)

    for key in (*_FAILURE_RESULT_RESERVED_KEYS, error_key):
        result.pop(key, None)

    if response is None:
        response = minimal_failure_response()

    result.update(
        {
            "reward": 0.0,
            "response": dict(response),
            error_key: error,
            NG_FAILURE_CLASS_KEY: failure_class,
        }
    )
    if terminal:
        result[NG_TERMINAL_KEY] = True
    if no_persist:
        result[NG_NO_PERSIST_KEY] = True
    return result
