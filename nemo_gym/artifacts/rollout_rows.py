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
"""The rollout rows a run writes: the private keys that mark a row, and the row as it is exported."""

from typing import Any

from nemo_gym.trajectory.trajectory_record import (
    NG_TRAJECTORY_KEY,
    NG_VERIFIER_INVOCATIONS_KEY,
)


NG_FAILURE_CLASS_KEY = "_ng_failure_class"
NG_NO_PERSIST_KEY = "_ng_no_persist"
# Set only by Gym, on the result it builds for a row the dispatch budget never started.
NG_DISPATCH_DRAINED_KEY = "_ng_dispatch_drained"
_MODEL_CALL_PAYLOAD_KEYS = ("request", "response", "request_raw", "response_raw")


def _strip_capture_payloads(result: dict[str, Any]) -> None:
    result.pop(NG_VERIFIER_INVOCATIONS_KEY, None)
    capture = result.get("ng_model_call_capture")
    calls = capture.get("calls") if isinstance(capture, dict) else None
    if isinstance(capture, dict):
        capture.pop("verifier_invocations", None)
    for call in calls if isinstance(calls, list) else []:
        if isinstance(call, dict):
            for key in _MODEL_CALL_PAYLOAD_KEYS:
                call.pop(key, None)


def _rollout_for_export(result: dict[str, Any]) -> dict[str, Any]:
    """Return an exporter view without the complete trajectory or raw capture payloads."""
    sanitized = dict(result)
    sanitized.pop(NG_TRAJECTORY_KEY, None)
    sanitized.pop("ng_model_call_capture", None)
    capture = result.get("ng_model_call_capture")
    if isinstance(capture, dict):
        sanitized_capture = dict(capture)
        calls = capture.get("calls")
        if isinstance(calls, list):
            sanitized_capture["calls"] = [
                {key: value for key, value in call.items() if key not in _MODEL_CALL_PAYLOAD_KEYS}
                for call in calls
                if isinstance(call, dict)
            ]
        else:
            sanitized_capture.pop("calls", None)
        sanitized["ng_model_call_capture"] = sanitized_capture
    return sanitized
