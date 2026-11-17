# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""The capture key the policy admission middleware reads from a generation path."""

import pytest

from nemo_gym.models._checkpoint import PolicyAdmissionMiddleware


TRACEPARENT = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"
NOT_GATED = "not gated"


class _RecordingGate:
    """Records the key each call is admitted and entered with."""

    def __init__(self) -> None:
        self.admitted: list[str | None] = []
        self.entered: list[str] = []

    async def admit(self, capture_key: str | None) -> None:
        self.admitted.append(capture_key)

    def enter(self, capture_key: str):
        self.entered.append(capture_key)
        raise _Entered


class _Entered(Exception):
    pass


async def _admitted_key(path: str) -> object:
    gate = _RecordingGate()
    forwarded: list[str] = []

    async def app(scope, receive, send):
        forwarded.append(scope["path"])

    try:
        await PolicyAdmissionMiddleware(app, gate)({"type": "http", "path": path}, None, None)
    except _Entered:
        assert gate.entered == [gate.admitted[0] or ""]
        return gate.admitted[0]
    assert forwarded == [path] and gate.admitted == []
    return NOT_GATED


@pytest.mark.parametrize(
    "path, capture_key",
    [
        ("/v1/responses", None),
        ("/ng-rollout/r-a1/v1/responses", "r-a1"),
        ("/ng-rollout/r-a1/v1/chat/completions", "r-a1"),
        ("/ng-rollout/r-a1/v1/messages", "r-a1"),
        ("/ng-rollout/r-a1/training-token-capture/v1/responses", "r-a1"),
        # The capture key is read before an agent instance name, a training tag and a trace segment.
        ("/ng-rollout/r-a1/ng-agent-instance/judge/v1/responses", "r-a1"),
        ("/ng-rollout/r-a1/ng-agent-instance/judge.1/training-token-capture/v1/responses", "r-a1"),
        (f"/ng-rollout/r-a1/ng-trace/{TRACEPARENT}/v1/responses", "r-a1"),
        (
            f"/ng-rollout/r-a1/ng-agent-instance/judge/training-token-capture/ng-trace/{TRACEPARENT}/v1/responses",
            "r-a1",
        ),
        (f"/ng-trace/{TRACEPARENT}/v1/responses", None),
        ("/ng-rollout//v1/responses", None),
        ("/ng-other/x/v1/responses", None),
        # Only generation routes are gated.
        ("/ng-rollout/r-a1/v1/models", NOT_GATED),
        ("/ng-rollout/r-a1/verify", NOT_GATED),
        ("/health", NOT_GATED),
    ],
)
async def test_the_capture_key_a_generation_call_is_admitted_with(path, capture_key):
    assert await _admitted_key(path) == capture_key
