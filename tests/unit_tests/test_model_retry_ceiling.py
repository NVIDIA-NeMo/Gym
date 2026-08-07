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
from typing import List
from unittest.mock import AsyncMock, MagicMock

from pytest import MonkeyPatch, raises

import nemo_gym.openai_utils
from nemo_gym.openai_utils import MODEL_RETRY_MAX_ATTEMPTS, NeMoGymAsyncOpenAI


def _response(status: int, body: bytes = b"") -> MagicMock:
    response = MagicMock()
    response.status = status
    response.ok = 200 <= status < 400
    response.content.read = AsyncMock(return_value=body)
    return response


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _patch_loop(monkeypatch: MonkeyPatch, responses: List[MagicMock], clock: _Clock) -> AsyncMock:
    request_mock = AsyncMock(side_effect=responses)
    monkeypatch.setattr(nemo_gym.openai_utils, "request", request_mock)
    monkeypatch.setattr(nemo_gym.openai_utils, "sleep", AsyncMock())
    monkeypatch.setattr(nemo_gym.openai_utils, "monotonic", clock)
    return request_mock


class TestModelRetryCeiling:
    async def test_persistent_rate_limit_stops_at_the_attempt_ceiling(self, monkeypatch: MonkeyPatch) -> None:
        """429 raises the loop's own budget every time, so only the ceiling ends this."""
        clock = _Clock()
        request_mock = _patch_loop(monkeypatch, [_response(429, b"slow down") for _ in range(50)], clock)
        raise_mock = AsyncMock()
        monkeypatch.setattr(nemo_gym.openai_utils, "raise_for_status", raise_mock)

        client = NeMoGymAsyncOpenAI(base_url="http://endpoint/v1", api_key="k")
        await client._request_with_retry(method="POST", url="http://endpoint/v1/chat/completions")

        assert MODEL_RETRY_MAX_ATTEMPTS == request_mock.await_count
        raise_mock.assert_awaited_once()

    async def test_persistent_rate_limit_stops_on_the_deadline(self, monkeypatch: MonkeyPatch) -> None:
        clock = _Clock()
        request_mock = _patch_loop(monkeypatch, [_response(429, b"slow down") for _ in range(50)], clock)
        monkeypatch.setattr(nemo_gym.openai_utils, "raise_for_status", AsyncMock())

        async def advancing_sleep(_seconds: float) -> None:
            clock.now += 200.0

        monkeypatch.setattr(nemo_gym.openai_utils, "sleep", advancing_sleep)

        client = NeMoGymAsyncOpenAI(base_url="http://endpoint/v1", api_key="k")
        await client._request_with_retry(method="POST", url="http://endpoint/v1/chat/completions")

        # The deadline fires well before the attempt ceiling would.
        assert request_mock.await_count < MODEL_RETRY_MAX_ATTEMPTS

    async def test_transient_rate_limit_then_success_does_not_spend_the_budget(self, monkeypatch: MonkeyPatch) -> None:
        clock = _Clock()
        request_mock = _patch_loop(
            monkeypatch, [_response(429, b"slow"), _response(429, b"slow"), _response(200)], clock
        )

        client = NeMoGymAsyncOpenAI(base_url="http://endpoint/v1", api_key="k")
        response = await client._request_with_retry(method="POST", url="http://endpoint/v1/chat/completions")

        assert 200 == response.status
        assert 3 == request_mock.await_count

    async def test_non_retryable_status_returns_immediately(self, monkeypatch: MonkeyPatch) -> None:
        clock = _Clock()
        request_mock = _patch_loop(monkeypatch, [_response(400, b"bad request")], clock)

        client = NeMoGymAsyncOpenAI(base_url="http://endpoint/v1", api_key="k")
        response = await client._request_with_retry(method="POST", url="http://endpoint/v1/chat/completions")

        assert 400 == response.status
        assert 1 == request_mock.await_count

    async def test_the_final_error_carries_the_endpoint_body(self, monkeypatch: MonkeyPatch) -> None:
        """`response.content` yields nothing on a second read, so the body has to be passed through."""
        clock = _Clock()
        _patch_loop(monkeypatch, [_response(500, b"upstream detail") for _ in range(50)], clock)

        captured = {}

        async def raise_for_status(response, response_content=None):
            captured["content"] = response_content
            raise RuntimeError("gave up")

        monkeypatch.setattr(nemo_gym.openai_utils, "raise_for_status", raise_for_status)

        client = NeMoGymAsyncOpenAI(base_url="http://endpoint/v1", api_key="k")
        with raises(RuntimeError):
            await client._request_with_retry(method="POST", url="http://endpoint/v1/chat/completions")

        assert "upstream detail" == captured["content"]
