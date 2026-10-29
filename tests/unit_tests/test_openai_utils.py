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
"""Tests of the OpenAI-compatible client in `nemo_gym.openai_utils`."""

import re
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest
from aiohttp import ClientResponseError, ClientTimeout
from pydantic import ValidationError

from nemo_gym import openai_utils as openai_utils_module
from nemo_gym.core import rollout_context
from nemo_gym.openai_utils import (
    MAX_NUM_TRIES,
    NeMoGymAsyncOpenAI,
    PermanentEndpointError,
    _error_body_is_permanent_auth,
    _error_body_is_permanent_quota,
)


@pytest.fixture(autouse=True)
def _jitter_at_its_upper_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nemo_gym.openai_utils.random.uniform", lambda low, high: high)


class TestOpenAIUtils:
    def test_invalid_retry_configuration_rejected(self):
        with pytest.raises(ValidationError):
            NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.com/v1", max_http_attempts=0)

    async def test_NeMoGymAsyncOpenAI(self) -> None:
        NeMoGymAsyncOpenAI(api_key="abc", base_url="https://api.openai.com/v1")

    @pytest.mark.parametrize("debug_enabled", [True, False])
    async def test_failed_response_logs_request_only_when_request_debug_is_enabled(
        self, monkeypatch, capsys, debug_enabled: bool
    ) -> None:
        # set_global_aiohttp_client sets the flag on server_utils at startup, after openai_utils was imported.
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", debug_enabled)
        monkeypatch.setattr(openai_utils_module, "raise_for_status", AsyncMock())
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.com/v1")
        request_kwargs = {"url": "https://example.com/v1/chat/completions", "json": {"model": "m"}}

        await client._raise_for_status(SimpleNamespace(ok=False), request_kwargs)

        logged = "Request kwargs:" in capsys.readouterr().out
        assert logged is debug_enabled

    async def test_create_embeddings_posts_to_embeddings_url(self, monkeypatch) -> None:
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.com/v1")
        response = SimpleNamespace(ok=True)
        request = AsyncMock(return_value=response)
        monkeypatch.setattr(client, "_request", request)
        monkeypatch.setattr(client, "_raise_for_status", AsyncMock())
        monkeypatch.setattr(openai_utils_module, "get_response_json", AsyncMock(return_value={"data": []}))

        result = await client.create_embeddings(model="m", input="hi")

        assert result == {"data": []}
        request.assert_awaited_once_with(
            method="POST", url="https://example.com/v1/embeddings", json={"model": "m", "input": "hi"}
        )

    async def test_explicit_attempt_cap_reaches_low_level_transport(
        self,
        monkeypatch,
    ) -> None:
        captured = {}
        response = SimpleNamespace(status=200)

        async def fake_request(**kwargs):
            captured.update(kwargs)
            return response

        monkeypatch.setattr(openai_utils_module, "request", fake_request)
        client = NeMoGymAsyncOpenAI(
            api_key="abc",
            base_url="https://api.openai.com/v1",
            max_num_tries=1,
            max_connection_retries=2,
            request_timeout_seconds=300,
            connect_timeout_seconds=60,
        )

        assert await client._request(method="POST", url="https://example.test") is response
        assert captured["_max_num_tries"] == 1
        assert captured["_max_connection_retries"] == 2
        assert isinstance(captured["timeout"], ClientTimeout)
        assert captured["timeout"].total == 300
        assert captured["timeout"].connect == 60

    async def test_request_id_names_the_rollout_inside_a_rollout_context(self, monkeypatch) -> None:
        captured = {}

        async def fake_request(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(status=200)

        monkeypatch.setattr(openai_utils_module, "request", fake_request)
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://api.openai.com/v1")
        with rollout_context("3-0"):
            await client._request(method="POST", url="https://example.test")
        first = captured["headers"]["X-Request-Id"]
        assert re.fullmatch(r"3-0-[0-9a-f]{8}", first)
        with rollout_context("3-0"):
            await client._request(method="POST", url="https://example.test")
        assert captured["headers"]["X-Request-Id"] != first

    async def test_request_id_is_absent_outside_a_rollout_and_never_overrides_the_callers(self, monkeypatch) -> None:
        captured = {}

        async def fake_request(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(status=200)

        monkeypatch.setattr(openai_utils_module, "request", fake_request)
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://api.openai.com/v1")
        await client._request(method="POST", url="https://example.test")
        assert "X-Request-Id" not in captured["headers"]
        with rollout_context("3-0"):
            await client._request(method="POST", url="https://example.test", headers={"x-request-id": "mine"})
        assert captured["headers"]["x-request-id"] == "mine"
        assert "X-Request-Id" not in captured["headers"]

    @pytest.mark.parametrize("internal", [False, True])
    @pytest.mark.parametrize("status", [429, 500, 503, 504])
    @pytest.mark.parametrize("captured", [False, True])
    async def test_explicit_attempt_cap_has_no_exhausted_status_sleep(
        self,
        monkeypatch,
        internal,
        status,
        captured,
    ) -> None:
        sleeps = []
        calls = 0

        class Body:
            async def read(self):
                return b"rate limited"

        response = SimpleNamespace(status=status, content=Body())

        async def fake_request(**kwargs):
            nonlocal calls
            del kwargs
            calls += 1
            return response

        async def fake_raise_for_status(actual_response, content=None):
            assert actual_response is response
            raise RuntimeError("rate limited")

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr(openai_utils_module, "request", fake_request)
        monkeypatch.setattr(
            openai_utils_module,
            "raise_for_status",
            fake_raise_for_status,
        )
        monkeypatch.setattr(openai_utils_module, "sleep", fake_sleep)
        client = NeMoGymAsyncOpenAI(
            api_key="abc",
            base_url="https://api.openai.com/v1",
            max_num_tries=None if captured else 1,
            internal=internal,
        )

        with pytest.raises(RuntimeError, match="rate limited"):
            await client._request(method="POST", url="https://example.test", _resend_after_disconnect=not captured)

        assert calls == 1
        assert sleeps == []

    def test_explicit_attempt_cap_only_supports_disabling_inner_retries(self) -> None:
        with pytest.raises(ValidationError, match="Input should be 1"):
            NeMoGymAsyncOpenAI(
                api_key="abc",
                base_url="https://api.openai.com/v1",
                max_num_tries=2,
            )

    def test_connect_timeout_requires_request_timeout(self) -> None:
        with pytest.raises(
            ValidationError,
            match="connect_timeout_seconds requires request_timeout_seconds",
        ):
            NeMoGymAsyncOpenAI(
                api_key="abc",
                base_url="https://api.openai.com/v1",
                connect_timeout_seconds=60,
            )

    async def test_external_endpoint_retries_are_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = SimpleNamespace(status=504, content=SimpleNamespace(read=AsyncMock(return_value=b"")))
        request = AsyncMock(side_effect=[response] * MAX_NUM_TRIES + [SimpleNamespace(status=200)])
        monkeypatch.setattr("nemo_gym.openai_utils.request", request)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", AsyncMock())

        async def raise_status(response, content=None):
            raise RuntimeError("bounded")

        monkeypatch.setattr("nemo_gym.openai_utils.raise_for_status", raise_status)

        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.com/v1")
        with pytest.raises(RuntimeError):
            await client._request_with_retry()

        assert request.await_count == MAX_NUM_TRIES

    @pytest.mark.parametrize("status", [404, 408])
    async def test_retry_reuses_request_with_growing_delay(self, monkeypatch, status):
        failure = SimpleNamespace(status=status, content=SimpleNamespace(read=AsyncMock(return_value=b"temporary")))
        success = SimpleNamespace(status=200)
        request = AsyncMock(side_effect=[failure, failure, success])
        sleep = AsyncMock()
        monkeypatch.setattr("nemo_gym.openai_utils.request", request)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", sleep)
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.com/v1")
        payload = {"model": "judge", "input": [{"role": "user", "content": "preserved answer"}]}
        original = deepcopy(payload)

        result = await client._request_with_retry(method="POST", url="https://example.com/v1/responses", json=payload)

        assert result is success
        assert request.await_count == 3
        assert all(c.kwargs["json"] == original for c in request.await_args_list)
        assert payload == original
        assert sleep.await_args_list == [call(1.0), call(2.0)]

    async def test_non_retryable_http_errors_are_returned_once(self, monkeypatch):
        response = SimpleNamespace(status=400)
        request = AsyncMock(return_value=response)
        sleep = AsyncMock()
        monkeypatch.setattr("nemo_gym.openai_utils.request", request)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", sleep)
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.com/v1")

        assert await client._request_with_retry() is response
        request.assert_awaited_once()
        sleep.assert_not_awaited()

    @pytest.mark.parametrize("status", [404, 408])
    @pytest.mark.parametrize("attempts", [1, 3, 5])
    @pytest.mark.parametrize("internal", [False, True])
    async def test_configured_attempt_limit_preserves_terminal_error(self, monkeypatch, status, attempts, internal):
        replies = [
            SimpleNamespace(status=status, content=SimpleNamespace(read=AsyncMock(return_value=b"error body")))
            for _ in range(attempts)
        ]
        request = AsyncMock(side_effect=replies)
        sleep = AsyncMock()
        monkeypatch.setattr("nemo_gym.openai_utils.request", request)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", sleep)

        async def raise_status(response, content=None):
            assert response is replies[-1]
            assert content == b"error body"
            raise RuntimeError("terminal error")

        monkeypatch.setattr("nemo_gym.openai_utils.raise_for_status", raise_status)
        client = NeMoGymAsyncOpenAI(
            api_key="abc",
            base_url="https://example.com/v1",
            max_http_attempts=attempts,
            internal=internal,
        )
        with pytest.raises(RuntimeError, match="terminal error"):
            await client._request_with_retry()
        assert request.await_count == attempts
        assert sleep.await_args_list == [call(2.0**i) for i in range(attempts - 1)]

    @pytest.mark.parametrize(
        "body,expected",
        [
            (b'{"error":{"code":"budget_exceeded"}}', True),
            (b'{"error":{"type":"insufficient_quota"}}', True),
            (b'{"error":{"code":"rate_limit_exceeded"}}', False),
            (b'{"detail":{"error":{"type":"insufficient_quota"}}}', True),
            (b'{"detail":{"error":{"code":"rate_limit_exceeded"}}}', False),
            (b"Quota exceeded for quota metric requests per minute. Retry in 30 seconds.", False),
            (b'{"error":{"message":"quota exceeded"}}', False),
            (b"budget_exceeded", False),
        ],
    )
    def test_permanent_quota_classifier(self, body, expected):
        assert _error_body_is_permanent_quota(body) is expected

    @pytest.mark.parametrize(
        "body,expected",
        [
            (b'{"error":{"code":"invalid_api_key"}}', True),
            (b'{"error":{"type":"authentication_error"}}', True),
            (b'{"detail":{"error":{"code":"invalid_api_key"}}}', True),
            (b"Incorrect API key provided", True),
            (b'{"error":"unauthorized"}', False),
            (b'{"error":{"message":"model not available"}}', False),
        ],
    )
    def test_permanent_auth_classifier(self, body, expected):
        assert _error_body_is_permanent_auth(body) is expected

    @pytest.mark.parametrize(
        "status,body",
        [
            (429, b'{"error":{"code":"budget_exceeded"}}'),
            (429, b'{"error":{"type":"insufficient_quota"}}'),
            (401, b'{"error":{"code":"invalid_api_key"}}'),
            (403, b'{"error":{"type":"authentication_error"}}'),
        ],
    )
    async def test_permanent_quota_and_auth_errors_fail_fast(self, monkeypatch, status, body):
        response = SimpleNamespace(status=status, content=SimpleNamespace(read=AsyncMock(return_value=body)))
        request = AsyncMock(return_value=response)
        sleep = AsyncMock()
        monkeypatch.setattr("nemo_gym.openai_utils.request", request)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", sleep)
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.com/v1")

        with pytest.raises(PermanentEndpointError) as first:
            await client._request_with_retry(url="https://example.com/v1/responses")
        with pytest.raises(PermanentEndpointError) as second:
            await client._request_with_retry(url="https://example.com/v1/responses")

        assert first.value is not second.value
        assert first.value.status == status
        assert first.value.response_content == body
        request.assert_awaited_once()
        sleep.assert_not_awaited()

    async def test_generic_quota_exceeded_429_still_retries(self, monkeypatch):
        failure = SimpleNamespace(
            status=429,
            content=SimpleNamespace(
                read=AsyncMock(return_value=b"Quota exceeded for quota metric requests per minute.")
            ),
        )
        success = SimpleNamespace(status=200)
        request = AsyncMock(side_effect=[failure, success])
        sleep = AsyncMock()
        monkeypatch.setattr("nemo_gym.openai_utils.request", request)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", sleep)
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.com/v1")

        result = await client._request_with_retry(url="https://example.com/v1/responses")

        assert result is success
        assert request.await_count == 2
        sleep.assert_awaited_once_with(1.0)

    async def test_transient_429_still_retries(self, monkeypatch):
        failure = SimpleNamespace(
            status=429,
            content=SimpleNamespace(read=AsyncMock(return_value=b'{"error":"rate_limit_exceeded"}')),
        )
        success = SimpleNamespace(status=200)
        request = AsyncMock(side_effect=[failure, success])
        sleep = AsyncMock()
        monkeypatch.setattr("nemo_gym.openai_utils.request", request)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", sleep)
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.com/v1")

        result = await client._request_with_retry(url="https://example.com/v1/responses")

        assert result is success
        assert request.await_count == 2
        sleep.assert_awaited_once_with(1.0)

    async def test_exhausted_transient_429_raises_without_tripping(self, monkeypatch):
        replies = [
            SimpleNamespace(
                status=429,
                content=SimpleNamespace(read=AsyncMock(return_value=b'{"error":"rate_limit_exceeded"}')),
            )
            for _ in range(MAX_NUM_TRIES)
        ]
        request = AsyncMock(side_effect=replies)
        sleep = AsyncMock()
        monkeypatch.setattr("nemo_gym.openai_utils.request", request)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", sleep)
        seen = {}

        async def raise_status(response, content=None):
            seen["response"] = response
            seen["content"] = content
            raise RuntimeError("terminal 429")

        monkeypatch.setattr("nemo_gym.openai_utils.raise_for_status", raise_status)
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.com/v1", max_rate_limit_wait_seconds=0)

        with pytest.raises(RuntimeError, match="terminal 429"):
            await client._request_with_retry(url="https://example.com/v1/responses")
        assert request.await_count == MAX_NUM_TRIES
        assert seen["content"] == b'{"error":"rate_limit_exceeded"}'
        assert sleep.await_count == MAX_NUM_TRIES - 1

        later = SimpleNamespace(status=200)
        request.side_effect = [later]
        assert await client._request_with_retry(url="https://example.com/v1/responses") is later

    async def test_auth_without_key_marker_is_returned_once(self, monkeypatch):
        response = SimpleNamespace(
            status=401,
            content=SimpleNamespace(read=AsyncMock(return_value=b'{"error":"unauthorized"}')),
        )
        request = AsyncMock(return_value=response)
        sleep = AsyncMock()
        monkeypatch.setattr("nemo_gym.openai_utils.request", request)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", sleep)
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.com/v1")

        assert await client._request_with_retry() is response
        assert await client._request_with_retry() is response
        assert request.await_count == 2
        sleep.assert_not_awaited()

    async def test_request_skips_the_wire_after_permanent_trip(self, monkeypatch):
        response = SimpleNamespace(
            status=429,
            content=SimpleNamespace(read=AsyncMock(return_value=b'{"error":{"code":"budget_exceeded"}}')),
        )
        request = AsyncMock(return_value=response)
        monkeypatch.setattr("nemo_gym.openai_utils.request", request)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", AsyncMock())
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.com/v1")

        with pytest.raises(PermanentEndpointError):
            await client._request(method="POST", url="https://example.com/v1/responses")
        with pytest.raises(PermanentEndpointError):
            await client._request(method="POST", url="https://example.com/v1/responses")

        request.assert_awaited_once()

    async def test_in_flight_retry_stops_after_sibling_trips(self, monkeypatch):
        import asyncio

        rate_limited = SimpleNamespace(
            status=429,
            content=SimpleNamespace(read=AsyncMock(return_value=b'{"error":"rate_limit_exceeded"}')),
        )
        spent = SimpleNamespace(
            status=429,
            content=SimpleNamespace(read=AsyncMock(return_value=b'{"error":{"code":"budget_exceeded"}}')),
        )
        gate = asyncio.Event()
        calls = 0

        async def fake_request(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return rate_limited
            return spent

        async def fake_sleep(_delay):
            await gate.wait()

        monkeypatch.setattr("nemo_gym.openai_utils.request", fake_request)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", fake_sleep)
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.com/v1")

        first = asyncio.create_task(client._request_with_retry(url="https://example.com/v1/responses"))
        await asyncio.sleep(0)
        with pytest.raises(PermanentEndpointError):
            await client._request_with_retry(url="https://example.com/v1/responses")
        gate.set()
        with pytest.raises(PermanentEndpointError):
            await first
        assert calls == 2

    async def test_create_chat_completion_surfaces_permanent_error_status(self, monkeypatch):
        body = b'{"error":{"code":"budget_exceeded"}}'
        response = SimpleNamespace(status=429, content=SimpleNamespace(read=AsyncMock(return_value=body)))
        monkeypatch.setattr("nemo_gym.openai_utils.request", AsyncMock(return_value=response))
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", AsyncMock())
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.com/v1")

        with pytest.raises(PermanentEndpointError) as exc_info:
            await client.create_chat_completion(model="judge", messages=[])

        assert isinstance(exc_info.value, ClientResponseError)
        assert exc_info.value.status == 429
        assert exc_info.value.response_content == body


def _busy(status: int = 429, retry_after: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        headers={"Retry-After": retry_after} if retry_after is not None else {},
        content=SimpleNamespace(read=AsyncMock(return_value=b'{"error":"rate_limit_exceeded"}')),
    )


class TestProviderBackoff:
    """How the client waits on a provider that is busy: growing jittered waits, Retry-After, a time budget."""

    def test_waits_are_jittered_across_the_whole_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        windows = []
        monkeypatch.setattr(
            "nemo_gym.openai_utils.random.uniform", lambda low, high: windows.append((low, high)) or low
        )
        assert openai_utils_module._retry_wait_seconds(3, _busy()) == 0.0
        assert windows == [(0, 4.0)]

    @pytest.mark.parametrize(
        ("retry_after", "expected"), [("7", 7.0), ("0.5", 0.5), ("3600", 60.0), ("soon", 1.0), ("0", 1.0), ("-5", 1.0)]
    )
    def test_retry_after_seconds_is_honoured_within_the_cap(self, retry_after: str, expected: float) -> None:
        assert openai_utils_module._retry_wait_seconds(1, _busy(retry_after=retry_after)) == expected

    def test_retry_after_date_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import datetime, timezone

        class _Now(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)

        monkeypatch.setattr("nemo_gym.openai_utils.datetime", _Now)
        for retry_after in ("Mon, 19 Oct 2026 12:00:20 GMT", "Mon, 19 Oct 2026 12:00:20 -0000"):
            assert openai_utils_module._retry_wait_seconds(1, _busy(retry_after=retry_after)) == 20.0

    @pytest.mark.parametrize("status", [429, 503])
    async def test_a_busy_provider_is_waited_out_until_the_budget_is_spent(
        self, monkeypatch: pytest.MonkeyPatch, status: int
    ) -> None:
        request = AsyncMock(return_value=_busy(status))
        sleep = AsyncMock()
        monkeypatch.setattr("nemo_gym.openai_utils.request", request)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", sleep)

        async def raise_status(response, content=None):
            raise RuntimeError("still busy")

        monkeypatch.setattr("nemo_gym.openai_utils.raise_for_status", raise_status)
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.com/v1", max_rate_limit_wait_seconds=200)

        with pytest.raises(RuntimeError, match="still busy"):
            await client._request_with_retry(url="https://example.com/v1/responses")

        # 1 + 2 + 4 + ... + 64 would pass 200 s, so the 8th answer is the last: 1+2+4+8+16+32+60+60 = 183 s.
        assert [c.args[0] for c in sleep.await_args_list] == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
        assert request.await_count == 9

    @pytest.mark.parametrize("tries", [7, 1025, 100000])
    def test_the_backoff_window_stays_capped_for_any_attempt(
        self, monkeypatch: pytest.MonkeyPatch, tries: int
    ) -> None:
        windows = []
        monkeypatch.setattr(
            "nemo_gym.openai_utils.random.uniform", lambda low, high: windows.append((low, high)) or low
        )
        openai_utils_module._retry_wait_seconds(tries, _busy())
        assert windows == [(0, 60.0)]

    async def test_an_internal_client_keeps_retrying_a_busy_server_past_a_thousand_tries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        success = SimpleNamespace(status=200)
        request = AsyncMock(side_effect=[_busy(429)] * 1025 + [success])
        monkeypatch.setattr("nemo_gym.openai_utils.request", request)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", AsyncMock())
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="http://127.0.0.1:8000/v1", internal=True)

        assert await client._request_with_retry(url="http://127.0.0.1:8000/v1/responses") is success
        assert request.await_count == 1026

    @pytest.mark.parametrize("status", [429, 503])
    async def test_a_positive_budget_ends_the_request_before_max_http_attempts_does(
        self, monkeypatch: pytest.MonkeyPatch, status: int
    ) -> None:
        request = AsyncMock(return_value=_busy(status, retry_after="1"))
        sleep = AsyncMock()
        monkeypatch.setattr("nemo_gym.openai_utils.request", request)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", sleep)

        async def raise_status(response, content=None):
            raise RuntimeError("still busy")

        monkeypatch.setattr("nemo_gym.openai_utils.raise_for_status", raise_status)
        client = NeMoGymAsyncOpenAI(
            api_key="abc", base_url="https://example.com/v1", max_http_attempts=36, max_rate_limit_wait_seconds=2
        )

        with pytest.raises(RuntimeError, match="still busy"):
            await client._request_with_retry(url="https://example.com/v1/responses")

        assert [c.args[0] for c in sleep.await_args_list] == [1.0, 1.0]
        assert request.await_count == 3

    async def test_a_zero_budget_keeps_busy_answers_on_the_attempt_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        request = AsyncMock(return_value=_busy(503))
        monkeypatch.setattr("nemo_gym.openai_utils.request", request)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", AsyncMock())
        monkeypatch.setattr("nemo_gym.openai_utils.random.uniform", lambda low, high: 0.0)

        async def raise_status(response, content=None):
            raise RuntimeError("still busy")

        monkeypatch.setattr("nemo_gym.openai_utils.raise_for_status", raise_status)
        client = NeMoGymAsyncOpenAI(
            api_key="abc", base_url="https://example.com/v1", max_http_attempts=3, max_rate_limit_wait_seconds=0
        )

        with pytest.raises(RuntimeError, match="still busy"):
            await client._request_with_retry(url="https://example.com/v1/responses")

        assert request.await_count == 3

    async def test_a_429_with_retry_after_waits_that_long_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        success = SimpleNamespace(status=200)
        request = AsyncMock(side_effect=[_busy(retry_after="3"), success])
        sleep = AsyncMock()
        monkeypatch.setattr("nemo_gym.openai_utils.request", request)
        monkeypatch.setattr("nemo_gym.openai_utils.sleep", sleep)
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.com/v1")

        assert await client._request_with_retry(url="https://example.com/v1/responses") is success
        sleep.assert_awaited_once_with(3.0)
