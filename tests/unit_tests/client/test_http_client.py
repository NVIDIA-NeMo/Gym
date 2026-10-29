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
"""Tests for `nemo_gym.client.http_client`."""

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
import asyncio
import logging
import multiprocessing
import socket
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from aiohttp import ClientOSError, ClientResponseError, RequestInfo, ServerDisconnectedError, TCPConnector
from multidict import CIMultiDict, CIMultiDictProxy
from omegaconf import DictConfig
from pydantic import ValidationError
from pytest import CaptureFixture, LogCaptureFixture, MonkeyPatch, fail, mark, raises
from yarl import URL

import nemo_gym.client.http_client
import nemo_gym.client.server_client
import nemo_gym.config.global_config
import nemo_gym.server_utils
from nemo_gym.client import GlobalAIOHTTPAsyncClientConfig, format_exception_with_response_body, raise_for_status
from nemo_gym.client.http_client import _format_upstream_error_log, _make_keepalive_socket_factory, _set_tcp_keepalive
from nemo_gym.config.global_config import (
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
)
from nemo_gym.telemetry import connection_pool
from nemo_gym.telemetry.connection_pool import connection_pool_capacity, report_connection_pool_capacity


_TCP_KEEPALIVE_TEST_IDLE = 42
_TCP_KEEPALIVE_TEST_INTERVAL = 7
_TCP_KEEPALIVE_TEST_PROBES = 2


_TEST_ADDR_INFO = (
    socket.AF_INET,
    socket.SOCK_STREAM,
    socket.IPPROTO_TCP,
    "",
    ("203.0.113.1", 443),
)


def _return_exception_from_child_process(error: ClientResponseError) -> ClientResponseError:
    return error


class TestRequestRetries:
    @mark.parametrize("internal", [False, True])
    @mark.parametrize("traced", [False, True])
    @mark.parametrize("metrics", [False, True])
    @mark.parametrize("failure_type", [ServerDisconnectedError, ClientOSError, TimeoutError, RuntimeError])
    async def test_single_attempt_preserves_failure_after_execution(
        self, monkeypatch: MonkeyPatch, internal: bool, traced: bool, metrics: bool, failure_type: type[Exception]
    ) -> None:
        effects = []
        failure = failure_type("response lost after execution")

        async def execute_then_fail(**kwargs):
            effects.append("applied")
            raise failure

        send = AsyncMock(side_effect=execute_then_fail)
        sleep = AsyncMock()
        monkeypatch.setattr("nemo_gym.client.http_client.get_global_aiohttp_client", lambda: MagicMock(request=send))
        monkeypatch.setattr(nemo_gym.client.http_client.asyncio, "sleep", sleep)
        monkeypatch.setattr(nemo_gym.client.http_client, "is_span_group_enabled", lambda _: traced)
        monkeypatch.setattr(nemo_gym.client.http_client, "is_trace_exporter_active", lambda: traced)
        counts = SimpleNamespace(outbound_in_flight=0)
        monkeypatch.setattr(nemo_gym.client.http_client, "request_counts", lambda: counts if metrics else None)
        monkeypatch.setattr("nemo_gym.telemetry.spans.client_span", lambda _: nullcontext(None))
        monkeypatch.setattr("nemo_gym.telemetry.contrib.inject_trace_context", lambda headers: None)

        with raises(failure_type) as exc_info:
            await nemo_gym.client.http_client.request(
                "POST",
                "http://server.test/step",
                _internal=internal,
                _resend_after_disconnect=False,
                json={"action": "move"},
            )

        assert exc_info.value is failure
        assert effects == ["applied"]
        send.assert_awaited_once_with(
            method="POST",
            url="http://server.test/step",
            data=b'{"action":"move"}',
            headers={"Content-Type": "application/json"},
        )
        sleep.assert_not_awaited()
        assert counts.outbound_in_flight == 0

    @mark.parametrize("internal", [False, True])
    @mark.parametrize("failure_type", [ServerDisconnectedError, ClientOSError, TimeoutError, RuntimeError])
    async def test_default_retries_recover(
        self, monkeypatch: MonkeyPatch, internal: bool, failure_type: type[Exception]
    ) -> None:
        response = MagicMock()
        send = AsyncMock(side_effect=[failure_type("transient failure"), response])
        sleep = AsyncMock()
        monkeypatch.setattr("nemo_gym.client.http_client.get_global_aiohttp_client", lambda: MagicMock(request=send))
        monkeypatch.setattr(nemo_gym.client.http_client.asyncio, "sleep", sleep)
        monkeypatch.setattr(nemo_gym.client.http_client, "is_span_group_enabled", lambda _: False)

        actual = await nemo_gym.client.http_client.request("POST", "http://server.test/step", _internal=internal)

        assert actual is response
        assert send.await_count == 2
        sleep.assert_awaited_once_with(0.5)


class TestHTTPClient:
    def test_global_aiohttp_client_exit_uses_owner_loop(self, monkeypatch: MonkeyPatch) -> None:
        owner_loop = asyncio.new_event_loop()

        class LoopBoundClient:
            closed = False

            async def close(self) -> None:
                assert asyncio.get_running_loop() is owner_loop
                self.closed = True

        client = LoopBoundClient()
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", client)
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_LOOP", owner_loop)
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY", True)

        try:
            nemo_gym.client.http_client.global_aiohttp_client_exit()
        finally:
            owner_loop.close()

        assert client.closed
        assert nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT is None
        assert nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_LOOP is None
        assert nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY is False

    async def test_global_aiohttp_client_exit_from_owner_loop_is_non_blocking(self, monkeypatch: MonkeyPatch) -> None:
        owner_loop = asyncio.get_running_loop()

        class LoopBoundClient:
            closed = False

            async def close(self) -> None:
                assert asyncio.get_running_loop() is owner_loop
                self.closed = True

        client = LoopBoundClient()
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", client)
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_LOOP", owner_loop)
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY", True)

        nemo_gym.client.http_client.global_aiohttp_client_exit()

        assert nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT is None
        assert nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_LOOP is None
        assert nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY is False
        await asyncio.sleep(0)
        assert client.closed

    def test_global_aiohttp_client_exit_closes_control_client_on_owner_loop(self, monkeypatch: MonkeyPatch) -> None:
        owner_loop = asyncio.new_event_loop()

        class LoopBoundClient:
            closed = False

            async def close(self) -> None:
                assert asyncio.get_running_loop() is owner_loop
                self.closed = True

        data_client, control_client = LoopBoundClient(), LoopBoundClient()
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", data_client)
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CONTROL_CLIENT", control_client)
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_LOOP", owner_loop)

        try:
            nemo_gym.client.http_client.global_aiohttp_client_exit()
        finally:
            owner_loop.close()

        assert data_client.closed
        assert control_client.closed
        assert nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT is None
        assert nemo_gym.client.http_client._GLOBAL_AIOHTTP_CONTROL_CLIENT is None
        assert nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_LOOP is None

    async def test_raise_for_status_preserves_message_across_process_boundary(self) -> None:
        headers = CIMultiDictProxy(
            CIMultiDict(
                [
                    ("x-request-id", "request-123"),
                    ("Retry-After", "10"),
                    ("retry-after", "20"),
                    ("Set-Cookie", "session=abc"),
                    ("Set-Cookie", "preferences=dark"),
                ]
            )
        )
        request_info = RequestInfo(
            url=URL("http://resources-server.test/verify"),
            method="POST",
            headers=headers,
            real_url=URL("http://resources-server.test/verify"),
        )
        original_error = ClientResponseError(
            request_info=request_info,
            history=(),
            status=500,
            message="verifier failed",
            headers=headers,
        )
        response = MagicMock()
        response.ok = False
        response.content.read = AsyncMock(return_value=b'{"detail":"backend unavailable"}')
        response.request_info = request_info
        response.raise_for_status.side_effect = original_error

        with raises(ClientResponseError) as exc_info:
            await raise_for_status(response)

        error = exc_info.value
        assert str(error) == ("500, message='verifier failed', url='http://resources-server.test/verify'")
        assert error.response_content == b'{"detail":"backend unavailable"}'

        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
            restored_error = executor.submit(_return_exception_from_child_process, error).result()

        assert isinstance(restored_error, ClientResponseError)
        assert str(restored_error) == str(error)
        assert restored_error.status == 500
        assert restored_error.message == "verifier failed"
        assert restored_error.response_content == error.response_content
        assert restored_error.request_info.method == "POST"
        assert isinstance(restored_error.request_info.headers, CIMultiDict)
        assert restored_error.request_info.headers["X-REQUEST-ID"] == "request-123"
        assert restored_error.request_info.headers.getall("RETRY-AFTER") == ["10", "20"]
        assert restored_error.request_info.headers.getall("set-cookie") == ["session=abc", "preferences=dark"]
        assert isinstance(restored_error.headers, CIMultiDict)
        assert restored_error.headers.getall("retry-after") == ["10", "20"]
        assert restored_error.headers.getall("SET-COOKIE") == ["session=abc", "preferences=dark"]

    async def test_raise_for_status_accepts_prefetched_content(self) -> None:
        request_info = RequestInfo(
            url=URL("http://judge.test/v1/responses"),
            method="POST",
            headers=CIMultiDictProxy(CIMultiDict()),
            real_url=URL("http://judge.test/v1/responses"),
        )
        original_error = ClientResponseError(
            request_info=request_info,
            history=(),
            status=429,
            message="Too Many Requests",
            headers=CIMultiDictProxy(CIMultiDict()),
        )
        response = MagicMock()
        response.ok = False
        response.content.read = AsyncMock(side_effect=AssertionError("body already consumed"))
        response.request_info = request_info
        response.raise_for_status.side_effect = original_error
        content = b'{"error":"rate_limit_exceeded"}'

        with raises(ClientResponseError) as exc_info:
            await raise_for_status(response, content)

        assert exc_info.value.response_content == content
        response.content.read.assert_not_awaited()

    async def test_raise_for_status_debug_print_omits_request_headers(
        self, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
    ) -> None:
        api_key = "sk-FAKE-CANARY"  # pragma: allowlist secret
        url = URL(f"http://model.test/v1/chat/completions?api-key={api_key}")
        request_info = RequestInfo(
            url=url,
            method="POST",
            headers=CIMultiDictProxy(CIMultiDict({"Authorization": f"Bearer {api_key}"})),
            real_url=url,
        )
        response = MagicMock()
        response.ok = False
        response.request_info = request_info
        response.raise_for_status.side_effect = ClientResponseError(
            request_info=request_info, history=(), status=401, message="Unauthorized"
        )
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", True)

        with raises(ClientResponseError):
            await raise_for_status(response, b'{"error":"invalid api key"}')

        printed = capsys.readouterr().out
        assert api_key not in printed
        assert "Request info: POST http://model.test/v1/chat/completions\n" in printed

    def test_global_aiohttp_client_request_debug_enabled(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", False)
        assert not nemo_gym.client.http_client.is_global_aiohttp_client_request_debug_enabled()

        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", True)
        assert nemo_gym.client.http_client.is_global_aiohttp_client_request_debug_enabled()

    @mark.parametrize("tracing_enabled", [False, True])
    @mark.parametrize("internal", [False, True])
    @mark.parametrize("error_type", [RuntimeError, ClientOSError, nemo_gym.client.http_client.ServerDisconnectedError])
    async def test_explicit_transport_attempt_cap_disables_request_retry(
        self,
        monkeypatch: MonkeyPatch,
        tracing_enabled: bool,
        internal: bool,
        error_type: type[Exception],
    ) -> None:
        monkeypatch.setattr(nemo_gym.client.http_client, "is_span_group_enabled", lambda _group: tracing_enabled)
        client = self._mock_global_client(monkeypatch, error_type("transport failed"), failures=None)

        with raises(error_type, match="transport failed"):
            await nemo_gym.client.http_client.request(
                method="POST",
                url="https://example.test",
                _internal=internal,
                _max_num_tries=1,
            )

        assert client.request.await_count == 1

    @mark.parametrize("attempt_cap", [0, -1])
    async def test_explicit_transport_attempt_cap_rejects_nonpositive_values(
        self, monkeypatch: MonkeyPatch, attempt_cap: int
    ) -> None:
        client = self._mock_global_client(monkeypatch, ClientOSError(), failures=0)
        with raises(ValueError, match="_max_num_tries must be at least 1"):
            await nemo_gym.client.http_client.request("POST", "https://example.test", _max_num_tries=attempt_cap)
        client.request.assert_not_awaited()

    @mark.parametrize("tracing_enabled", [False, True])
    async def test_explicit_transport_attempt_cap_counts_mixed_errors(
        self,
        monkeypatch: MonkeyPatch,
        tracing_enabled: bool,
    ) -> None:
        monkeypatch.setattr(nemo_gym.client.http_client, "is_span_group_enabled", lambda _group: tracing_enabled)
        client = MagicMock()
        client.request = AsyncMock(
            side_effect=[
                RuntimeError("generic failure"),
                nemo_gym.client.http_client.ClientOSError("socket failure"),
                "must not be reached",
            ]
        )
        monkeypatch.setattr(
            "nemo_gym.client.http_client.get_global_aiohttp_client",
            lambda: client,
        )
        monkeypatch.setattr(
            nemo_gym.client.http_client.asyncio,
            "sleep",
            AsyncMock(),
        )

        with raises(
            nemo_gym.client.http_client.ClientOSError,
            match="socket failure",
        ):
            await nemo_gym.client.http_client.request(
                method="POST",
                url="https://example.test",
                _max_num_tries=2,
            )

        assert client.request.await_count == 2

    async def test_get_global_aiohttp_client_uses_defaults_without_config_or_cli_parse(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """A process Gym did not launch (a Ray task worker) has no config and argv that is not Gym's.

        The client must come up with the default limits instead of parsing that argv and exiting.
        """
        monkeypatch.setattr(nemo_gym.config.global_config, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.delenv(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME, raising=False)
        monkeypatch.delenv(nemo_gym.utils.server_process.NEMO_GYM_FASTAPI_NUM_WORKERS, raising=False)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "/usr/lib/python3/site-packages/ray/_private/workers/default_worker.py",
                "--node-ip-address=127.0.0.1",
                "--node-manager-port=40000",
                "--worker-type=WORKER",
            ],
        )
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_CONFIG", None)
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", False)
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_QUEUE_TELEMETRY", False)

        try:
            client = nemo_gym.client.http_client.get_global_aiohttp_client()
        except SystemExit as e:
            fail(f"get_global_aiohttp_client parsed the process argv as Gym CLI overrides and exited: {e!r}")

        try:
            defaults = GlobalAIOHTTPAsyncClientConfig()
            assert client.connector.limit == defaults.global_aiohttp_connector_limit
            assert client.connector.limit_per_host == defaults.global_aiohttp_connector_limit_per_host
            assert nemo_gym.config.global_config._GLOBAL_CONFIG_DICT is None
        finally:
            await client.close()

    async def test_get_global_aiohttp_client_parses_an_explicit_parser_config(self, monkeypatch: MonkeyPatch) -> None:
        """A caller that passes a parser config gets the limits it describes, not the defaults."""
        monkeypatch.setattr(nemo_gym.config.global_config, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.delenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
        monkeypatch.delenv(nemo_gym.utils.server_process.NEMO_GYM_FASTAPI_NUM_WORKERS, raising=False)
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_CONFIG", None)
        parser_config = nemo_gym.config.global_config.GlobalConfigDictParserConfig(
            initial_global_config_dict=DictConfig(
                {"global_aiohttp_connector_limit": 7, "global_aiohttp_connector_limit_per_host": 3}
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
        )

        client = nemo_gym.client.http_client.get_global_aiohttp_client(global_config_dict_parser_config=parser_config)
        try:
            assert (client.connector.limit, client.connector.limit_per_host) == (7, 3)
        finally:
            await client.close()

    def test_keepalive_socket_factory_sets_keepalive_sockopts(self, monkeypatch: MonkeyPatch) -> None:
        mock_sock = MagicMock()
        socket_ctor_mock = MagicMock(return_value=mock_sock)
        monkeypatch.setattr(socket, "socket", socket_ctor_mock)

        factory = _make_keepalive_socket_factory(
            idle_seconds=_TCP_KEEPALIVE_TEST_IDLE,
            interval_seconds=_TCP_KEEPALIVE_TEST_INTERVAL,
            probes=_TCP_KEEPALIVE_TEST_PROBES,
        )
        result = factory(_TEST_ADDR_INFO)

        assert result is mock_sock
        socket_ctor_mock.assert_called_once_with(
            family=_TEST_ADDR_INFO[0], type=_TEST_ADDR_INFO[1], proto=_TEST_ADDR_INFO[2]
        )
        mock_sock.setsockopt.assert_any_call(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for opt_name, opt_value in (
            ("TCP_KEEPIDLE", _TCP_KEEPALIVE_TEST_IDLE),
            ("TCP_KEEPINTVL", _TCP_KEEPALIVE_TEST_INTERVAL),
            ("TCP_KEEPCNT", _TCP_KEEPALIVE_TEST_PROBES),
        ):
            opt = getattr(socket, opt_name, None)
            if opt is not None:
                mock_sock.setsockopt.assert_any_call(socket.IPPROTO_TCP, opt, opt_value)

    def test_keepalive_socket_factory_skips_missing_platform_sockopts(self, monkeypatch: MonkeyPatch) -> None:
        mock_sock = MagicMock()
        socket_ctor_mock = MagicMock(return_value=mock_sock)
        monkeypatch.setattr(socket, "socket", socket_ctor_mock)
        for opt_name in ("TCP_KEEPIDLE", "TCP_KEEPALIVE", "TCP_KEEPINTVL", "TCP_KEEPCNT"):
            monkeypatch.delattr(socket, opt_name, raising=False)

        factory = _make_keepalive_socket_factory(
            idle_seconds=_TCP_KEEPALIVE_TEST_IDLE,
            interval_seconds=_TCP_KEEPALIVE_TEST_INTERVAL,
            probes=_TCP_KEEPALIVE_TEST_PROBES,
        )
        factory(_TEST_ADDR_INFO)

        mock_sock.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    def test_keepalive_idle_falls_back_to_macos_tcp_keepalive(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.delattr(socket, "TCP_KEEPIDLE", raising=False)
        monkeypatch.setattr(socket, "TCP_KEEPALIVE", 0x10, raising=False)
        mock_sock = MagicMock()

        _set_tcp_keepalive(
            mock_sock, _TCP_KEEPALIVE_TEST_IDLE, _TCP_KEEPALIVE_TEST_INTERVAL, _TCP_KEEPALIVE_TEST_PROBES
        )

        mock_sock.setsockopt.assert_any_call(socket.IPPROTO_TCP, 0x10, _TCP_KEEPALIVE_TEST_IDLE)

    def test_GlobalAIOHTTPAsyncClientConfig_keepalive_defaults(self) -> None:
        cfg = GlobalAIOHTTPAsyncClientConfig()
        assert cfg.global_aiohttp_tcp_keepalive_idle_seconds == 60
        assert cfg.global_aiohttp_tcp_keepalive_interval_seconds == 10
        assert cfg.global_aiohttp_tcp_keepalive_probes == 3

    @mark.parametrize(
        ("workers", "expected_total", "expected_per_host"),
        [(1, 101, 17), (4, 25, 4), (16, 6, 1)],
    )
    def test_connection_pool_capacity_divides_aggregate_limits(
        self, workers: int, expected_total: int, expected_per_host: int
    ) -> None:
        cfg = GlobalAIOHTTPAsyncClientConfig(
            global_aiohttp_connector_limit=101,
            global_aiohttp_connector_limit_per_host=17,
        )

        capacity = connection_pool_capacity(cfg, workers)

        assert capacity.total == expected_total
        assert capacity.per_host == expected_per_host

    def test_connection_pool_capacity_rounds_intended_concurrency_up(self) -> None:
        cfg = GlobalAIOHTTPAsyncClientConfig(
            global_aiohttp_intended_concurrency=13,
            global_aiohttp_intended_concurrency_per_host=5,
        )

        capacity = connection_pool_capacity(cfg, workers=4)

        assert (capacity.intended, capacity.intended_per_host) == (4, 2)

    @mark.parametrize("workers", [0, -1])
    def test_connection_pool_capacity_rejects_invalid_worker_count(self, workers: int) -> None:
        with raises(ValueError, match="worker count must be at least 1"):
            connection_pool_capacity(GlobalAIOHTTPAsyncClientConfig(), workers)

    @mark.parametrize(
        ("total", "per_host", "workers"),
        [(3, 2, 4), (3, 1024, 4), (100 * 1024, 8, 16), (0, 8, 16), (3, 0, 4), (1, 1024, 2), (1024, 1, 2)],
        ids=[
            "both",
            "total-only",
            "per-host-only",
            "per-host-with-unlimited-total",
            "total-with-unlimited-per-host",
            "total-of-one",
            "per-host-of-one",
        ],
    )
    def test_connection_pool_capacity_rejects_zero_effective_limit(
        self, total: int, per_host: int, workers: int
    ) -> None:
        cfg = GlobalAIOHTTPAsyncClientConfig(
            global_aiohttp_connector_limit=total,
            global_aiohttp_connector_limit_per_host=per_host,
        )

        with raises(ValueError, match="must remain at least 1"):
            connection_pool_capacity(cfg, workers=workers)

    @mark.parametrize("field", ["global_aiohttp_connector_limit", "global_aiohttp_connector_limit_per_host"])
    def test_connection_pool_config_rejects_negative_limits(self, field: str) -> None:
        with raises(ValidationError):
            GlobalAIOHTTPAsyncClientConfig(**{field: -1})

    @mark.parametrize("field", ["global_aiohttp_intended_concurrency", "global_aiohttp_intended_concurrency_per_host"])
    def test_connection_pool_config_rejects_nonpositive_intended_concurrency(self, field: str) -> None:
        with raises(ValidationError):
            GlobalAIOHTTPAsyncClientConfig(**{field: 0})

    @mark.parametrize(("total", "per_host"), [(0, 0), (0, 64), (64, 0)])
    @mark.parametrize("workers", [1, 4, 16])
    def test_connection_pool_capacity_preserves_explicit_unlimited_limits(
        self, total: int, per_host: int, workers: int
    ) -> None:
        cfg = GlobalAIOHTTPAsyncClientConfig(
            global_aiohttp_connector_limit=total,
            global_aiohttp_connector_limit_per_host=per_host,
        )
        capacity = connection_pool_capacity(cfg, workers=workers)
        assert capacity.total == (total // workers if total else 0)
        assert capacity.per_host == (per_host // workers if per_host else 0)

    def test_connection_pool_capacity_reports_effective_limits_and_warns(
        self,
        caplog: LogCaptureFixture,
        capsys: CaptureFixture[str],
        monkeypatch: MonkeyPatch,
    ) -> None:
        cfg = GlobalAIOHTTPAsyncClientConfig(
            global_aiohttp_connector_limit=10,
            global_aiohttp_connector_limit_per_host=6,
            global_aiohttp_intended_concurrency=12,
            global_aiohttp_intended_concurrency_per_host=8,
        )
        capacity = connection_pool_capacity(cfg, workers=4)
        monkeypatch.setattr(connection_pool, "_ephemeral_port_capacity", lambda: 5)
        connection_pool._REPORTED_CAPACITIES.clear()

        with caplog.at_level(logging.INFO, logger="nemo_gym.telemetry.connection_pool"):
            report_connection_pool_capacity(cfg, capacity, visible=True)

        visible_report = capsys.readouterr().out
        assert "aggregate_total=10" in visible_report
        assert "effective_total=2" in visible_report
        assert "intended per-worker concurrency 3 exceeds effective total limit 2" in caplog.text
        assert "intended per-host concurrency 2 exceeds effective per-host limit 1" in caplog.text
        assert "aggregate intended per-host concurrency 8" in caplog.text

    @mark.parametrize(
        ("workers", "effective_total", "per_worker_per_host", "intended_per_host"),
        [(4, 16, 256, 250), (16, 4, 64, 63)],
    )
    def test_per_host_report_and_warning_respect_the_total_clamp(
        self,
        workers: int,
        effective_total: int,
        per_worker_per_host: int,
        intended_per_host: int,
        caplog: LogCaptureFixture,
        capsys: CaptureFixture[str],
        monkeypatch: MonkeyPatch,
    ) -> None:
        """aiohttp serves min(limit, limit_per_host), so a large per-host limit is not reachable.

        Reporting the configured value instead of the enforced one hid real oversubscription:
        with limit=64/workers=4 a host can only reach 16, however large limit_per_host is.
        """
        cfg = GlobalAIOHTTPAsyncClientConfig(
            global_aiohttp_connector_limit=64,
            global_aiohttp_connector_limit_per_host=1024,
            global_aiohttp_intended_concurrency_per_host=1000,
        )
        capacity = connection_pool_capacity(cfg, workers=workers)
        monkeypatch.setattr(connection_pool, "_ephemeral_port_capacity", lambda: None)
        connection_pool._REPORTED_CAPACITIES.clear()

        with caplog.at_level(logging.INFO, logger="nemo_gym.telemetry.connection_pool"):
            report_connection_pool_capacity(cfg, capacity, visible=True)

        visible_report = capsys.readouterr().out
        # The per-worker value is distinct from both the aggregate config and total clamp.
        assert "aggregate_per_host=1024" in visible_report
        assert f"effective_per_host={effective_total}" in visible_report
        assert f"per_worker_per_host={per_worker_per_host}" in visible_report
        assert "configured_per_host=" not in visible_report
        assert (
            f"intended per-host concurrency {intended_per_host} exceeds effective per-host limit {effective_total}"
            in caplog.text
        )

    @mark.parametrize(
        ("total", "workers", "effective_total"),
        [(100 * 1024, 1, "102400"), (100 * 1024, 2, "51200"), (0, 1, "unlimited"), (0, 2, "unlimited")],
    )
    def test_connection_pool_limit_alone_does_not_warn_against_file_descriptor_budget(
        self,
        total: int,
        workers: int,
        effective_total: str,
        caplog: LogCaptureFixture,
        monkeypatch: MonkeyPatch,
    ) -> None:
        cfg = GlobalAIOHTTPAsyncClientConfig(
            global_aiohttp_connector_limit=total,
        )
        capacity = connection_pool_capacity(cfg, workers=workers)
        connection_pool._REPORTED_CAPACITIES.clear()
        monkeypatch.setattr(connection_pool.resource, "getrlimit", lambda _resource: (65535, 65535))
        monkeypatch.setattr(connection_pool, "_ephemeral_port_capacity", lambda: None)

        with caplog.at_level(logging.INFO, logger="nemo_gym.telemetry.connection_pool"):
            report_connection_pool_capacity(cfg, capacity)

        assert f"effective_total={effective_total} " in caplog.text
        assert "file_descriptor_soft_limit=65535" in caplog.text
        assert not any(record.levelno >= logging.WARNING for record in caplog.records)

    @mark.parametrize("workers", [1, 4])
    def test_unlimited_limits_are_reported_and_skip_demand_warnings(
        self,
        workers: int,
        caplog: LogCaptureFixture,
        capsys: CaptureFixture[str],
        monkeypatch: MonkeyPatch,
    ) -> None:
        cfg = GlobalAIOHTTPAsyncClientConfig(
            global_aiohttp_connector_limit=0,
            global_aiohttp_connector_limit_per_host=0,
            global_aiohttp_intended_concurrency=4096,
            global_aiohttp_intended_concurrency_per_host=1024,
        )
        capacity = connection_pool_capacity(cfg, workers=workers)
        connection_pool._REPORTED_CAPACITIES.clear()
        monkeypatch.setattr(connection_pool.resource, "getrlimit", lambda _resource: (1048576, 1048576))
        monkeypatch.setattr(connection_pool, "_ephemeral_port_capacity", lambda: None)

        with caplog.at_level(logging.INFO, logger="nemo_gym.telemetry.connection_pool"):
            report_connection_pool_capacity(cfg, capacity, visible=True)

        visible_report = capsys.readouterr().out
        for field in (
            "aggregate_total",
            "aggregate_per_host",
            "effective_total",
            "effective_per_host",
            "per_worker_per_host",
        ):
            assert f"{field}=unlimited " in visible_report
        assert not any(record.levelno >= logging.WARNING for record in caplog.records)

    def test_unlimited_total_still_checks_a_finite_per_host_limit(
        self,
        caplog: LogCaptureFixture,
        capsys: CaptureFixture[str],
        monkeypatch: MonkeyPatch,
    ) -> None:
        cfg = GlobalAIOHTTPAsyncClientConfig(
            global_aiohttp_connector_limit=0,
            global_aiohttp_connector_limit_per_host=8,
            global_aiohttp_intended_concurrency=4096,
            global_aiohttp_intended_concurrency_per_host=16,
        )
        capacity = connection_pool_capacity(cfg, workers=1)
        connection_pool._REPORTED_CAPACITIES.clear()
        monkeypatch.setattr(connection_pool.resource, "getrlimit", lambda _resource: (1048576, 1048576))
        monkeypatch.setattr(connection_pool, "_ephemeral_port_capacity", lambda: None)

        with caplog.at_level(logging.INFO, logger="nemo_gym.telemetry.connection_pool"):
            report_connection_pool_capacity(cfg, capacity, visible=True)

        visible_report = capsys.readouterr().out
        assert "effective_total=unlimited " in visible_report
        assert "effective_per_host=8 " in visible_report
        assert "intended per-host concurrency 16 exceeds effective per-host limit 8" in caplog.text
        assert "exceeds effective total limit" not in caplog.text

    def test_unlimited_per_host_is_clamped_to_the_total_limit(
        self, caplog: LogCaptureFixture, monkeypatch: MonkeyPatch
    ) -> None:
        cfg = GlobalAIOHTTPAsyncClientConfig(
            global_aiohttp_connector_limit=8,
            global_aiohttp_connector_limit_per_host=0,
            global_aiohttp_intended_concurrency_per_host=20,
        )
        capacity = connection_pool_capacity(cfg, workers=1)
        connection_pool._REPORTED_CAPACITIES.clear()
        monkeypatch.setattr(connection_pool, "_ephemeral_port_capacity", lambda: None)

        with caplog.at_level(logging.INFO, logger="nemo_gym.telemetry.connection_pool"):
            report_connection_pool_capacity(cfg, capacity)

        assert "effective_per_host=8 per_worker_per_host=unlimited" in caplog.text
        assert "intended per-host concurrency 20 exceeds effective per-host limit 8" in caplog.text

    def test_intended_concurrency_warns_with_file_descriptor_prefix(
        self, caplog: LogCaptureFixture, monkeypatch: MonkeyPatch
    ) -> None:
        cfg = GlobalAIOHTTPAsyncClientConfig(
            global_aiohttp_connector_limit=1000,
            global_aiohttp_intended_concurrency=100,
        )
        capacity = connection_pool_capacity(cfg, workers=1)
        connection_pool._REPORTED_CAPACITIES.clear()
        monkeypatch.setattr(connection_pool.resource, "getrlimit", lambda _resource: (64, 64))
        monkeypatch.setattr(connection_pool, "_ephemeral_port_capacity", lambda: None)

        with caplog.at_level(logging.WARNING, logger="nemo_gym.telemetry.connection_pool"):
            report_connection_pool_capacity(cfg, capacity)

        assert (
            "aiohttp file-descriptor capacity may be exhausted: intended per-worker concurrency 100 can exhaust "
            "the file-descriptor soft limit 64"
        ) in caplog.text
        assert "aiohttp connection pool may queue requests" not in caplog.text

    async def test_connection_pool_telemetry_is_not_installed_when_disabled(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
        monkeypatch.setattr("nemo_gym.utils.server_process.get_nemo_gym_fastapi_num_workers", lambda: 1)
        monkeypatch.setattr(connection_pool, "is_metrics_exporter_active", lambda: False)
        monkeypatch.setattr(nemo_gym.client.http_client, "is_nemo_gym_fastapi_worker", lambda: True)
        report = MagicMock()
        monkeypatch.setattr(nemo_gym.client.http_client, "report_connection_pool_capacity", report)

        client = nemo_gym.client.http_client.set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
        try:
            assert type(client.connector) is TCPConnector
            assert client.trace_configs == []
            report.assert_not_called()
        finally:
            await client.close()
            monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)

    async def test_connection_pool_telemetry_is_installed_when_enabled(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
        monkeypatch.setattr("nemo_gym.utils.server_process.get_nemo_gym_fastapi_num_workers", lambda: 1)
        monkeypatch.setattr(connection_pool, "is_metrics_exporter_active", lambda: True)
        monkeypatch.setattr(nemo_gym.client.http_client, "is_span_group_enabled", lambda _group: False)
        monkeypatch.setattr(nemo_gym.client.http_client, "is_nemo_gym_fastapi_worker", lambda: True)

        client = nemo_gym.client.http_client.set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
        try:
            assert isinstance(client.connector, connection_pool.QueueTimedTCPConnector)
            assert client.trace_configs == []
        finally:
            await client.close()
            monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)

    @mark.parametrize(
        "field, value",
        [
            ("global_aiohttp_tcp_keepalive_idle_seconds", 0),
            ("global_aiohttp_tcp_keepalive_idle_seconds", 32768),
            ("global_aiohttp_tcp_keepalive_interval_seconds", 0),
            ("global_aiohttp_tcp_keepalive_interval_seconds", 32768),
            ("global_aiohttp_tcp_keepalive_probes", 0),
            ("global_aiohttp_tcp_keepalive_probes", 128),
        ],
    )
    def test_GlobalAIOHTTPAsyncClientConfig_rejects_out_of_range_keepalive(self, field: str, value: int) -> None:
        # Linux setsockopt returns EINVAL for these, so they must fail at config load instead.
        with raises(ValidationError):
            GlobalAIOHTTPAsyncClientConfig.model_validate({field: value})

    def test_keepalive_socket_factory_uses_configured_values(self, monkeypatch: MonkeyPatch) -> None:
        mock_sock = MagicMock()
        socket_ctor_mock = MagicMock(return_value=mock_sock)
        monkeypatch.setattr(socket, "socket", socket_ctor_mock)

        cfg = GlobalAIOHTTPAsyncClientConfig(
            global_aiohttp_tcp_keepalive_idle_seconds=123,
            global_aiohttp_tcp_keepalive_interval_seconds=45,
            global_aiohttp_tcp_keepalive_probes=6,
        )
        factory = _make_keepalive_socket_factory(
            idle_seconds=cfg.global_aiohttp_tcp_keepalive_idle_seconds,
            interval_seconds=cfg.global_aiohttp_tcp_keepalive_interval_seconds,
            probes=cfg.global_aiohttp_tcp_keepalive_probes,
        )
        factory(_TEST_ADDR_INFO)

        for opt_name, opt_value in (
            ("TCP_KEEPIDLE", 123),
            ("TCP_KEEPINTVL", 45),
            ("TCP_KEEPCNT", 6),
        ):
            opt = getattr(socket, opt_name, None)
            if opt is not None:
                mock_sock.setsockopt.assert_any_call(socket.IPPROTO_TCP, opt, opt_value)

    def test_upstream_error_log_has_bounded_body_and_redacted_url(self) -> None:
        request_info = RequestInfo(
            url=URL("http://policy.test/v1/responses?api_key=secret"),
            method="POST",
            headers=CIMultiDictProxy(CIMultiDict()),
            real_url=URL("http://policy.test/v1/responses?api_key=secret"),
        )
        error = ClientResponseError(
            request_info=request_info,
            history=(),
            status=500,
            message="policy failed",
        )
        error.response_content = (
            b"Traceback (most recent call last):\nValueError: actionable inner failure\n" + b"x" * 3000
        )

        message = _format_upstream_error_log("TestSimpleServer___my_server", error)

        assert "[upstream_request_failed]" in message
        assert "server=TestSimpleServer___my_server" in message
        assert "method=POST url=http://policy.test/v1/responses status=500" in message
        assert "ValueError: actionable inner failure" in message
        assert "api_key=secret" not in message
        assert message.endswith("…")
        assert len(message) < 2200

    def test_format_exception_with_response_body_carries_the_upstream_reason(self) -> None:
        request_info = RequestInfo(
            url=URL("http://agent.test/v1/agent_sessions"),
            method="POST",
            headers=CIMultiDictProxy(CIMultiDict()),
            real_url=URL("http://agent.test/v1/agent_sessions"),
        )
        refusal = ClientResponseError(
            request_info=request_info, history=(), status=422, message="Unprocessable Content"
        )
        refusal.response_content = b'{"detail":"Simple Agent does not support sandbox access"}'
        bodiless = ClientResponseError(request_info=request_info, history=(), status=503, message="Unavailable")

        # str() alone names only the status, reason phrase, and URL.
        assert format_exception_with_response_body(refusal) == (
            f'{refusal}: {{"detail":"Simple Agent does not support sandbox access"}}'
        )
        assert format_exception_with_response_body(bodiless) == str(bodiless)
        assert format_exception_with_response_body(ValueError("invalid payload")) == "invalid payload"

    def _mock_global_client(self, monkeypatch: MonkeyPatch, error: Exception, failures: int | None) -> MagicMock:
        """Global-client stand-in whose request() raises `error` on the first `failures` calls, then succeeds.

        With `failures=None`, request() always raises. Once called more often than any test needs, it fails the test
        instead, so a retry loop that never stops fails fast instead of hanging. It uses `fail()`, whose BaseException
        the loop's `except Exception` cannot catch.
        """
        client = MagicMock()
        # The most any test needs is MAX_NUM_TRIES + 3 calls.
        allowed_calls = nemo_gym.client.http_client.MAX_NUM_TRIES + 6
        attempts = 0

        def request(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts > allowed_calls:
                fail("retry loop did not stop")
            if failures is None or attempts <= failures:
                raise error
            return client.success_response

        client.request = AsyncMock(side_effect=request)
        monkeypatch.setattr("nemo_gym.client.http_client.get_global_aiohttp_client", lambda: client)
        monkeypatch.setattr(nemo_gym.client.http_client.asyncio, "sleep", AsyncMock())
        return client

    @mark.parametrize("tracing_enabled", [False, True])
    @mark.parametrize(("attempt_cap", "expected_attempts"), [(None, 3), (1, 1), (5, 3)])
    async def test_request_bounded_connection_retries_surface_dead_endpoint(
        self, monkeypatch: MonkeyPatch, tracing_enabled: bool, attempt_cap: int | None, expected_attempts: int
    ) -> None:
        monkeypatch.setattr(nemo_gym.client.http_client, "is_span_group_enabled", lambda _group: tracing_enabled)
        client = self._mock_global_client(monkeypatch, ClientOSError(), failures=None)
        with raises(ClientOSError):
            await nemo_gym.client.http_client.request(
                "POST", "http://dead-host:1/v1", _max_num_tries=attempt_cap, _max_connection_retries=3
            )
        assert client.request.await_count == expected_attempts
        assert "_max_num_tries" not in client.request.call_args.kwargs
        assert "_max_connection_retries" not in client.request.call_args.kwargs

    @mark.parametrize(("attempt_cap", "expected_attempts"), [(1, 1), (5, 3)])
    async def test_request_caps_generic_errors_at_the_lower_attempt_limit(
        self, monkeypatch: MonkeyPatch, attempt_cap: int, expected_attempts: int
    ) -> None:
        client = self._mock_global_client(monkeypatch, TimeoutError("upstream timed out"), failures=None)

        with raises(TimeoutError, match="upstream timed out"):
            await nemo_gym.client.http_client.request(
                "POST", "http://slow-host:1/v1", _max_num_tries=attempt_cap, _max_connection_retries=3
            )

        assert client.request.await_count == expected_attempts

    async def test_request_connection_retries_unbounded_by_default(self, monkeypatch: MonkeyPatch) -> None:
        client = self._mock_global_client(monkeypatch, ClientOSError(), failures=4)
        response = await nemo_gym.client.http_client.request("POST", "http://flaky-host:1/v1")
        assert response is client.success_response
        assert client.request.await_count == 5

    @mark.parametrize("tracing_enabled", [False, True])
    async def test_request_retry_cap_bounds_internal_callers(
        self, monkeypatch: MonkeyPatch, tracing_enabled: bool
    ) -> None:
        monkeypatch.setattr(nemo_gym.client.http_client, "is_span_group_enabled", lambda _group: tracing_enabled)
        client = self._mock_global_client(monkeypatch, RuntimeError("boom"), failures=None)
        # Above MAX_NUM_TRIES, so the test can tell the cap apart from MAX_NUM_TRIES.
        cap = nemo_gym.client.http_client.MAX_NUM_TRIES + 2

        with raises(RuntimeError, match="boom"):
            await nemo_gym.client.http_client.request(
                "POST", "http://head:1/v1", _internal=True, _max_connection_retries=cap
            )

        assert client.request.await_count == cap
        assert nemo_gym.client.http_client.asyncio.sleep.await_count == cap - 1

    async def test_request_internal_callers_without_a_cap_retry_past_max_num_tries(
        self, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
    ) -> None:
        failures = nemo_gym.client.http_client.MAX_NUM_TRIES + 2
        client = self._mock_global_client(monkeypatch, RuntimeError("boom"), failures=failures)

        response = await nemo_gym.client.http_client.request("POST", "http://head:1/v1", _internal=True)

        assert response is client.success_response
        assert client.request.await_count == failures + 1
        assert capsys.readouterr().out == ""

    async def test_request_external_callers_stop_at_max_num_tries(
        self, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
    ) -> None:
        client = self._mock_global_client(monkeypatch, RuntimeError("boom"), failures=None)

        with raises(RuntimeError, match="boom"):
            await nemo_gym.client.http_client.request("POST", "http://model:1/v1")

        assert client.request.await_count == nemo_gym.client.http_client.MAX_NUM_TRIES
        out = capsys.readouterr().out
        assert out.count("Hit an exception while making a request") == nemo_gym.client.http_client.MAX_NUM_TRIES
