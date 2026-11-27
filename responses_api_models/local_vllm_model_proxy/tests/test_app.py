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
import socket
import threading
import time
from unittest.mock import MagicMock

import pytest

import responses_api_models.local_vllm_model_proxy.app
from nemo_gym.client import ServerClient
from responses_api_models.local_vllm_model_proxy.app import (
    LocalVLLMModelProxyServer,
    LocalVLLMModelProxyServerConfig,
)


class TestApp:
    def _setup_server(self, **overrides):
        config = LocalVLLMModelProxyServerConfig(
            host="0.0.0.0",
            port=8081,
            entrypoint="",
            name="",
            return_token_id_information=False,
            uses_reasoning_parser=True,
            model_server={"type": "responses_api_models", "name": "dummy ref"},
            **overrides,
        )
        return LocalVLLMModelProxyServer(
            config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={})
        )

    def test_sanity(self) -> None:
        self._setup_server()

    def test_setup_webserver_sanity(self, monkeypatch) -> None:
        server = self._setup_server()

        sleep_mock = MagicMock()
        monkeypatch.setattr(responses_api_models.local_vllm_model_proxy.app, "sleep", sleep_mock)

        server.server_client.poll_for_status.side_effect = ["error", "error", "success"]
        server.server_client.global_config_dict = {}

        get_first_server_config_dict_mock = MagicMock()
        monkeypatch.setattr(
            responses_api_models.local_vllm_model_proxy.app,
            "get_first_server_config_dict",
            get_first_server_config_dict_mock,
        )

        json_mock = MagicMock()
        json_mock.json.return_value = {
            "base_url": ["abcd", "defg"],
            "api_key": "my api key",  # pragma: allowlist secret
            "model": "my model",
        }
        requests_mock = MagicMock()
        requests_mock.get.return_value = json_mock
        monkeypatch.setattr(responses_api_models.local_vllm_model_proxy.app, "requests", requests_mock)

        server.setup_webserver()

        assert sleep_mock.call_count == 2

        expected_model = "my model"
        actual_model = server.config.model
        assert expected_model == actual_model

        assert len(server._clients) == 2

    def test_a_model_server_that_never_comes_up_fails_startup_within_the_limit(self, monkeypatch) -> None:
        server = self._setup_server(model_server_wait_timeout_seconds=25)
        # Each poll "sleeps" 10 s on a fake clock, so the 25 s limit ends the wait after three polls. If the limit were
        # ignored the polls would end on their own, with a success, and the raises below would fail.
        now = [0.0]
        monkeypatch.setattr(responses_api_models.local_vllm_model_proxy.app, "monotonic", lambda: now[0])
        monkeypatch.setattr(
            responses_api_models.local_vllm_model_proxy.app,
            "sleep",
            lambda seconds: now.__setitem__(0, now[0] + seconds),
        )
        server.server_client.poll_for_status.side_effect = ["error"] * 5 + ["success"]

        with pytest.raises(TimeoutError, match="was not up within 25"):
            server.setup_webserver()

        assert server.server_client.poll_for_status.call_count == 3  # at 0 s, 10 s and 20 s

    def test_a_model_server_that_comes_up_only_after_the_deadline_is_not_accepted(self, monkeypatch) -> None:
        server = self._setup_server(model_server_wait_timeout_seconds=5)
        now = [0.0]
        monkeypatch.setattr(responses_api_models.local_vllm_model_proxy.app, "monotonic", lambda: now[0])
        # The sleep is capped at the 5 s that remain, not the usual 10 s, so the next poll never happens after the limit.
        monkeypatch.setattr(
            responses_api_models.local_vllm_model_proxy.app,
            "sleep",
            lambda seconds: now.__setitem__(0, now[0] + seconds),
        )
        server.server_client.poll_for_status.side_effect = ["error", "success"]

        with pytest.raises(TimeoutError, match="was not up within 5"):
            server.setup_webserver()

        assert server.server_client.poll_for_status.call_count == 1

    def test_a_poll_that_succeeds_after_the_deadline_is_not_accepted_and_each_poll_fits_the_budget(
        self, monkeypatch
    ) -> None:
        server = self._setup_server(model_server_wait_timeout_seconds=1)
        now = [0.0]
        monkeypatch.setattr(responses_api_models.local_vllm_model_proxy.app, "monotonic", lambda: now[0])
        monkeypatch.setattr(responses_api_models.local_vllm_model_proxy.app, "sleep", lambda seconds: None)
        timeouts = []

        def slow_success(server_name, *, timeout_seconds, total_deadline):
            assert total_deadline  # a limit is set, so the poll must end at its budget in total
            timeouts.append(timeout_seconds)
            now[0] += 5  # the answer comes after the 1 s limit
            return "success"

        server.server_client.poll_for_status.side_effect = slow_success

        with pytest.raises(TimeoutError, match="was not up within 1"):
            server.setup_webserver()

        assert timeouts == [1]  # the poll was limited to the budget that remained, not its usual 5 s

    BODY = b'{"base_url": ["abcd"], "api_key": "key", "model": "m"}'  # pragma: allowlist secret

    @staticmethod
    def _peer(pieces: list[tuple[bytes, float]]):
        """A loopback server that answers its first request by sending each piece after its delay, then closes.

        Returns its address and a function that reports whether the whole answer was sent."""
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        finished = threading.Event()

        def serve() -> None:
            connection, _ = listener.accept()
            with connection, listener:
                connection.recv(65536)
                try:
                    for piece, delay in pieces:
                        time.sleep(delay)
                        connection.sendall(piece)
                    finished.set()
                except OSError:
                    pass  # The client gave up.

        threading.Thread(target=serve, daemon=True).start()
        return f"http://127.0.0.1:{listener.getsockname()[1]}", finished

    @classmethod
    def _reply(cls, body: bytes) -> list[tuple[bytes, float]]:
        head = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body)
        return [(head, 0), (body, 0)]

    def _serve_config(self, monkeypatch, pieces, **overrides):
        server = self._setup_server(**overrides)
        server.server_client.poll_for_status.side_effect = ["success"]
        server.server_client.global_config_dict = {}
        url, finished = self._peer(pieces)
        server.server_client._build_server_base_url.return_value = url
        monkeypatch.setattr(
            responses_api_models.local_vllm_model_proxy.app, "get_first_server_config_dict", MagicMock()
        )
        return server, finished

    def _setup_within(self, server, seconds: float):
        """Run `setup_webserver` under an outer guard that does not rely on the bound being tested."""
        outcome = {}

        def run() -> None:
            try:
                server.setup_webserver()
            except BaseException as error:
                outcome["error"] = error

        started = time.monotonic()
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(seconds)
        assert not worker.is_alive(), f"setup_webserver was still running after {seconds} s"
        return outcome.get("error"), time.monotonic() - started

    def test_without_a_limit_a_healthy_config_response_populates_the_clients(self, monkeypatch) -> None:
        server, _ = self._serve_config(monkeypatch, self._reply(self.BODY))

        error, _ = self._setup_within(server, 10)

        assert error is None
        assert (server.config.base_url, server.config.api_key, server.config.model) == (["abcd"], "key", "m")

    def test_a_healthy_config_response_within_the_limit_populates_the_clients(self, monkeypatch) -> None:
        server, _ = self._serve_config(monkeypatch, self._reply(self.BODY), model_server_wait_timeout_seconds=30)

        error, _ = self._setup_within(server, 10)

        assert error is None
        assert (server.config.base_url, server.config.api_key, server.config.model) == (["abcd"], "key", "m")

    def test_a_config_body_that_dribbles_past_the_limit_ends_at_the_limit(self, monkeypatch) -> None:
        # Valid JSON, one byte every 50 ms (about 2.5 s in all), against a 0.2 s limit.
        head = self._reply(self.BODY)[0]
        pieces = [head] + [(self.BODY[i : i + 1], 0.05) for i in range(len(self.BODY))]
        server, finished = self._serve_config(monkeypatch, pieces, model_server_wait_timeout_seconds=0.2)

        error, elapsed = self._setup_within(server, 5)

        assert isinstance(error, TimeoutError) and "was not received within 0.2" in str(error)
        assert elapsed < 1.5 and not finished.is_set()

    def test_response_headers_that_dribble_past_the_limit_end_at_the_limit(self, monkeypatch) -> None:
        head = b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(self.BODY)
        pieces = [(head[i : i + 1], 0.05) for i in range(len(head))] + [(self.BODY, 0)]
        server, finished = self._serve_config(monkeypatch, pieces, model_server_wait_timeout_seconds=0.2)

        error, elapsed = self._setup_within(server, 5)

        assert isinstance(error, TimeoutError) and "was not received within 0.2" in str(error)
        assert elapsed < 1.5 and not finished.is_set()

    def test_a_wait_limit_of_zero_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._setup_server(model_server_wait_timeout_seconds=0)
