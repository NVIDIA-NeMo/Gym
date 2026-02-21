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
from io import StringIO
from unittest.mock import MagicMock

import requests
from pytest import MonkeyPatch

from nemo_gym.cli import ServerInstanceDisplayConfig
from nemo_gym.server_status import StatusCommand
from nemo_gym.server_utils import ServerClient


class TestServerStatus:
    def test_server_process_info_creation_sanity(self) -> None:
        ServerInstanceDisplayConfig(
            pid=12345,
            server_type="resources_server",
            name="test_server",
            process_name="test_server",
            host="127.0.0.1",
            port=8000,
            url="http://127.0.0.1:8000",
            uptime_seconds=100,
            status="success",
            entrypoint="test_server",
        )

    def test_display_status_no_servers(self, monkeypatch: MonkeyPatch) -> None:
        text_trap = StringIO()
        monkeypatch.setattr("sys.stdout", text_trap)

        cmd = StatusCommand()
        cmd.display_status([])

        output = text_trap.getvalue()
        assert "No NeMo Gym servers found running." in output

    def test_display_status_with_servers(self, monkeypatch: MonkeyPatch) -> None:
        text_trap = StringIO()
        monkeypatch.setattr("sys.stdout", text_trap)

        servers = [
            ServerInstanceDisplayConfig(
                pid=123,
                server_type="resources_servers",
                name="test_resource",
                process_name="test_resource_server",
                host="127.0.0.1",
                port=8000,
                url="http://127.0.0.1:8000",
                uptime_seconds=100,
                status="success",
                entrypoint="test_server",
            ),
            ServerInstanceDisplayConfig(
                pid=456,
                server_type="responses_api_models",
                name="test_model",
                process_name="test_model",
                host="127.0.0.1",
                port=8001,
                url="http://127.0.0.1:8001",
                uptime_seconds=200,
                status="connection_error",
                entrypoint="test_model",
            ),
        ]

        cmd = StatusCommand()
        cmd.display_status(servers)

        output = text_trap.getvalue()
        assert "2 servers found (1 healthy, 1 unhealthy)" in output
        assert "123" in output
        assert "456" in output
        assert "test_resource" in output
        assert "test_model" in output

    def test_check_health(self, monkeypatch: MonkeyPatch) -> None:
        cmd = StatusCommand()
        server_info = ServerInstanceDisplayConfig(
            pid=123,
            server_type="resources_servers",
            name="test_server",
            process_name="test_server",
            host=None,
            port=None,
            url=None,
            uptime_seconds=100,
            status="unknown_error",
            entrypoint="app.py",
        )

        # no url
        result = cmd.check_health(server_info)
        assert result == "unknown_error"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get = MagicMock(return_value=mock_response)
        monkeypatch.setattr(requests, "get", mock_get)

        # valid
        server_info.host = "127.0.0.1"
        server_info.port = 8000
        server_info.url = "http://127.0.0.1:8000"
        result = cmd.check_health(server_info)
        assert result == "success"
        mock_get.assert_called_once_with("http://127.0.0.1:8000", timeout=2)

        # errors
        mock_get = MagicMock(side_effect=requests.exceptions.ConnectionError())
        monkeypatch.setattr(requests, "get", mock_get)
        result = cmd.check_health(server_info)
        assert result == "connection_error"

        mock_get = MagicMock(side_effect=requests.exceptions.Timeout())
        monkeypatch.setattr(requests, "get", mock_get)
        result = cmd.check_health(server_info)
        assert result == "timeout"

        mock_get = MagicMock(side_effect=ValueError("Unexpected error"))
        monkeypatch.setattr(requests, "get", mock_get)
        result = cmd.check_health(server_info)
        assert result == "unknown_error"

    def test_discover_servers(self, monkeypatch: MonkeyPatch) -> None:
        mock_instances = [
            {
                "process_name": "test_resources_server",
                "server_type": "resources_servers",
                "name": "test_resource",
                "host": "127.0.0.1",
                "port": 8000,
                "url": "http://127.0.0.1:8000",
                "entrypoint": "app.py",
                "pid": 12345,
                "start_time": 1000.0,
            },
            {
                "process_name": "test_model_server",
                "server_type": "responses_api_models",
                "name": "test_model",
                "host": "127.0.0.1",
                "port": 8001,
                "url": "http://127.0.0.1:8001",
                "entrypoint": "model.py",
                "pid": 12346,
                "start_time": 2000.0,
            },
        ]

        mock_head_config = MagicMock()
        mock_head_config.host = "127.0.0.1"
        mock_head_config.port = 11000

        monkeypatch.setattr(ServerClient, "load_head_server_config", lambda: mock_head_config)

        mock_response = MagicMock()
        mock_response.json.return_value = mock_instances

        mock_get = MagicMock(
            side_effect=[
                mock_response,
                MagicMock(status_code=200),
                requests.exceptions.ConnectionError(),
            ]
        )
        monkeypatch.setattr(requests, "get", mock_get)

        mock_time = MagicMock(return_value=10000.0)
        monkeypatch.setattr("nemo_gym.server_status.time", mock_time)

        cmd = StatusCommand()
        servers = cmd.discover_servers()

        assert len(servers) == 2, "Should find 2 NeMo Gym servers"

        assert servers[0].pid == 12345
        assert servers[0].name == "test_resource"
        assert servers[0].server_type == "resources_servers"
        assert servers[0].host == "127.0.0.1"
        assert servers[0].port == 8000
        assert servers[0].status == "success"
        assert servers[0].uptime_seconds == 9000.0

        assert servers[1].pid == 12346
        assert servers[1].name == "test_model"
        assert servers[1].server_type == "responses_api_models"
        assert servers[1].host == "127.0.0.1"
        assert servers[1].port == 8001
        assert servers[1].status == "connection_error"
        assert servers[1].uptime_seconds == 8000.0

        assert mock_get.call_count == 3
        first_call = mock_get.call_args_list[0]
        assert first_call[0][0] == "http://127.0.0.1:11000/server_instances"

    def test_discover_servers_head_server_down(self, monkeypatch: MonkeyPatch, capsys) -> None:
        mock_head_config = MagicMock()
        mock_head_config.host = "127.0.0.1"
        mock_head_config.port = 11000

        from nemo_gym.server_utils import ServerClient

        monkeypatch.setattr(ServerClient, "load_head_server_config", lambda: mock_head_config)

        mock_get = MagicMock(side_effect=requests.exceptions.ConnectionError("Connection refused"))
        monkeypatch.setattr(requests, "get", mock_get)

        cmd = StatusCommand()
        servers = cmd.discover_servers()

        assert len(servers) == 0
        captured = capsys.readouterr()
        assert "Could not connect to head server" in captured.out
        assert "ng_run" in captured.out
