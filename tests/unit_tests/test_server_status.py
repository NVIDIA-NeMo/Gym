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

import psutil
import requests
from pytest import MonkeyPatch

from nemo_gym.global_config import NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME
from nemo_gym.server_status import ServerProcessInfo, StatusCommand, parse_server_info


class TestServerStatus:
    def test_server_process_info_creation_sanity(self) -> None:
        ServerProcessInfo(
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

    def test_parse_server_info_invalid_config(self, capsys) -> None:
        mock_proc = MagicMock()
        mock_proc.info = {"pid": 123, "create_time": 1000.0}

        env = {}
        result = parse_server_info(mock_proc, ["python", "test_server.py"], env)
        assert result is None

        env = {
            NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME: "test_server",
            NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME: "",
        }
        result = parse_server_info(mock_proc, ["python", "test_server.py"], env)
        assert result is None

        env = {
            NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME: "test_server",
            NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME: """
test_server:
  resources_servers: {}
""",
        }
        result = parse_server_info(mock_proc, ["python", "test_server.py"], env)
        assert result is None

        captured = capsys.readouterr()
        assert "Error parsing PID 123" in captured.out
        assert "IndexError" in captured.out

    def test_parse_server_info_valid(self) -> None:
        config_yaml = """
test_resources_server:
  resources_servers:
    test_resource:
      host: 127.0.0.1
      port: 8000
      entrypoint: app.py
"""

        mock_proc = MagicMock()
        mock_proc.info = {"pid": 123, "create_time": 1000.0}

        env = {
            NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME: "test_resources_server",
            NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME: config_yaml,
        }

        result = parse_server_info(mock_proc, ["python", "app.py"], env)

        assert result is not None
        assert result.pid == 123
        assert result.name == "test_resource"
        assert result.server_type == "resources_servers"
        assert result.host == "127.0.0.1"
        assert result.port == 8000

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
            ServerProcessInfo(
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
            ServerProcessInfo(
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
        server_info = ServerProcessInfo(
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
        mock_proc1 = MagicMock()

        # resource server (healthy)
        mock_proc1.info = {
            "pid": 12345,
            "name": "python",
            "cmdline": ["python", "app.py"],
            "create_time": 1000.0,
            "environ": {
                NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME: "test_resources_server",
                NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME: """
test_resources_server:
  resources_servers:
    test_resource:
      host: 127.0.0.1
      port: 8000
      entrypoint: app.py
""",
            },
        }

        # model server (unhealthy)
        mock_proc2 = MagicMock()
        mock_proc2.info = {
            "pid": 12346,
            "name": "python",
            "cmdline": ["python", "model.py"],
            "create_time": 2000.0,
            "environ": {
                NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME: "test_model_server",
                NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME: """
test_model_server:
  responses_api_models:
    test_model:
      host: 127.0.0.1
      port: 8001
      entrypoint: model.py
""",
            },
        }

        mock_proc3 = MagicMock()
        mock_proc3.info = {
            "pid": 99999,
            "name": "python",
            "cmdline": ["python", "other.py"],
            "create_time": 3000.0,
            "environ": {},
        }

        mock_proc4 = MagicMock()
        mock_proc4.info.get.side_effect = psutil.NoSuchProcess(99998)

        mock_proc5 = MagicMock()
        mock_proc5.info.get.side_effect = psutil.AccessDenied(99997)

        mock_process_iter = MagicMock(return_value=iter([mock_proc1, mock_proc2, mock_proc3, mock_proc4, mock_proc5]))
        monkeypatch.setattr("psutil.process_iter", mock_process_iter)

        mock_get = MagicMock(side_effect=[MagicMock(status_code=200), requests.exceptions.ConnectionError()])
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

        mock_process_iter.assert_called_once_with(["pid", "name", "cmdline", "create_time", "environ"])
