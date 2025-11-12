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

    def test_parse_server_info_missing_env_vars(self) -> None:
        mock_proc = MagicMock()
        mock_proc.info = {"pid": 123, "create_time": 1000.0}

        result = parse_server_info(mock_proc, ["python", "test_server.py"], {})
        assert result is None

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
