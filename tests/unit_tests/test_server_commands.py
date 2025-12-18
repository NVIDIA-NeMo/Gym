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

from nemo_gym.cli import ServerInstanceDisplayConfig
from nemo_gym.server_commands import StatusCommand, StopCommand


class TestServerCommands:
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

    def test_stop_server_force(self, monkeypatch: MonkeyPatch) -> None:
        server_info = ServerInstanceDisplayConfig(
            pid=99999,
            server_type="resources_servers",
            name="test_server",
            process_name="test_server",
            host="127.0.0.1",
            port=8000,
            url="http://127.0.0.1:8000",
            uptime_seconds=100,
            status="success",
            entrypoint="app.py",
        )

        mock_proc = MagicMock()
        mock_process_cls = MagicMock(return_value=mock_proc)
        monkeypatch.setattr("psutil.Process", mock_process_cls)

        cmd = StopCommand()
        result = cmd.stop_server(server_info, force=True)

        assert result["success"] is True
        assert result["method"] == "force"
        mock_proc.kill.assert_called_once()

    def test_stop_command_stop_all(self, monkeypatch: MonkeyPatch) -> None:
        cmd = StopCommand()

        # No servers running
        monkeypatch.setattr(cmd.status_cmd, "discover_servers", lambda: [])
        results = cmd.stop_all(force=False)

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "No servers found" in results[0]["message"]

        # With running servers
        servers = [
            ServerInstanceDisplayConfig(
                pid=123,
                server_type="resources_servers",
                name="server1",
                process_name="server1",
                host="127.0.0.1",
                port=8000,
                url="http://127.0.0.1:8000",
                uptime_seconds=100,
                status="success",
                entrypoint="app.py",
            )
        ]

        mock_stop_result = {
            "server": servers[0],
            "success": True,
            "method": "graceful",
            "message": "Stopped server1",
        }

        monkeypatch.setattr(cmd.status_cmd, "discover_servers", lambda: servers)
        monkeypatch.setattr(cmd, "stop_server", lambda s, f: mock_stop_result)

        results = cmd.stop_all(force=False)

        assert len(results) == 1
        assert results[0]["success"] is True

    def test_stop_command_stop_by_name(self, monkeypatch: MonkeyPatch) -> None:
        cmd = StopCommand()

        servers = [
            ServerInstanceDisplayConfig(
                pid=123,
                server_type="resources_servers",
                name="test_server",
                process_name="test_server",
                host="127.0.0.1",
                port=8000,
                url="http://127.0.0.1:8000",
                uptime_seconds=100,
                status="success",
                entrypoint="app.py",
            )
        ]

        # Found by name
        mock_stop_result = {
            "server": servers[0],
            "success": True,
            "method": "graceful",
            "message": "Stopped test_server",
        }

        monkeypatch.setattr(cmd.status_cmd, "discover_servers", lambda: servers)
        monkeypatch.setattr(cmd, "stop_server", lambda s, f: mock_stop_result)

        results = cmd.stop_by_name("test_server", force=False)

        assert len(results) == 1
        assert results[0]["success"] is True

        # Not found by name
        monkeypatch.setattr(cmd.status_cmd, "discover_servers", lambda: [])
        results = cmd.stop_by_name("nonexistent", force=False)

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "No server found" in results[0]["message"]

    def test_stop_command_stop_by_port(self, monkeypatch: MonkeyPatch) -> None:
        cmd = StopCommand()

        servers = [
            ServerInstanceDisplayConfig(
                pid=123,
                server_type="resources_servers",
                name="test_server",
                process_name="test_server",
                host="127.0.0.1",
                port=8000,
                url="http://127.0.0.1:8000",
                uptime_seconds=100,
                status="success",
                entrypoint="app.py",
            )
        ]

        # Found on port
        mock_stop_result = {
            "server": servers[0],
            "success": True,
            "method": "graceful",
            "message": "Stopped server on port 8000",
        }

        monkeypatch.setattr(cmd.status_cmd, "discover_servers", lambda: servers)
        monkeypatch.setattr(cmd, "stop_server", lambda s, f: mock_stop_result)

        results = cmd.stop_by_port(8000, force=False)

        assert len(results) == 1
        assert results[0]["success"] is True

        # Not found on port
        monkeypatch.setattr(cmd.status_cmd, "discover_servers", lambda: [])
        results = cmd.stop_by_port(9999, force=False)

        assert len(results) == 1
        assert results[0]["success"] is False
        assert "No server found" in results[0]["message"]
