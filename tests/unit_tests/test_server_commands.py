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

    def test_status_command_check_health(self, monkeypatch: MonkeyPatch) -> None:
        cmd = StatusCommand()

        # success
        server_info = ServerInstanceDisplayConfig(
            pid=123,
            server_type="resources_servers",
            name="test_server",
            process_name="test_server",
            host="127.0.0.1",
            port=8000,
            url="http://127.0.0.1:8000",
            uptime_seconds=100,
            status="unknown_error",
            entrypoint="app.py",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        monkeypatch.setattr("requests.get", lambda *args, **kwargs: mock_response)

        status = cmd.check_health(server_info)
        assert status == "success"

        monkeypatch.setattr(
            "requests.get", lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.ConnectionError())
        )
        status = cmd.check_health(server_info)
        assert status == "connection_error"

        # timeout
        monkeypatch.setattr(
            "requests.get", lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.Timeout())
        )
        status = cmd.check_health(server_info)
        assert status == "timeout"

        # generic or other exception
        monkeypatch.setattr(
            "requests.get", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Something unexpected happened"))
        )
        status = cmd.check_health(server_info)
        assert status == "unknown_error"

        # no url
        server_info.url = None
        status = cmd.check_health(server_info)
        assert status == "unknown_error"

    def test_status_command_display_status_no_servers(self, monkeypatch: MonkeyPatch) -> None:
        text_trap = StringIO()
        monkeypatch.setattr("sys.stdout", text_trap)

        cmd = StatusCommand()
        cmd.display_status([])

        output = text_trap.getvalue()
        assert "No NeMo Gym servers found running." in output

    def test_status_command_display_status_with_servers(self, monkeypatch: MonkeyPatch) -> None:
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

        # Mock connection error
        mock_config = MagicMock()
        mock_config.host = "localhost"
        mock_config.port = 11000
        monkeypatch.setattr("nemo_gym.server_commands.ServerClient.load_head_server_config", lambda: mock_config)
        servers = cmd.discover_servers()
        assert servers == []
        output = text_trap.getvalue()
        assert "Could not connect to head server" in output
        assert "Is the head server running? Start it with: `ng_run`" in output

    def test_status_command_discover_servers(self, monkeypatch: MonkeyPatch) -> None:
        cmd = StatusCommand()

        mock_config = MagicMock()
        mock_config.host = "localhost"
        mock_config.port = 11000
        monkeypatch.setattr("nemo_gym.server_commands.ServerClient.load_head_server_config", lambda: mock_config)

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "process_name": "test_server_config",
                "server_type": "resources_servers",
                "name": "test_server",
                "host": "127.0.0.1",
                "port": 8000,
                "url": "http://127.0.0.1:8000",
                "entrypoint": "app.py",
                "pid": 12345,
                "start_time": 1000.0,
            }
        ]

        monkeypatch.setattr("requests.get", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr("nemo_gym.server_commands.time", lambda: 1100.0)  # 100 seconds later

        monkeypatch.setattr(cmd, "check_health", lambda s: "success")

        servers = cmd.discover_servers()

        assert len(servers) == 1
        assert servers[0].name == "test_server"
        assert servers[0].pid == 12345
        assert servers[0].port == 8000
        assert servers[0].uptime_seconds == 100.0
        assert servers[0].status == "success"

    def test_stop_command_stop_server_force(self, monkeypatch: MonkeyPatch) -> None:
        cmd = StopCommand()

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

        mock_child = MagicMock()
        mock_proc = MagicMock()
        mock_proc.children.return_value = [mock_child]

        monkeypatch.setattr("psutil.Process", lambda pid: mock_proc)

        result = cmd.stop_server(server_info, force=True)

        assert result["success"] is True
        assert result["method"] == "force"
        mock_child.kill.assert_called_once()
        mock_proc.kill.assert_called_once()
        mock_proc.wait.assert_called_once()

    def test_stop_command_stop_server_graceful(self, monkeypatch: MonkeyPatch) -> None:
        cmd = StopCommand()

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

        mock_child = MagicMock()
        mock_proc = MagicMock()
        mock_proc.children.return_value = [mock_child]
        mock_proc.wait.return_value = None

        monkeypatch.setattr("psutil.Process", lambda pid: mock_proc)

        result = cmd.stop_server(server_info, force=False)

        assert result["success"] is True
        assert result["method"] == "graceful"
        mock_child.send_signal.assert_called_once()
        mock_proc.send_signal.assert_called_once()

    def test_stop_command_stop_server_timeout_then_terminate(self, monkeypatch: MonkeyPatch) -> None:
        cmd = StopCommand()

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

        mock_child = MagicMock()
        mock_proc = MagicMock()
        mock_proc.children.return_value = [mock_child]

        mock_proc.wait.side_effect = [psutil.TimeoutExpired(1), None]

        monkeypatch.setattr("psutil.Process", lambda pid: mock_proc)

        result = cmd.stop_server(server_info, force=False)

        assert result["success"] is True
        assert result["method"] == "terminate"
        mock_child.terminate.assert_called_once()
        mock_proc.terminate.assert_called_once()

    def test_stop_command_stop_server_double_timeout_failure(self, monkeypatch: MonkeyPatch) -> None:
        cmd = StopCommand()

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

        mock_child = MagicMock()
        mock_proc = MagicMock()
        mock_proc.children.return_value = [mock_child]

        mock_proc.wait.side_effect = psutil.TimeoutExpired(1)

        monkeypatch.setattr("psutil.Process", lambda pid: mock_proc)

        result = cmd.stop_server(server_info, force=False)

        assert result["success"] is False
        assert result["method"] == "failed"
        assert "use --force" in result["message"]

    def test_stop_command_stop_server_no_such_process(self, monkeypatch: MonkeyPatch) -> None:
        cmd = StopCommand()

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

        monkeypatch.setattr("psutil.Process", lambda pid: (_ for _ in ()).throw(psutil.NoSuchProcess(99999)))

        result = cmd.stop_server(server_info, force=False)

        assert result["success"] is False
        assert result["method"] == "no_process"

    def test_stop_command_stop_server_access_denied(self, monkeypatch: MonkeyPatch) -> None:
        cmd = StopCommand()

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

        monkeypatch.setattr("psutil.Process", lambda pid: (_ for _ in ()).throw(psutil.AccessDenied()))

        result = cmd.stop_server(server_info, force=False)

        assert result["success"] is False
        assert result["method"] == "access_denied"

    def test_stop_command_display_results(self, monkeypatch: MonkeyPatch) -> None:
        text_trap = StringIO()
        monkeypatch.setattr("sys.stdout", text_trap)

        cmd = StopCommand()
        results = [
            {"success": True, "message": "Stopped server1"},
            {"success": True, "message": "Stopped server2"},
            {"success": False, "message": "Failed to stop server3"},
        ]

        cmd.display_results(results)

        output = text_trap.getvalue()
        assert "Stopping NeMo Gym servers" in output
        assert "✓" in output
        assert "✗" in output
        assert "2 of 3 servers stopped successfully, 1 failed" in output

    def test_stop_command_stop_server_force(self, monkeypatch: MonkeyPatch) -> None:
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

        # Mock when child process is already dead
        mock_child = MagicMock()
        mock_child.kill.side_effect = psutil.NoSuchProcess(99999)
        mock_proc = MagicMock()
        mock_proc.children.return_value = [mock_child]
        monkeypatch.setattr("psutil.Process", lambda pid: mock_proc)
        result = cmd.stop_server(server_info, force=True)
        assert result["success"] is True
        assert result["method"] == "force"
        mock_child.kill.assert_called_once()
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
