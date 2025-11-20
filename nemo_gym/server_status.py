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
from dataclasses import dataclass, field
from signal import SIGINT
from time import time
from typing import List

import requests
from devtools import pprint

from nemo_gym.server_utils import ServerClient, ServerInstanceDisplayConfig, ServerStatus


def stop_server(server_info: ServerProcessInfo, force: bool = False) -> dict:
    """Stop a single server process."""

    try:
        proc = psutil.Process(server_info.pid)

        if force:
            proc.kill()
            proc.wait(timeout=2)

            return {
                "server": server_info,
                "success": True,
                "method": "force",
                "message": f"Force stopped {server_info.name}",
            }
        else:
            # Graceful shutdown, then wait
            proc.send_signal(SIGINT)

            try:
                proc.wait(timeout=10)

                return {
                    "server_info": server_info,
                    "success": True,
                    "method": "graceful",
                    "message": f"Gracefully stopped {server_info.name}",
                }
            except psutil.TimeoutExpired:
                # Graceful didn't work, so terminate and wait
                proc.terminate()

                try:
                    proc.wait(timeout=5)

                    return {
                        "server": server_info,
                        "success": True,
                        "method": "terminate",
                        "message": f"Terminated {server_info.name} (graceful shutdown timed out)",
                    }
                except psutil.TimeoutExpired:
                    return {
                        "server": server_info,
                        "success": False,
                        "method": "failed",
                        "message": f"Failed to stop {server_info.name} - use --force",
                    }
    except psutil.NoSuchProcess:
        return {
            "server": server_info,
            "success": False,
            "method": "no_process",
            "message": f"{server_info.name} not found",
        }
    except psutil.AccessDenied:
        return {
            "server": server_info,
            "success": False,
            "method": "access_denied",
            "message": f"Access denied to stop {server_info.name} (PID: {server_info.pid})",
        }
    except Exception as e:
        return {
            "server": server_info,
            "success": False,
            "method": "error",
            "message": f"Error stopping {server_info.name}: {e}",
        }


class StatusCommand:
    """Main class to check server status"""

    def check_health(self, server_info: ServerInstanceDisplayConfig) -> ServerStatus:
        """Check if server is responding"""
        if not server_info.url:
            return "unknown_error"

        try:
            requests.get(server_info.url, timeout=2)
            return "success"
        except requests.exceptions.ConnectionError:
            return "connection_error"
        except requests.exceptions.Timeout:
            return "timeout"
        except Exception:
            return "unknown_error"

    def discover_servers(self) -> List[ServerInstanceDisplayConfig]:
        """Find all running NeMo Gym server processes"""

        try:
            head_server_config = ServerClient.load_head_server_config()
            head_url = f"http://{head_server_config.host}:{head_server_config.port}"

            response = requests.get(f"{head_url}/server_instances", timeout=5)
            response.raise_for_status()
            instances = response.json()

            servers = []
            current_time = time()

            for inst in instances:
                uptime = current_time - inst.get("start_time", current_time)
                server_info = ServerInstanceDisplayConfig(
                    process_name=inst["process_name"],
                    server_type=inst["server_type"],
                    name=inst["name"],
                    host=inst.get("host"),
                    port=inst.get("port"),
                    url=inst.get("url"),
                    entrypoint=inst.get("entrypoint"),
                    pid=inst.get("pid"),
                    uptime_seconds=uptime,
                    status="unknown_error",
                )
                server_info.status = self.check_health(server_info)
                servers.append(server_info)

            return servers

        except (requests.RequestException, ConnectionError) as e:
            print(f"""
Could not connect to head server: {e}
Is the head server running? Start it with: `ng_run`
            """)
            return []

    def display_status(self, servers: List[ServerInstanceDisplayConfig]) -> None:
        """Show server info in a table"""

        def format_uptime(uptime_seconds: float) -> str:
            """Format uptime in a human readable format"""
            minutes, seconds = divmod(uptime_seconds, 60)
            hours, minutes = divmod(minutes, 60)
            days, hours = divmod(hours, 24)
            return f"{int(days)}d {int(hours)}h {int(minutes)}m {seconds:.1f}s"

        if not servers:
            print("No NeMo Gym servers found running.")
            return

        print("\nNeMo Gym Server Status:\n")

        for i, server in enumerate(servers, 1):
            status_icon = "✓" if server.status == "success" else "✗"
            print(f"[{i}] {status_icon} {server.process_name} ({server.server_type}/{server.name})")
            display_dict = {
                "server_type": server.server_type,
                "name": server.name,
                "port": server.port,
                "pid": server.pid,
                "uptime_seconds": format_uptime(server.uptime_seconds),
            }
            pprint(display_dict)

        healthy_count = sum(1 for s in servers if s.status == "success")
        print(f"""
{len(servers)} servers found ({healthy_count} healthy, {len(servers) - healthy_count} unhealthy)
""")


@dataclass
class StopCommand:
    """Class to stop gym servers"""

    status_cmd: StatusCommand = field(default_factory=StatusCommand)

    def stop_all(self, force: bool = False) -> List[dict]:
        """Stop all running servers"""
        servers = self.status_cmd.discover_servers()
        return [stop_server(server, force) for server in servers]

    def stop_by_name(self, name: str, force: bool = False) -> List[dict]:
        """Stop a server by name"""
        servers = self.status_cmd.discover_servers()
        # TODO: handle multiple matching servers?
        matching = next((s for s in servers if s.name == name), None)

        if not matching:
            return [{"success": False, "message": f"No server found with name: {name}"}]

        return [stop_server(matching, force)]

    def stop_by_port(self, port: int, force: bool = False) -> List[dict]:
        """Stop a server on a specific port"""
        servers = self.status_cmd.discover_servers()
        matching = next((s for s in servers if s.port == port), None)

        if not matching:
            return [{"success": False, "message": f"No server found with port: {port}"}]

        return [stop_server(matching, force)]
